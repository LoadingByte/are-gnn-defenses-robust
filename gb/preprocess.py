from typing import Optional

import numba
import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.torchext import sum, mul, div, sp_new_values, sp_eye, sp_svd
from gb.typing import Float, Int

patch_typeguard()
batch = None
nodes = None
dim_1 = None
dim_2 = None


@typechecked
def add_loops(A: TensorType["batch": ..., "nodes", "nodes"]) -> TensorType["batch": ..., "nodes", "nodes"]:
    """Adds ones to the diagonal of each adjacency matrix."""

    N = A.shape[-1]
    if A.is_sparse:
        if A.ndim == 2:
            return A + sp_eye(N, device=A.device)
        elif A.ndim == 3:
            return A + sp_eye(N, batch_size=A.shape[0], device=A.device)
        else:
            raise ValueError("Sparse adjacency matrices with more than one batch dimension are not supported")
    else:
        return A + torch.eye(N, device=A.device)


@typechecked
def gcn_norm(A: TensorType["batch": ..., "nodes", "nodes"], sqrt: bool = True) \
        -> TensorType["batch": ..., "nodes", "nodes"]:
    """Computes D^{-1/2} @ A @ D^{-1/2}, with D being the diagonal degree matrix."""

    deg = sum(A, dim=-1, dense=True)
    # Note: We assign an "epsilon" degree to vertices which actually have degree 0.
    # This is just to avoid dividing by zero in the next step. However, in
    # these cases, the dividends are zero anyway (since a vertex with degree
    # 0 has no edges), hence we do not alter the resulting matrix A_hat.
    deg.clamp_min_(1e-5)
    if A.is_sparse:
        # A coalesced matrix will be required for the main normalization step below.
        A = A.coalesce()
        if A.ndim == 2:
            prods = deg[A.indices()].prod(dim=0)
        else:
            # The following code also works for sparse matrices with multiple batch dimensions; however, since
            # sp.mm() doesn't support batching, torch.bmm() only supports one batch dimension, and there's
            # seemingly no replacement at the moment, we cannot use this functionality (yet).
            base_tuple = tuple(A.indices()[:-2])
            prods = deg[base_tuple + (A.indices()[-2],)] * deg[base_tuple + (A.indices()[-1],)]
        if sqrt:
            prods.sqrt_()
        return sp_new_values(A, A.values() / prods)
    else:
        if sqrt:
            deg.sqrt_()
        return A / deg[..., :, None] / deg[..., None, :]


@typechecked
def low_rank(mat: TensorType["batch": ..., "dim_1", "dim_2"], rank: Int) \
        -> TensorType["batch": ..., "dim_1", "dim_2", torch.strided]:
    """Note: Assumes that the input matrix has a lot of 0 entries."""

    U, S, VT = sp_svd(mat, rank)
    return U @ (S[..., None] * VT)  # equiv. to U @ S @ VT


@typechecked
def personalized_page_rank(
        A: TensorType["batch": ..., "nodes", "nodes"],
        teleport_proba: Float,
        neighbors: Optional[Int],
        approx: bool = False
) -> TensorType["batch": ..., "nodes", "nodes"]:
    """
    As proposed in the paper "Diffusion Improves Graph Learning".
    """

    N = A.shape[-1]
    if not approx:
        if A.is_sparse:
            A = A.to_dense()
        A = gcn_norm(add_loops(A))
        A = teleport_proba * torch.inverse(torch.eye(N, device=A.device) - (1 - teleport_proba) * A)
        if neighbors is None:
            return A / A.sum(dim=-1).clamp_min(1e-5)  # clamping trick from gcn_norm()
        else:
            top_values, top_indices = A.topk(neighbors, dim=-1)
            top_values = top_values / top_values.sum(dim=-1).clamp_min(1e-5)[:, None]  # clamping trick again
            return torch.zeros_like(A).scatter(-1, top_indices, top_values)
    else:
        if neighbors is None:
            raise ValueError("PPR approximation does not support topk=None")
        if not A.is_sparse:
            A = A.to_sparse()
        A = add_loops(A).coalesce()
        deg = sum(A, dim=-1, dense=True)
        A_ppr = []
        for A_sub, deg_sub in [(A, deg)] if A.ndim == 2 else zip(A, deg):
            A_sub_csr = A_sub.to_sparse_csr()
            neighbors, weights = _calc_ppr_topk_parallel(
                A_sub_csr.crow_indices().cpu().numpy(), A_sub_csr.col_indices().cpu().numpy(), deg_sub.cpu().numpy(),
                numba.float32(teleport_proba), 1e-4, N, neighbors
            )
            A_ppr.append(torch.sparse_coo_tensor(
                indices=torch.vstack([
                    torch.arange(N).repeat_interleave(torch.tensor([len(ns) for ns in neighbors])),
                    torch.from_numpy(np.concatenate(neighbors))
                ]),
                values=torch.from_numpy(np.concatenate(weights)),
                size=(N, N), device=A.device
            ))
        A_ppr = A_ppr[0] if A.ndim == 2 else torch.stack(A_ppr).coalesce()
        # GCN-like normalization
        deg_sqrt = deg.sqrt()
        A_ppr = div(mul(A_ppr, deg_sqrt[..., None]), deg_sqrt.clamp_min(1e-5)[..., None, :])  # clamping trick again
        # Row normalization
        return div(A_ppr, sum(A_ppr, dim=-1, dense=True).clamp_min(1e-5)[..., None])  # clamping trick again


@numba.njit(parallel=True)
def _calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, N, topk):
    js = [np.zeros(0, dtype=np.int64)] * N
    vals = [np.zeros(0, dtype=np.float32)] * N
    for i in numba.prange(N):
        j, val = _calc_ppr_node(numba.int64(i), indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


@numba.njit(cache=True, locals={"_val": numba.float32, "res": numba.float32, "res_vnode": numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())
