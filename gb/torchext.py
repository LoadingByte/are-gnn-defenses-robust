import multiprocessing
import os
from typing import Union, Optional, List, Tuple, Callable

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from threadpoolctl import threadpool_limits
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.typing import Int

patch_typeguard()
batch = None
dim_1 = None
dim_2 = None
k = None


@typechecked
def matmul(t1: TensorType, t2: TensorType[torch.strided]) -> TensorType:
    if t1.is_sparse:
        if t1.ndim == 2:
            return torch.sparse.mm(t1, t2)
        elif t1.ndim == 3:
            if t2.ndim == 2:
                # We have also tried reshaping mat1 to a 2D matrix, doing a 2D matmul, and then reshaping the
                # result back to a 3D tensor, but this turned out to be faster. Also, using expand() instead of
                # tile() doesn't make it faster.
                t2 = t2.tile(t1.shape[0], 1, 1)
            return torch.bmm(t1, t2)
        else:
            # https://pytorch.org/docs/stable/sparse.html#supported-linear-algebra-operations
            raise ValueError(f"Sparse-dense matrix mul is only only supported with at most one batch dimension")
    else:
        return t1 @ t2


@typechecked
def add_matmul(t1: TensorType[torch.strided], t2: TensorType, t3: TensorType[torch.strided]) -> TensorType:
    if t1.ndim == 2 and t2.ndim == 2 and t3.ndim == 2:
        return torch.addmm(t1, t2, t3)
    elif not t2.is_sparse and t2.ndim == 3 and t3.ndim == 3:
        return torch.baddbmm(t1, t2, t3)
    else:
        return t1 + matmul(t2, t3)


@typechecked
def sum(t: TensorType, dim: Int, dense: bool = False) -> TensorType:
    return (torch.sparse.sum(t, dim).to_dense() if dense else torch.sparse.sum(t, dim)) if t.is_sparse else t.sum(dim)


@typechecked
def softmax(t: TensorType, dim: Int) -> TensorType:
    return torch.sparse.softmax(t, dim=(dim if dim >= 0 else t.ndim + dim)) if t.is_sparse else t.softmax(dim=dim)


@typechecked
def abs(t: TensorType) -> TensorType:
    return _operate_on_nonzero_elems(torch.abs, t, op_requires_coalesced=True, op_keeps_0=True)


@typechecked
def neq0(t: TensorType) -> TensorType:
    return _operate_on_nonzero_elems(lambda elems: elems != 0, t, op_requires_coalesced=True, op_keeps_0=True)


@typechecked
def zero_keeping_exp(t: TensorType) -> TensorType:
    return _operate_on_nonzero_elems(torch.exp, t, op_requires_coalesced=True, op_keeps_0=False)


@typechecked
def _operate_on_nonzero_elems(
        op: Callable[[TensorType[torch.strided]], TensorType[torch.strided]],
        t: TensorType,
        op_requires_coalesced: bool,
        op_keeps_0: bool
) -> TensorType:
    if t.is_sparse:
        if op_requires_coalesced and not t.is_coalesced():
            raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
        return sp_new_values(t, _operate_on_nonzero_elems(op, t._values(), False, op_keeps_0))
    else:
        if op_keeps_0:
            return op(t)
        else:
            tr = t.clone()
            idx = tuple(tr.nonzero().T)
            tr[idx] = op(tr[idx])
            return tr


@typechecked
def mul(t1: TensorType, t2: TensorType) -> TensorType:
    return _combine_elem_wise(torch.mul, t1, t2, requires_coalesced=False)


@typechecked
def div(t1: TensorType, t2: TensorType) -> TensorType:
    return _combine_elem_wise(torch.div, t1, t2, requires_coalesced=False)


@typechecked
def _combine_elem_wise(
        op: Callable[[TensorType, TensorType], TensorType],
        t1: TensorType,
        t2: TensorType,
        requires_coalesced: bool
) -> TensorType:
    t1_sp = t1.is_sparse
    t2_sp = t2.is_sparse
    if t1_sp and not t2_sp:
        if requires_coalesced and not t1.is_coalesced():
            raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
        t2_values = t2.broadcast_to(t1.shape)[tuple(t1._indices())]
        return sp_new_values(t1, op(t1._values(), t2_values))
    elif not t1_sp and t2_sp:
        if requires_coalesced and not t2.is_coalesced():
            raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
        t1_values = t1.broadcast_to(t2.shape)[tuple(t2._indices())]
        return sp_new_values(t2, op(t1_values, t2._values()))
    else:
        return op(t1, t2)


@typechecked
def sp_new_values(t: TensorType[torch.sparse_coo], values: TensorType[torch.strided]) -> TensorType[torch.sparse_coo]:
    out = torch.sparse_coo_tensor(t._indices(), values, t.shape)
    # If the input tensor was coalesced, the output one will be as well since we don't modify the indices.
    if t.is_coalesced():
        with torch.no_grad():
            out._coalesced_(True)
    return out


@typechecked
def sp_tile(t: TensorType[torch.sparse_coo], n: Int) -> TensorType[torch.sparse_coo]:
    nnz = t._values().shape[0]
    tiled = torch.sparse_coo_tensor(
        indices=torch.vstack([torch.arange(n, device=t.device).repeat_interleave(nnz), t._indices().tile(n)]),
        values=t._values().tile(n),
        size=(n, *t.shape)
    )
    # From construction, we know that if the original tensor was coalesced, the tiled one will be as well.
    if t.is_coalesced():
        with torch.no_grad():
            tiled._coalesced_(True)
    return tiled


@typechecked
def sp_diag(values: TensorType["batch": ..., "dim_1", torch.strided]) \
        -> TensorType["batch": ..., "dim_1", "dim_1", torch.sparse_coo]:
    shape = (*values.shape, values.shape[-1])
    if values.ndim == 1:
        return torch.sparse_coo_tensor(
            indices=torch.arange(values.shape[0], device=values.device).tile(2, 1),
            values=values,
            size=shape
        )
    else:
        return torch.sparse_coo_tensor(
            indices=torch.vstack([
                torch.arange(values.shape[0], device=values.device).repeat_interleave(values.shape[1]),
                torch.arange(values.shape[1], device=values.device).tile(2, values.shape[0])
            ]),
            values=values.flatten(),
            size=shape
        )


@typechecked
def sp_eye(mat_size: Int, batch_size: Optional[Int] = None, device: Optional[torch.device] = None) \
        -> TensorType["batch": ..., "dim_1", "dim_1", torch.sparse_coo]:
    if batch_size is None:
        return sp_diag(torch.ones(mat_size, device=device))
    else:
        return sp_diag(torch.ones((batch_size, mat_size), device=device))


@typechecked
def sp_to_scipy(mat: TensorType["batch": ..., "dim_1", "dim_2"]) -> Union[coo_matrix, List[coo_matrix]]:
    # First convert to sparse if necessary and then transfer to the CPU for two reasons:
    #  - to_sparse() is quicker on the GPU.
    #  - Transferring a lightweight sparse matrix to the CPU is of course quicker than transferring a dense matrix.
    if not mat.is_sparse:
        mat = mat.to_sparse()
    mat = mat.cpu()
    if mat.ndim == 2:
        return coo_matrix((mat._values().numpy(), mat._indices().numpy()), mat.shape)
    else:
        # Note: We employ the following custom code to get sub-matrices because directly getting them using mat[i] or an
        # iterator over mat turns out to be slower.
        mat_values = mat._values().numpy()
        mat_indices = mat._indices().numpy()
        # If the first dimension's indices are not in ascending order, sort such that they are. We use the "stable" sort
        # algorithm because we expect the indices to be partially ordered, which is a bad situation for quicksort.
        # Note that this custom sorting here turns out to have more stable performance than coalescing the sparse matrix
        # either on the GPU or on the CPU.
        if not np.all(mat_indices[0, :-1] <= mat_indices[0, 1:]):
            sorter = np.argsort(mat_indices[0], kind="stable")
            mat_values = mat_values[sorter]
            mat_indices = mat_indices[:, sorter]
        scipy_list = []
        low = 0
        for b in range(mat.shape[0]):
            high = np.searchsorted(mat_indices[0], b + 1)
            scipy_list.append(coo_matrix((mat_values[low:high], mat_indices[1:, low:high]), mat.shape[1:]))
            low = high
        return scipy_list


@typechecked
def sp_svd(mat: TensorType["batch": ..., "dim_1", "dim_2"], limit: Int) \
        -> Tuple[
            TensorType["batch": ..., "dim_1", "k", torch.strided],
            TensorType["batch": ..., "k", torch.strided],
            TensorType["batch": ..., "k", "dim_2", torch.strided]
        ]:
    dev = mat.device
    # Compute the top-k singular values and vectors using scipy's ARPACK, which is 15 times faster than PyTorch.
    # Note: We have found that even for our adjacency matrices, svds() is faster than eigenproblem scipy functions.
    if mat.ndim == 2:
        U, S, VT = svds(sp_to_scipy(mat), limit)
        # Transfer the singular values and vectors back to the GPU.
        return torch.from_numpy(U).to(dev), torch.from_numpy(S).to(dev), torch.from_numpy(VT).to(dev)
    else:
        global _pool
        if _pool is None:
            _pool = multiprocessing.Pool(len(os.sched_getaffinity(0)), threadpool_limits, (1, "blas"))
        results = list(_pool.starmap(svds, [(m, limit) for m in sp_to_scipy(mat)]))
        # Transfer the singular values and vectors back to the GPU.
        Us = torch.from_numpy(np.stack([U for U, _, _ in results])).to(dev)
        Ss = torch.from_numpy(np.stack([S for _, S, _ in results])).to(dev)
        VTs = torch.from_numpy(np.stack([VT for _, _, VT in results])).to(dev)
        return Us, Ss, VTs


_pool = None
