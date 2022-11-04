import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.torchext import sp_svd
from gb.typing import Int, IntSeq

patch_typeguard()
nodes = None
features = None
classes = None
batch = None
dim_1 = None
dim_2 = None


@typechecked
def margin(
        scores: TensorType["batch": ..., "nodes", "classes"],
        y_true: TensorType["nodes", torch.int64]
) -> TensorType["batch": ..., "nodes"]:
    all_nodes = torch.arange(y_true.shape[0])
    # Get the scores of the true classes.
    scores_true = scores[..., all_nodes, y_true]
    # Get the highest scores when not considering the true classes.
    scores_mod = scores.clone()
    scores_mod[..., all_nodes, y_true] = -np.inf
    scores_pred_excl_true = scores_mod.amax(dim=-1)
    return scores_true - scores_pred_excl_true


@typechecked
def accuracy(
        scores: TensorType["batch": ..., "nodes", "classes"],
        y_true: TensorType["nodes"]
) -> TensorType["batch": ...]:
    return (scores.argmax(dim=-1) == y_true).count_nonzero(dim=-1) / y_true.shape[0]


@typechecked
def receptive_nodes(
        A: TensorType["nodes", "nodes", torch.strided],
        target_nodes: IntSeq, steps: Int
) -> TensorType[-1, torch.strided]:
    """
    Returns the indices of all nodes whose features have an influence on one of the given nodes in a GNN with the given
    number of convolution steps.
    """
    vec = torch.zeros(A.shape[0], device=A.device)
    vec[target_nodes] = 1
    A_loops = A + torch.eye(A.shape[0], device=A.device)
    for _ in range(steps):
        vec = A_loops @ vec
    return vec.nonzero()[:, 0]


@typechecked
def receptive_field(
        A: TensorType["nodes", "nodes", torch.strided],
        target_nodes: IntSeq, steps: Int
) -> TensorType["nodes", "nodes", torch.strided]:
    """
    Returns an adjacency-like binary matrix which is true at every edge whose presence or absence has an influence
    on one of the given nodes in a GNN with the given number of convolution steps.
    """
    reached_nodes = receptive_nodes(A, target_nodes, steps - 1)
    A_mask = torch.zeros_like(A)
    A_mask[reached_nodes, :] = 1
    A_mask[:, reached_nodes] = 1
    return A_mask


@typechecked
def pairwise_squared_euclidean(
        X: TensorType["batch":..., "nodes", "features", torch.strided],
        Z: TensorType["batch":..., "nodes", "features", torch.strided]
) -> TensorType["batch": ..., "nodes", "nodes", torch.strided]:
    squared_X_feat_norms = (X * X).sum(dim=-1)  # sxfn_i = <X_i|X_i>
    squared_Z_feat_norms = (Z * Z).sum(dim=-1)  # szfn_i = <Z_i|Z_i>
    pairwise_feat_dot_prods = X @ Z.transpose(-2, -1)  # pfdp_ij = <X_i|Z_j>
    return (-2 * pairwise_feat_dot_prods + squared_X_feat_norms[:, None] + squared_Z_feat_norms[None, :]).clamp_min(0)


@typechecked
def pairwise_cosine(
        X: TensorType["batch":..., "nodes", "features", torch.strided]
) -> TensorType["batch": ..., "nodes", "nodes", torch.strided]:
    pairwise_feat_dot_prods = X @ X.transpose(-2, -1)  # pfdp_ij = <X_i|X_j>
    range_ = torch.arange(pairwise_feat_dot_prods.shape[-1])
    feat_norms = pairwise_feat_dot_prods[..., range_, range_].sqrt()  # fn_i = ||X_i||_2
    feat_norms = torch.where(feat_norms < 1e-8, torch.tensor(1.0, device=X.device), feat_norms)
    return pairwise_feat_dot_prods / feat_norms[..., :, None] / feat_norms[..., None, :]


@typechecked
def pairwise_jaccard(X: TensorType["nodes", "features", torch.strided]) -> TensorType["nodes", "nodes", torch.strided]:
    X_flip = 1 - X
    # Per node-node pair, number of features which occur in both nodes.
    and_cnt = X @ X.T
    # Per node-node pair, number of features which occur in at least one node.
    or_cnt = and_cnt + X @ X_flip.T + X_flip @ X.T
    return and_cnt / or_cnt


@typechecked
def pro_gnn_feature_dissimilarity(
        A: TensorType["nodes", "nodes", torch.strided],
        X: TensorType["nodes", "features", torch.strided]
) -> TensorType["nodes", "nodes", torch.strided]:
    """Computes a matrix B with B_ij = ||X_i / sqrt(degree(i)) - X_j / sqrt(degree(j))||_2^2"""
    deg = A.sum(dim=-1)
    deg_sqrt = deg.sqrt()
    pairwise_feat_dot_prods = X @ X.T  # pfdp_ij = <X_i|X_j>
    weighted_squared_feat_norms = pairwise_feat_dot_prods.diag() / deg  # wsfn_i = ||X_i / sqrt(degree(i))||_2^2
    return (-2 * pairwise_feat_dot_prods / deg_sqrt[:, None] / deg_sqrt[None, :]
            + weighted_squared_feat_norms[:, None] + weighted_squared_feat_norms[None, :])


@typechecked
def eigenspace_alignment(
        mat: TensorType["dim_1", "dim_2", torch.strided],
        rank: Int,
        symmetrize: str = "survivor_avg"
) -> TensorType["dim_1", "dim_2", torch.strided]:
    """
    :param symmetrize: 'survivor_avg' measures the average length of perturbations that will survive the SVD.
                       'survivor_sum' is the same but multiplied by 2.
                       'avg' measures the average length of perturbation vectors after the SVD, not only in the
                       direction of the original perturbation.
    """

    U, _, VT = sp_svd(mat, rank)
    # V contains mat's top principal components. The projection matrix onto the corresponding space would be V @ V.T.
    # Find the values on the projection matrix' diagonal.
    proj_diag_V = (VT * VT).sum(dim=0, keepdim=True)
    # Do the same for the other subspace.
    proj_diag_U = (U * U).sum(dim=1, keepdim=True)
    if symmetrize == "survivor_avg":
        return (proj_diag_V + proj_diag_U) / 2
    elif symmetrize == "survivor_sum":
        return proj_diag_V + proj_diag_U
    elif symmetrize == "avg":
        return (proj_diag_V.sqrt() + proj_diag_U.sqrt()) / 2
    else:
        raise ValueError(f"Only 'survivor_avg', 'survivor_sum', or 'avg' allowed for symmetrize, not '{symmetrize}'")
