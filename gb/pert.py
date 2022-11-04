import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()
budget = None
dim_1 = None
dim_2 = None


@typechecked
def edge_diff_matrix(
        edge_pert: TensorType["budget", 2, torch.int64, torch.strided],
        A: TensorType["dim_1", "dim_2", torch.strided]
) -> TensorType["dim_1", "dim_2", torch.strided]:
    diff_mat = torch.zeros_like(A)
    diffs = 1 - 2 * A[edge_pert[:, 0], edge_pert[:, 1]]
    diff_mat[edge_pert[:, 0], edge_pert[:, 1]] = diffs
    diff_mat[edge_pert[:, 1], edge_pert[:, 0]] = diffs
    return diff_mat


@typechecked
def feat_diff_matrix(
        feat_pert: TensorType["budget", 2, torch.int64, torch.strided],
        X: TensorType["dim_1", "dim_2", torch.strided]
) -> TensorType["dim_1", "dim_2", torch.strided]:
    diff_mat = torch.zeros_like(X)
    diffs = 1 - 2 * X[feat_pert[:, 0], feat_pert[:, 1]]
    diff_mat[feat_pert[:, 0], feat_pert[:, 1]] = diffs
    return diff_mat


@typechecked
def sp_edge_diff_matrix(
        edge_pert: TensorType["budget", 2, torch.int64, torch.strided],
        A: TensorType["dim_1", "dim_2", torch.strided]
) -> TensorType["dim_1", "dim_2", torch.sparse_coo]:
    return torch.sparse_coo_tensor(
        indices=torch.hstack([edge_pert[:, :].T, edge_pert[:, [1, 0]].T]).to(A.device),
        values=(1 - 2 * A[edge_pert[:, 0], edge_pert[:, 1]]).tile(2),
        size=A.shape
    )


@typechecked
def sp_feat_diff_matrix(
        feat_pert: TensorType["budget", 2, torch.int64, torch.strided],
        X: TensorType["dim_1", "dim_2", torch.strided]
) -> TensorType["dim_1", "dim_2", torch.sparse_coo]:
    return torch.sparse_coo_tensor(
        indices=feat_pert.T.to(X.device),
        values=(1 - 2 * X[feat_pert[:, 0], feat_pert[:, 1]]),
        size=X.shape
    )
