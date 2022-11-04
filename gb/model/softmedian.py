import math
from itertools import repeat

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb import kernels
from gb.metric import pairwise_squared_euclidean
from gb.torchext import sum, mul, matmul, softmax, sp_new_values
from gb.typing import Float

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
channels_in = None
channels_out = None


@typechecked
class SoftMedianPropagation(nn.Module):

    def __init__(self, temperature: Float = 1.0, only_weight_neighbors: bool = True, expect_sparse: bool = True):
        super().__init__()
        self.temperature = temperature
        self.only_weight_neighbors = only_weight_neighbors
        self.expect_sparse = expect_sparse

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "channels_in"]
    ) -> TensorType["batch_out": ..., "nodes", "channels_out"]:
        if (A.is_cuda and self.expect_sparse) or self.only_weight_neighbors:
            A_sp = A if A.is_sparse else A.to_sparse()
        if A.is_cuda and self.expect_sparse:
            k = kernels.get()
            if A.ndim == 2 and X.ndim == 2:
                median_indices = k.dimmedian_idx(X, A_sp)
            else:
                median_indices = torch.stack([
                    k.dimmedian_idx(X_sub, A_sub)
                    for A_sub, X_sub in zip(repeat(A_sp) if A.ndim == 2 else A_sp, repeat(X) if X.ndim == 2 else X)
                ])
        else:
            # Note: When A is dense, we could also use the following code instead of making A sparse and calling the
            # custom kernel dimmedian_idx, however, that turns out to be significantly slower. So we only use the
            # following code when computing on the CPU or when A is not sparse.
            with torch.no_grad():
                A_ds = A.to_dense() if A.is_sparse else A
                sort = torch.argsort(X, dim=-2)
                med_idx = (A_ds[:, sort.T].transpose(-2, -3).cumsum(dim=-1) < A_ds.sum(dim=-1)[:, None] / 2).sum(dim=-1)
                median_indices = sort.gather(-2, med_idx.T)
        X_median = X.broadcast_to(median_indices.shape).gather(-2, median_indices)  # "x bar" in the paper
        if self.only_weight_neighbors:
            *batch_idx, row_idx, col_idx = A_sp._indices()
            diff = X_median[(*batch_idx, row_idx)] - (X[col_idx] if X.ndim == 2 else X[(*batch_idx, col_idx)])
            dist = sp_new_values(A_sp, diff.norm(dim=-1))  # c in the paper
        else:
            dist = (pairwise_squared_euclidean(X_median, X) + 1e-8).sqrt()  # c in the paper
        weights = softmax(-dist / (self.temperature * math.sqrt(X.shape[-1])), dim=-1)  # s in the paper
        A_weighted = mul(weights, A)  # "s * a" in the paper
        normalizers = sum(A, dim=-1, dense=True) / sum(A_weighted, dim=-1, dense=True)  # C in the paper
        return matmul(mul(A_weighted, normalizers[..., None]), X)  # Eq. 7 in the paper
