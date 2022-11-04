from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.metric import pairwise_cosine
from gb.model.fittable import StandardFittable
from gb.model.gcn import GCNConv
from gb.model.tapeable import TapeableModule, TapeableParameter
from gb.preprocess import gcn_norm
from gb.torchext import sum, mul, div, abs, neq0, zero_keeping_exp, sp_diag
from gb.typing import Int, Float, IntSeq

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
features = None
classes = None


@typechecked
class GNNGuard(TapeableModule, StandardFittable):

    def __init__(
            self,
            n_feat: Int,
            n_class: Int,
            hidden_dims: IntSeq,
            mimic_ref_impl: bool = False,
            prune_edges: bool = False,
            dropout: Float = 0.5,
            div_limit: Union[str, Float] = "auto"
    ):
        super().__init__()
        self.mimic_ref_impl = mimic_ref_impl
        self.prune_edges = prune_edges
        # Just register these pruning weights as a buffer because they never receive any gradient anyway.
        self.register_buffer("pruning_weight", torch.empty(2) if prune_edges else None)
        self.pre_beta = TapeableParameter(torch.empty(()))
        self.dropout = nn.Dropout(dropout)
        self.div_limit = div_limit
        self.convs = nn.ModuleList([
            GCNConv(in_dim, out_dim, bias=True)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])
        self.reset_parameters(constr=True)

    def reset_parameters(self, constr: bool = False) -> None:
        if self.pruning_weight is not None:
            nn.init.xavier_uniform_(self.pruning_weight[None])
        self.pre_beta = nn.init.uniform_(torch.empty_like(self.pre_beta))
        if not constr:
            for conv in self.convs:
                conv.reset_parameters()

    def forward(
            self,
            A: TensorType["batch_A":..., "nodes", "nodes"],
            X: TensorType["batch_X":..., "nodes", "features"]
    ) -> TensorType["batch_out":..., "nodes", "classes"]:
        # Note: The reference implementation is buggy in this function, as it overwrites A with Alpha, but then later
        # feeds the modified A into _edge_weights() as had it been the original A. Here, we do not mimic that bug.
        beta = self.pre_beta.sigmoid()  # ensure between 0 and 1
        for idx, conv in enumerate(self.convs):
            Alpha = self._edge_weights(A, X)
            if idx == 0:
                W = Alpha
            else:
                # Layer-wise graph memory
                W = beta * W + (1 - beta) * Alpha
            del Alpha  # free memory
            X = conv(gcn_norm(W), X)
            if idx != len(self.convs) - 1:
                X = self.dropout(F.relu(X))
        return X

    def _edge_weights(
            self,
            A: TensorType["batch_A":..., "nodes", "nodes"],
            X: TensorType["batch_X":..., "nodes", "features"]
    ) -> TensorType["batch_out":..., "nodes", "nodes"]:
        # No gradients pass through the paper authors' implementation of this method.
        # Also, if we passed gradients through this, we would get unstable gradients due to the cosine distance.
        X = X.detach()

        # Build attention matrix from the pairwise cosine similarity matrix.
        cos = pairwise_cosine(X)
        if self.mimic_ref_impl:
            # Not in the paper, but in the reference implementation. It is vital to do this adjustment before we
            # multiply with A to not unnecessarily limit the gradient information flowing to A (which would, as
            # experiments show, substantially hinder attacking).
            cos[cos < 0.1] = 0
        # We multiply this way (it works because we know that A is binary) to get gradient information through to A.
        S = mul(A, cos)
        del cos  # free memory
        # Normalize S, yielding Alpha as defined in the paper respectively the reference implementation.
        if not self.mimic_ref_impl:
            N = sum(neq0(S).int(), dim=-1, dense=True)
            S_sums = sum(S, dim=-1, dense=True)
            S_sums[S_sums.abs() < (1e-8 if self.div_limit == "auto" else self.div_limit)] = 1
            Alpha = mul(S, (N / ((N + 1) * S_sums))[..., None])
        else:
            S_sums = sum(abs(S), dim=-1, dense=True)
            # Note: Taken from sklearn's normalize().
            S_sums[S_sums < ((10 * torch.finfo(S_sums.dtype).eps) if self.div_limit == "auto" else self.div_limit)] = 1
            Alpha = div(S, S_sums[..., None])
        del S, S_sums  # free memory

        # Edge pruning
        if self.prune_edges:
            edges = Alpha.nonzero()
            char_vec = torch.vstack([Alpha[edges[:, 0], edges[:, 1]], Alpha[edges[:, 1], edges[:, 0]]])
            drop_score = (self.pruning_weight @ char_vec).sigmoid()
            Alpha[tuple(edges[drop_score <= 0.5].T)] = 0

        # Add back the diagonal as defined in the paper respectively reference implementation. We do this after the
        # learnable edge dropping to not include these self-edges in the learnable edge dropping, which would quite
        # certainly stand against the paper's intent. This is also how the reference implementation does it.
        if self.mimic_ref_impl:
            # In contrast to the paper, the reference implementation computes N using Alpha only at this stage.
            N = sum(neq0(Alpha).int(), dim=-1, dense=True)
        Alpha = Alpha + sp_diag(1 / (N + 1))
        del N  # free memory

        if self.mimic_ref_impl:
            # Not in the paper, but in the reference implementation:
            Alpha = zero_keeping_exp(Alpha.coalesce() if Alpha.is_sparse else Alpha)

        return Alpha
