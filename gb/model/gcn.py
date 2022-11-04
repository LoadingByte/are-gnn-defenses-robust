from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.model.fittable import StandardFittable
from gb.model.plumbing import Preprocess
from gb.model.tapeable import TapeableModule, TapeableParameter
from gb.preprocess import add_loops, gcn_norm
from gb.torchext import matmul
from gb.typing import Int, Float, IntSeq

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
features = None
classes = None
channels_in = None
channels_out = None


@typechecked
class MatmulPropagation(nn.Module):

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "channels_out"]
    ) -> TensorType["batch_out": ..., "nodes", "channels_out"]:
        return matmul(A, X)


@typechecked
class GCNConv(TapeableModule):

    def __init__(self, in_dim: Int, out_dim: Int, bias: bool = True, propagation: Optional[nn.Module] = None):
        super().__init__()
        self.weight = TapeableParameter(torch.empty(out_dim, in_dim))
        self.bias = TapeableParameter(torch.empty(out_dim)) if bias else None
        self.propagation = propagation if propagation is not None else MatmulPropagation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Note: Initialization is adopted from PyTorch Geometric.
        self.weight = nn.init.xavier_uniform_(torch.empty_like(self.weight))
        if self.bias is not None:
            self.bias = nn.init.zeros_(torch.empty_like(self.bias))

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "channels_in"]
    ) -> TensorType["batch_out": ..., "nodes", "channels_out"]:
        out = self.propagation(A, matmul(X, self.weight.T))
        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_dim={self.weight.shape[1]}, out_dim={self.weight.shape[0]}, bias={self.bias is not None}"


@typechecked
class GCN(TapeableModule, StandardFittable):

    def __init__(
            self,
            n_feat: Int,
            n_class: Int,
            hidden_dims: IntSeq,
            bias: bool = True,
            propagation: Optional[nn.Module] = None,
            loops: bool = True,
            activation: str = "relu",
            dropout: Float = 0.5
    ):
        super().__init__()
        self.norm = Preprocess(lambda A: gcn_norm(add_loops(A) if loops else A))
        self.convs = nn.ModuleList([
            GCNConv(in_dim, out_dim, bias, propagation)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])
        if activation == "none":
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown model activation: {activation}")
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "features"]
    ) -> TensorType["batch_out": ..., "nodes", "classes"]:
        A = self.norm(A)
        for conv in self.convs[:-1]:
            X = self.dropout(self.activation(conv(A, X)))
        X = self.convs[-1](A, X)
        return X
