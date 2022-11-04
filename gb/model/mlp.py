import math

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.model.fittable import StandardFittable
from gb.model.tapeable import TapeableModule, TapeableParameter
from gb.typing import Int, Float, IntSeq

patch_typeguard()
batch = None
nodes = None
features = None
classes = None
channels_in = None
channels_out = None


@typechecked
class TapeableLinear(TapeableModule):

    def __init__(self, in_dim: Int, out_dim: Int, bias: bool = True):
        super().__init__()
        self.weight = TapeableParameter(torch.empty(out_dim, in_dim))
        self.bias = TapeableParameter(torch.empty(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # This method is mostly copied from torch.nn.Linear.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: TensorType["batch": ..., "channels_in"]) -> TensorType["batch": ..., "channels_out"]:
        return F.linear(X, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_dim={self.weight.shape[1]}, out_dim={self.weight.shape[0]}, bias={self.bias is not None}"


@typechecked
class MLP(TapeableModule, StandardFittable):

    def __init__(
            self,
            n_feat: Int,
            n_class: Int,
            hidden_dims: IntSeq,
            bias: bool = True,
            dropout: Float = 0.5
    ):
        super().__init__()
        self.linears = nn.ModuleList([
            TapeableLinear(in_dim, out_dim, bias)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, X: TensorType["batch": ..., "nodes", "features"]) -> TensorType["batch": ..., "nodes", "classes"]:
        for linear in self.linears[:-1]:
            X = self.dropout(F.relu(linear(X)))
        X = self.linears[-1](X)
        return X
