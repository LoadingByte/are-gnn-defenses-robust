from typing import Union, Optional, Tuple

import torch
import torch.nn.functional as F
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
class RGCNConv(TapeableModule):

    def __init__(self, in_dim: Int, out_dim: Int, gamma: Float):
        super().__init__()
        self.gamma = gamma
        self.weight_M = TapeableParameter(torch.empty(out_dim, in_dim))
        self.weight_V = TapeableParameter(torch.empty(out_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight_M = nn.init.xavier_uniform_(torch.empty_like(self.weight_M))
        self.weight_V = nn.init.xavier_uniform_(torch.empty_like(self.weight_V))

    def forward(
            self,
            AM: TensorType["batch_A": ..., "nodes", "nodes"],
            AV: TensorType["batch_A": ..., "nodes", "nodes"],
            M: TensorType["batch_X": ..., "nodes", "channels_in"],
            V: TensorType["batch_X": ..., "nodes", "channels_in"]
    ) -> Tuple[
        TensorType["batch_out": ..., "nodes", "channels_out"],
        TensorType["batch_out": ..., "nodes", "channels_out"]
    ]:
        first_layer = M is V  # because both M and V are X in the first layer
        M = F.elu(matmul(M, self.weight_M.T))
        V = F.relu(matmul(V, self.weight_V.T))
        if first_layer:
            self.cached_M = M
            self.cached_V = V
        attention = torch.exp(-self.gamma * V)
        M = matmul(AM, M * attention)
        V = matmul(AV, V * attention * attention)
        return M, V

    def extra_repr(self) -> str:
        return f"in_dim={self.weight_M.shape[1]}, out_dim={self.weight_M.shape[0]}, gamma={self.gamma}"


@typechecked
class RGCN(TapeableModule, StandardFittable):
    """Implementation as provided by both the authors and the DeepRobust library; but differs from the RGCN paper!"""

    def __init__(
            self,
            n_feat: Int,
            n_class: Int,
            hidden_dims: IntSeq,
            gamma: Float = 1.0,
            dropout: Float = 0.5,
            sqrt_eps: Union[str, Float] = "auto"
    ):
        super().__init__()
        self.norm_AM = Preprocess(lambda A: gcn_norm(add_loops(A), sqrt=True))
        self.norm_AV = Preprocess(lambda A: gcn_norm(add_loops(A), sqrt=False))
        self.dropout = nn.Dropout(dropout)
        self.sqrt_eps = sqrt_eps
        self.convs = nn.ModuleList([
            RGCNConv(in_dim, out_dim, gamma)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "features"]
    ) -> TensorType["batch_out": ..., "nodes", "classes"]:
        AM = self.norm_AM(A)
        AV = self.norm_AV(A)
        X = self.dropout(X)
        M, V = self.convs[0](AM, AV, X, X)
        for conv in self.convs[1:]:
            M, V = conv(AM, AV, self.dropout(M), self.dropout(V))
        eps = torch.randn_like(V)
        return M + eps * (V + (1e-8 if self.sqrt_eps == "auto" else self.sqrt_eps)).sqrt()

    def fitting_regularizer(self, reg_kl: Float = 5e-4, **kwargs) -> Optional[TensorType[()]]:
        if reg_kl == 0:
            return None
        first_conv = self.convs[0]
        M = first_conv.cached_M
        V = first_conv.cached_V
        del first_conv.cached_M
        del first_conv.cached_V
        return reg_kl * (0.5 * (M.square() + V - (V + 1e-8).log()).mean(dim=-1)).sum(dim=-1)
