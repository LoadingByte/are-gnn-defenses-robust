from contextlib import nullcontext
from typing import Sequence

import torch
from torch import autograd
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.model.fittable import Fittable
from gb.model.tapeable import TapeableModule, TapeableParameter, TapeableSGD, TapeableAdam
from gb.typing import Int, Float

patch_typeguard()
batch_X = None
nodes = None
features = None
classes = None

_STD_ADJ_LR = 1e-2


@typechecked
class ProGNN(TapeableModule, Fittable):
    """Property-GNN, reimplemented following the paper authors' implementation in DeepRobust."""

    def __init__(self, A: TensorType["nodes", "nodes"], gnn: TapeableModule):
        super().__init__()
        self.gnn = gnn
        self.A = A
        self.S = TapeableParameter(torch.empty_like(A))
        self.reset_parameters(constr=True)

    def reset_parameters(self, constr: bool = False) -> None:
        self.S = self.A.clone()
        if not constr:
            self.gnn.reset_parameters()

    def forward(self, X: TensorType["batch_X": ..., "nodes", "features"]) \
            -> TensorType["batch_X": ..., "nodes", "classes"]:
        return self.gnn(self.S, X)

    def fitting_setup(
            self,
            *, differentiable: bool = False,
            optimizer: str = "adam",
            gnn_lr: Float = 1e-2,
            gnn_weight_decay: Float = 5e-4,
            adj_lr: Float = _STD_ADJ_LR,
            adj_momentum: Float = 0.9,
            # Ignored, but verify there are no extraneous kwargs:
            adj_optim_interval=None, reg_adj_deviate=None, reg_adj_l1=None, reg_adj_nuclear=None, reg_feat_smooth=None
    ) -> None:
        if optimizer == "sgd":
            self.optim_gnn = TapeableSGD(self.gnn.tapeable_parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
        elif optimizer == "adam":
            self.optim_gnn = TapeableAdam(self.gnn.tapeable_parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        self.optim_adj = TapeableSGD([self.tapeable_parameter("S")], lr=adj_lr, momentum=adj_momentum)

    def fitting_scores(self, *, fw_args: Sequence, **kwargs) -> TensorType["nodes", "classes"]:
        return self(*fw_args)

    def fitting_step(
            self,
            *, epoch: Int,
            train_cost: TensorType[()],
            fw_args: Sequence,
            differentiable: bool = False,
            adj_lr: Float = _STD_ADJ_LR,
            adj_optim_interval: Int = 2,
            reg_adj_deviate: Float = 1.0,
            reg_adj_l1: Float = 5e-4,
            reg_adj_nuclear: Float = 1.5,
            reg_feat_smooth: Float = 1e-3,
            **kwargs
    ) -> None:
        do_adj_optim = epoch % adj_optim_interval == 0

        # Compute the gradient of the training cross entropy. Only compute the gradient w.r.t. the adjacency matrix if
        # we'll need that in a moment.
        gnn_param_tensors = self.optim_gnn.param_tensors()
        if do_adj_optim:
            S_grad, *gnn_grads = autograd.grad(train_cost, [self.S, *gnn_param_tensors], create_graph=differentiable)
        else:
            gnn_grads = autograd.grad(train_cost, gnn_param_tensors, create_graph=differentiable)

        if do_adj_optim:
            regularizer = torch.tensor(0.0, device=self.S.device)
            # Deviation regularization
            if reg_adj_deviate != 0:
                regularizer += reg_adj_deviate * (self.S - self.A).square().sum()
            # Feature smoothness regularization
            if reg_feat_smooth != 0:
                S_sym = (self.S + self.S.T) / 2
                deg = S_sym.sum(dim=-1)
                L = deg.diag() - S_sym
                fac = 1 / (deg + 1e-3).sqrt()
                fac[torch.isinf(fac)] = 0
                L_hat = L * fac[..., :, None] * fac[..., None, :]
                X = fw_args[0]
                regularizer += reg_feat_smooth * (X * (L_hat @ X)).sum()  # equiv. to (X.T @ L_hat @ X).trace()
            if reg_adj_deviate != 0 or reg_feat_smooth != 0:
                # Add the gradient of the differentiable regularizers w.r.t. the adjacency matrix to the previously
                # computed gradient. After that, we'll have the gradient of "cost + regularizer" w.r.t. the adj mat.
                S_grad = S_grad + torch.autograd.grad(regularizer, self.S, create_graph=differentiable)[0]
            # Step the optimizer using the gradient we just finished accumulating.
            self.optim_adj.step([S_grad])

            with torch.no_grad():
                # Nuclear norm proximal operator
                if reg_adj_nuclear != 0:
                    if differentiable:
                        raise ValueError("Cannot train differentiably when reg_adj_nuclear is not 0")
                    S_U, S_S, S_VT = torch.linalg.svd(self.S)
                    mod_S_S = (S_S - adj_lr * reg_adj_nuclear).clamp(min=0)
                    self.S = S_U @ mod_S_S.diag() @ S_VT

            with nullcontext() if differentiable else torch.no_grad():
                # L1 norm proximal operator
                if reg_adj_l1 != 0:
                    self.S = self.S.sign() * (self.S.abs() - adj_lr * reg_adj_l1).clamp(min=0)
                # Projection
                self.S = self.S.clamp(0, 1)

        self.optim_gnn.step(gnn_grads)
