from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import autograd, nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.model.fittable import StandardFittable
from gb.model.mlp import MLP
from gb.model.plumbing import Preprocess
from gb.model.tapeable import TapeableModule
from gb.preprocess import add_loops, gcn_norm
from gb.torchext import add_matmul
from gb.typing import Int, Float

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
features = None
classes = None


@typechecked
class GRAND(TapeableModule, StandardFittable):

    def __init__(self, mlp: MLP, dropnode: Float = 0.5, order: Int = 2, mlp_input_dropout: Float = 0.5):
        super().__init__()
        self.mlp = mlp
        self.dropnode = dropnode
        self.order = order
        self.mlp_input_dropout = nn.Dropout(mlp_input_dropout)
        self.adj_norm = Preprocess(lambda A: gcn_norm(add_loops(A)))
        self.feat_norm = Preprocess(lambda X: F.normalize(X, p=1, dim=-1))

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "features"]
    ) -> TensorType["batch_out": ..., "nodes", "classes"]:
        A = self.adj_norm(A)
        X = self.feat_norm(X)
        return self.mlp(self.mlp_input_dropout(self._rand_prop(A, X)))

    def _rand_prop(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "features"]
    ) -> TensorType["batch_out": ..., "nodes", "features"]:
        if self.training:
            mask = (torch.rand(X.shape[0], device=X.device) <= 1 - self.dropnode)[:, None]
            X = X * mask
        else:
            X = X * (1 - self.dropnode)

        return _Propagate.apply(A, X, self.order) / (self.order + 1)

    def fitting_scores(self, *, fw_args: Sequence, **kwargs) -> TensorType["nodes", "classes"]:
        self.eval()
        scores = self(*fw_args)
        self.train()
        return scores

    def fitting_step(
            self,
            *, epoch: Int,
            fw_args: Sequence,
            y: TensorType["nodes", torch.int64, torch.strided],
            train_nodes: TensorType[-1, torch.int64, torch.strided],
            differentiable: bool = False,
            n_samples: Int = 2,
            reg_consistency: Float = 1,
            sharpening_temperature: Float = 0.5,
            # Ignored, but verify there are no extraneous kwargs:
            train_cost=None, optimizer=None, lr=None, weight_decay=None
    ) -> None:
        # Main loss
        train_scores = [self(*fw_args) for _ in range(n_samples)]
        train_cost = sum(F.cross_entropy(s[train_nodes], y[train_nodes]) for s in train_scores) / n_samples

        # Consistency regularizer
        train_scores_softmax = [s.softmax(dim=-1) for s in train_scores]
        powed_avg = (sum(train_scores_softmax) / n_samples).pow(1 / sharpening_temperature)
        sharpened = (powed_avg / powed_avg.sum(dim=1, keepdim=True)).detach()  # from the author's code; has to be kept
        consistency = sum((s - sharpened).square().sum(dim=1).mean() for s in train_scores_softmax) / n_samples

        loss = train_cost + reg_consistency * consistency
        self.optim.step(autograd.grad(loss, self.optim.param_tensors(), create_graph=differentiable))


@typechecked
class _Propagate(autograd.Function):

    @staticmethod
    def forward(
            ctx,
            A: TensorType["batch_A":..., "nodes", "nodes"],
            X: TensorType["batch_X":..., "nodes", "features"],
            order: Int
    ) -> TensorType["batch_out":..., "nodes", "features"]:
        ctx.order = order
        ctx.save_for_backward(A, X)
        out = X
        for _ in range(order):
            out = add_matmul(X, A, out)
        return out

    @staticmethod
    def backward(
            ctx,
            grad_output: TensorType["batch_out":..., "nodes", "features"]
    ) -> Tuple[TensorType["batch_A":..., "nodes", "nodes"], None, None]:
        if ctx.needs_input_grad[1]:
            raise ValueError("Gradient computation for X is not supported yet")
        # Forward pass returned:
        #   A @ (A @ (A @ X + X) + X) + X ...
        # Gradient of A w.r.t. grad_output (represented by the letter "g") will be:
        #   (g) @ (A @ (A @ X + X) + X).T + (A.T @ g) @ (A @ X + X).T + (A.T @ A.T @ g) @ (X).T ...
        A, X = ctx.saved_tensors
        # First compute the interim results (X, A @ X + X, ...)
        interim_results = [X]
        running_interim_result = X
        for _ in range(ctx.order - 1):
            running_interim_result = add_matmul(X, A, running_interim_result)
            interim_results.append(running_interim_result)
        # Then compute the interim grad outputs (g, A.T @ g, ...), directly multiply them with the intermediate results
        # and sum them up to obtain the final gradient.
        grad = grad_output @ interim_results[-1].T
        running_grad_output = grad_output
        for interim_result in interim_results[-2::-1]:
            running_grad_output = A.T @ running_grad_output
            grad = add_matmul(grad, running_grad_output, interim_result.T)
        return grad, None, None
