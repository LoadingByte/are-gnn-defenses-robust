from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torchtyping import TensorType, patch_typeguard
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from gb.metric import accuracy
from gb.model.tapeable import TapeableModule, TapeableSGD, TapeableAdam
from gb.typing import Int, Float

patch_typeguard()
nodes = None
classes = None


@typechecked
class Fittable(ABC):
    """
    Mixin for TapeableModules that adds a stateless fit() method and expects implementations of some abstract methods.
    """

    def fit(
            self,
            fw_args: Sequence,
            y: TensorType["nodes", torch.int64, torch.strided],
            train_nodes: TensorType[-1, torch.int64, torch.strided],
            val_nodes: TensorType[-1, torch.int64, torch.strided],
            *, repetitions: Int = 1,
            max_epochs: Int = 3000,
            patience: Optional[Int] = 50,
            yield_best: bool = True,
            progress: bool = True,
            scores_cb: Optional[Callable[[TensorType["nodes", "classes"]], None]] = None,
            scores_cbs: Optional[Sequence[Callable[[TensorType["nodes", "classes"]], None]]] = None,
            metric_cb: Optional[Callable[[float, float], None]] = None,
            metric_cbs: Optional[Sequence[Callable[[float, float], None]]] = None,
            **kwargs
    ) -> None:
        if not isinstance(self, TapeableModule):
            raise ValueError("Fittable only supports tapeable modules")
        if repetitions != 1 and not yield_best:
            raise ValueError("Can only use multiple repetitions in combination with yield_best at the moment")
        if scores_cbs is None and scores_cb is not None:
            if repetitions != 1:
                raise ValueError("Cannot use a single scores_cb for multiple repetitions")
            scores_cbs = [scores_cb]
            del scores_cb
        if metric_cbs is None and metric_cb is not None:
            if repetitions != 1:
                raise ValueError("Cannot use a single metric_cb for multiple repetitions")
            metric_cbs = [metric_cb]
            del metric_cb

        self.train()
        self.fitting_setup(**kwargs)

        if yield_best:
            best_val_cost = np.inf
            best_state = None
        for rep in trange(repetitions, leave=False) if progress and repetitions != 1 else range(repetitions):
            if rep != 0:
                self.reset_parameters()
            if patience is not None:
                best_rep_val_cost = np.inf
                best_rep_epoch = None
            for epoch in trange(max_epochs, leave=False) if progress else range(max_epochs):
                scores = self.fitting_scores(fw_args=fw_args, **kwargs)
                if scores_cbs is not None:
                    scores_cbs[rep](scores)
                # For now, cost is hardcoded as cross entropy, but if need be, we can always make this abstract later.
                train_cost = F.cross_entropy(scores[train_nodes], y[train_nodes])
                if yield_best or patience is not None or progress or metric_cbs is not None:
                    val_cost = F.cross_entropy(scores[val_nodes], y[val_nodes]).item()
                if progress and epoch % 20 == 0:
                    train_acc = accuracy(scores[train_nodes], y[train_nodes]).item()
                    val_acc = accuracy(scores[val_nodes], y[val_nodes]).item()
                    tqdm.write(f"Epoch {epoch:4}: train_cost={train_cost.item():.5f}, val_cost={val_cost:.5f}  "
                               f"(train_acc={train_acc:.5f}, val_acc={val_acc:.5f})")
                if metric_cbs is not None:
                    metric_cbs[rep](train_cost.item(), val_cost)

                if patience is not None:
                    if val_cost < best_rep_val_cost:
                        best_rep_val_cost = val_cost
                        best_rep_epoch = epoch
                    elif epoch >= best_rep_epoch + patience:
                        break

                if yield_best and val_cost < best_val_cost:
                    best_val_cost = val_cost
                    del best_state  # free memory early
                    best_state = deepcopy(self.state_dict())

                self.fitting_step(
                    epoch=epoch, train_cost=train_cost, fw_args=fw_args, y=y, train_nodes=train_nodes, **kwargs
                )

        if yield_best:
            self.load_state_dict(best_state)

        self.eval()

    @abstractmethod
    def fitting_setup(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fitting_scores(self, *, fw_args: Sequence, **kwargs) -> TensorType["nodes", "classes"]:
        """Returns a score matrix that will be used for validation and also passed on to fitting_step()."""
        raise NotImplementedError

    @abstractmethod
    def fitting_step(
            self,
            *, epoch: Int,
            train_cost: TensorType[()],
            fw_args: Sequence,
            y: TensorType["nodes", torch.int64, torch.strided],
            train_nodes: TensorType[-1, torch.int64, torch.strided],
            **kwargs
    ) -> None:
        """
        Gets both the train_cost computed previously from the result of fitting_scores(), and forward args, y and
        train_nodes which could be used to compute new scores and a new train_cost. Steps the model parameters.
        """
        raise NotImplementedError


@typechecked
class StandardFittable(Fittable):
    """
    Mixin for TapeableModules that adds a fit() method by inheriting from the Fittable mixin and implements its abstract
    methods with a standard fitting procedure.
    """

    def fitting_setup(self, *, optimizer: str = "adam", lr: Float = 1e-2, weight_decay: Float = 1e-2, **kwargs) -> None:
        if optimizer == "sgd":
            self.optim = TapeableSGD(self.tapeable_parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "adam":
            self.optim = TapeableAdam(self.tapeable_parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def fitting_scores(self, *, fw_args: Sequence, **kwargs) -> TensorType["nodes", "classes"]:
        return self(*fw_args)

    def fitting_regularizer(self, *, epoch: Int, **kwargs) -> Optional[TensorType[()]]:
        return None

    def fitting_step(self, *, epoch: Int, train_cost: TensorType[()], differentiable: bool = False, **kwargs) -> None:
        regularizer = self.fitting_regularizer(epoch=epoch, **kwargs)
        loss = train_cost if regularizer is None else train_cost + regularizer
        self.optim.step(autograd.grad(loss, self.optim.param_tensors(), create_graph=differentiable))
