from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Optional, Iterable, Generator, Sequence, Tuple, List, Dict

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb.typing import Float

patch_typeguard()


@typechecked
class TapeableParameter:
    """A dumb container for a tensor. To be used together with tapeable modules."""

    def __init__(self, data: torch.Tensor):
        self._data = data.requires_grad_()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data.requires_grad_()

    def __deepcopy__(self, memodict=None):
        # It is unnecessary to clone the data tensor as it must never be modified inplace.
        return TapeableParameter(self.data)


@typechecked
class TapeableModule(nn.Module):
    """
    A module that doesn't accept regular nn.Parameters, but only tapeable parameters in a similar fashion. Accessing
    an object member which has previously been set to a tapeable parameter actually yields respectively swaps out the
    wrapped tensor of that tapeable parameter -- hence there is no difference between code that uses regular vs.
    tapeable parameters --, but the tapeable parameter object itself can't be accessed this way. It can only be
    retrieved using the tapeable_parameters() generator function.
    """

    def __init__(self):
        super().__init__()
        self._tapeable_parameters = OrderedDict()

    def tapeable_parameters(self, recurse: bool = True) -> Generator[TapeableParameter, None, None]:
        memo = set()
        for module in self.modules() if recurse else (self,):
            if isinstance(module, TapeableModule):
                for param in module._tapeable_parameters.values():
                    if param not in memo:
                        memo.add(param)
                        yield param

    def tapeable_parameter(self, name: str) -> TapeableParameter:
        return self._tapeable_parameters[name]

    def __getattr__(self, name: str) -> Any:
        if name in self._tapeable_parameters:
            return self._tapeable_parameters[name].data
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_tapeable_parameters")
        if isinstance(value, TapeableParameter):
            params[name] = value
        elif params is not None and name in params:
            if value is None or isinstance(value, torch.Tensor):
                params[name].data = value
            else:
                raise TypeError(f"Cannot assign '{torch.typename(value)}' as tapeable parameter '{name}' (tensor, "
                                "tapeable parameter, or None expected)")
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._tapeable_parameters:
            del self._tapeable_parameters[name]
        else:
            super().__delattr__(name)

    def _apply(self, fn: callable) -> TapeableModule:
        super()._apply(fn)
        for param in self._tapeable_parameters.values():
            param.data = fn(param.data)
        return self

    def get_extra_state(self) -> Dict[str, TapeableParameter]:
        return self._tapeable_parameters

    def set_extra_state(self, state: Dict[str, TapeableParameter]):
        for name, value in state.items():
            self._tapeable_parameters[name].data = value.data

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        raise ValueError("Regular nn.Parameters are disallowed in a tapeable module")


@typechecked
class TapeableSGD:

    def __init__(
            self,
            params: Iterable[TapeableParameter],
            lr: Float,
            momentum: Float = 0,
            dampening: Float = 0,
            weight_decay: Float = 0
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.momentum_buffers = [None] * len(self.params)

    def param_tensors(self) -> List[TensorType]:
        return [param.data for param in self.params]

    def step(self, grads: Sequence[TensorType]) -> None:
        with nullcontext() if any(grad.requires_grad for grad in grads) else torch.no_grad():
            for i, grad in enumerate(grads):
                param_data = self.params[i].data
                momentum_buffer = self.momentum_buffers[i]

                if self.weight_decay != 0:
                    grad = grad.add(param_data, alpha=self.weight_decay)
                if self.momentum != 0:
                    if momentum_buffer is None:
                        momentum_buffer = grad
                    else:
                        momentum_buffer = (momentum_buffer * self.momentum).add(grad, alpha=1 - self.dampening)
                    grad = momentum_buffer
                param_data = param_data.add(grad, alpha=-self.lr)

                self.params[i].data = param_data
                self.momentum_buffers[i] = momentum_buffer


@typechecked
class TapeableAdam:

    def __init__(
            self,
            params: Iterable[TapeableParameter],
            lr: Float = 1e-3,
            betas: Tuple[Float, Float] = (0.9, 0.999),
            eps: Float = 1e-8,
            weight_decay: Float = 0
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_cnt = 0
        self.exp_avgs = [torch.zeros_like(param.data) for param in self.params]
        self.exp_avg_sqs = [torch.zeros_like(param.data) for param in self.params]

    def param_tensors(self) -> List[TensorType]:
        return [param.data for param in self.params]

    def step(self, grads: Sequence[TensorType]) -> None:
        self.step_cnt += 1

        bias_correction1 = 1 - self.beta1 ** self.step_cnt
        bias_correction2 = 1 - self.beta2 ** self.step_cnt
        step_size = self.lr / bias_correction1

        with nullcontext() if any(grad.requires_grad for grad in grads) else torch.no_grad():
            for i, grad in enumerate(grads):
                param_data = self.params[i].data
                exp_avg = self.exp_avgs[i]
                exp_avg_sq = self.exp_avg_sqs[i]

                if self.weight_decay != 0:
                    grad = grad.add(param_data, alpha=self.weight_decay)
                # Decay the first and second moment running average coefficient
                exp_avg = (exp_avg * self.beta1).add(grad, alpha=1 - self.beta1)
                exp_avg_sq = (exp_avg_sq * self.beta2).addcmul(grad, grad.conj(), value=1 - self.beta2)
                # If we differentiate through the training procedure, force the gradient of exp_avg_sq to 0 where that
                # vector is 0. If we would not do this, the gradient would be infinite, leading to numerical issues.
                # This has been adapted from: https://github.com/facebookresearch/higher/blob/main/higher/optim.py
                # In addition, we require that exp_avg_sq is at least the smallest possible floating point number before
                # computing its square root. This doesn't change anything in the outcome, but turns out to be necessary
                # to evade all occurring numerical instabilities; don't ask me why.
                if exp_avg_sq.requires_grad:
                    exp_avg_sq.register_hook(self._get_mask_closure(exp_avg_sq == 0))
                    exp_avg_sq_sqrt = exp_avg_sq.clamp_min(torch.finfo(exp_avg_sq.dtype).tiny).sqrt()
                else:
                    exp_avg_sq_sqrt = exp_avg_sq.sqrt()
                denom = exp_avg_sq_sqrt / math.sqrt(bias_correction2) + self.eps
                param_data = param_data.addcdiv(exp_avg, denom, value=-step_size)

                self.params[i].data = param_data
                self.exp_avgs[i] = exp_avg
                self.exp_avg_sqs[i] = exp_avg_sq

    @staticmethod
    def _get_mask_closure(mask):
        def closure(grad):
            grad = torch.where(mask, torch.tensor(0.0, device=grad.device), grad)
            if grad.requires_grad:
                grad.register_hook(closure)
            return grad

        return closure
