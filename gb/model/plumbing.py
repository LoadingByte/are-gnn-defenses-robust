from __future__ import annotations

import inspect
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Union, Optional, Collection, Sequence, Tuple, Callable, Generator

from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()
nodes = None
features = None


@contextmanager
@typechecked
def changed_fields(module: nn.Module, **kwargs) -> Generator[None, None, None]:
    if len(kwargs) == 0:
        yield
    else:
        consumed = set()
        memories = []
        for submodule in module.modules():
            memory = []
            for k, new_v in kwargs.items():
                if k in submodule.__dict__:
                    memory.append((k, submodule.__dict__[k]))
                    submodule.__dict__[k] = new_v
                    consumed.add(k)
            memories.append((submodule, memory))
        if len(consumed) != len(kwargs):
            raise ValueError(f"Supplied unused field names to 'with changed_fields()': {kwargs.keys() - consumed}")
        yield
        for submodule, memory in memories:
            for k, old_v in memory:
                submodule.__dict__[k] = old_v


@typechecked
class GraphSequential(nn.Sequential):

    def __init__(self, *args: Union[nn.Module, OrderedDict[str, nn.Module]], return_A_X: bool = False):
        super().__init__(*args)
        self.return_A_X = return_A_X

    def reset_parameters(self) -> None:
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()

    def forward(self, A: TensorType[..., "nodes", "nodes"], X: TensorType[..., "nodes", "features"]) \
            -> Union[TensorType, Tuple[TensorType[..., "nodes", "nodes"], TensorType[..., "nodes", "features"]]]:
        for idx, module in enumerate(self):
            if self.return_A_X or idx != len(self) - 1:
                A, X = self._preprocessing_step(module, A, X)
            else:
                params = tuple(inspect.signature(module.forward).parameters.keys())
                if params == ("A", "X"):
                    return module(A, X)
                elif params == ("A",):
                    return module(A)
                else:
                    return module(X)
        return A, X

    def sub(
            self,
            include: Collection[str] = (),
            exclude: Collection[str] = (),
            return_A_X: Optional[bool] = None
    ) -> GraphSequential:
        expected_names = set(include) | set(exclude)
        for name, _ in self.named_children():
            try:
                expected_names.remove(name)
            except KeyError:
                pass
        if expected_names:
            raise ValueError(f"Did not find modules with these names in this GraphSequential: {expected_names}")

        return GraphSequential(OrderedDict([
            (name, child) for name, child in self.named_children()
            if (not include or name in include) and name not in exclude
        ]), return_A_X=return_A_X if return_A_X is not None else self.return_A_X)

    def fit(self, fw_args: Sequence, *args, **kwargs) -> None:
        A, X = fw_args
        for idx, module in enumerate(self):
            if idx != len(self) - 1:
                A, X = self._preprocessing_step(module, A, X)
            else:
                module.fit((A, X), *args, **kwargs)

    def _preprocessing_step(
            self,
            module: nn.Module,
            A: TensorType[..., "nodes", "nodes"],
            X: TensorType[..., "nodes", "features"]
    ) -> Union[TensorType, Tuple[TensorType[..., "nodes", "nodes"], TensorType[..., "nodes", "features"]]]:
        params = tuple(inspect.signature(module.forward).parameters.keys())
        if params == ("A", "X"):
            out = module(A, X)
            if isinstance(out, tuple):
                return out
            else:
                return A, out
        elif params == ("A",):
            return module(A), X
        else:
            return A, module(X)


_AdjMat = TensorType[..., "nodes", "nodes"]
_FeatMat1 = TensorType[..., "nodes", "features_1"]
_FeatMat2 = TensorType[..., "nodes", "features_2"]
_MetricMat = TensorType[..., "metric": ...]


@typechecked
class Preprocess(nn.Module):

    def __init__(self, prep_fn: Callable[[TensorType], TensorType]):
        super().__init__()
        self.prep_fn = prep_fn
        self._cache_in = _TensorId()
        self._cache_out = None

    def forward(self, tensor: TensorType) -> TensorType:
        if not self._cache_in.equals(tensor):
            self._cache_in.set(tensor)
            self._cache_out = self.prep_fn(tensor)
        return self._cache_out


@typechecked
class PreprocessA(Preprocess):

    def __init__(self, prep_fn: Callable[[_AdjMat], _AdjMat]):
        super().__init__(prep_fn)

    def forward(self, A: _AdjMat) -> _AdjMat:
        return super().forward(A)


@typechecked
class PreprocessX(Preprocess):

    def __init__(self, prep_fn: Callable[[_FeatMat1], _FeatMat2]):
        super().__init__(prep_fn)

    def forward(self, X: _FeatMat1) -> _FeatMat2:
        return super().forward(X)


@typechecked
class PreprocessAUsingXMetric(nn.Module):

    def __init__(self, metric_fn: Callable[[_FeatMat1], _MetricMat], prep_fn: Callable[[_AdjMat, _MetricMat], _AdjMat]):
        super().__init__()
        self.metric_fn = metric_fn
        self.prep_fn = prep_fn
        self._cache_metric_in = _TensorId()
        self._cache_metric_out = None
        self._cache_prep_in = _TensorId()
        self._cache_prep_out = None

    def forward(self, A: _AdjMat, X: _FeatMat1) -> Tuple[_AdjMat, _FeatMat1]:
        updated_metric = False
        if not self._cache_metric_in.equals(X):
            updated_metric = True
            self._cache_metric_in.set(X)
            self._cache_metric_out = self.metric_fn(X)
        if updated_metric or not self._cache_prep_in.equals(A):
            self._cache_prep_in.set(A)
            self._cache_prep_out = self.prep_fn(A, self._cache_metric_out)
        return self._cache_prep_out, X


@typechecked
class _TensorId:

    def __init__(self):
        self._ref = None
        self._version = -1

    def set(self, tensor: TensorType):
        self._ref = weakref.ref(tensor)
        self._version = tensor._version

    def equals(self, other: TensorType) -> bool:
        if self._ref is None:
            return False
        tensor = self._ref()
        return tensor is not None and other is tensor and other._version == self._version
