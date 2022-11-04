from __future__ import annotations

import math
import re
from functools import cached_property
from itertools import cycle, count, zip_longest, starmap, product, takewhile
from operator import itemgetter
from typing import Any, Optional, Iterable, Sequence, List, Dict, Callable, Union, Tuple, Collection

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter
from munch import Munch
from torchtyping import patch_typeguard
from typeguard import typechecked

from gb.data import get_dataset, get_all_benchmark_targets, get_num_nodes_per_benchmark_target
from gb.typing import Int, Float, FloatSeq

patch_typeguard()

COLOR_CYCLE: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Computed ourselves
MLP_TEST_ACCURACY: Dict[str, float] = {"citeseer": 0.67, "cora": 0.65}


# General notation:
#   mr = misclassification rate
#   mono = monotonic


@typechecked
def group(
        elems: Iterable[Any],
        key: Optional[Callable[[Any], Any]],
        *,
        filt: Optional[Callable[[Any, Any], bool]] = None,
        sort: Optional[Callable[[Any, Any], Any]] = None
) -> Dict[Any, List[Any]]:
    if key is None:
        grouping = {None: list(elems)}
    else:
        grouping = {}
        for elem in elems:
            k = key(elem)
            if filt is None or filt(elem, k):
                grouping.setdefault(k, []).append(elem)
    sort_key = (lambda item: sort(item[1], item[0])) if sort is not None else itemgetter(0)
    return dict(sorted(grouping.items(), key=sort_key))


@typechecked
def plot_shuffler(
        exs: Iterable[Munch],
        *,
        plot: Callable[[Munch, ...], Any],
        filt: Optional[Callable[[Munch], bool]] = None,
        agrp: Optional[Callable[[Munch], str]] = None,
        asrt: Optional[Callable[[Munch, str], Any]] = None,
        shar: Optional[Callable[[Munch], Any]] = None,
        pgrp: Optional[Callable[[Munch], str]] = None,
        psrt: Optional[Callable[[Munch, str], Any]] = None,
        conf: Union[None, Callable[[], Any], Callable[[Int], Any]] = None,
        cols: Int = 2
) -> None:
    filt_exs = exs if filt is None else [ex for ex in exs if filt(ex)]
    axgrouping = group(filt_exs, agrp, sort=asrt)
    prev_axs = {}
    for axgrp_idx, (axgrp_key, axgrp_exs) in enumerate(axgrouping.items()):
        prev_ax = None
        if shar is not None:
            prev_ax_key = None if shar is None else shar(axgrp_exs[0])
            prev_ax = prev_axs.get(prev_ax_key)
        ax = plt.subplot(math.ceil(len(axgrouping) / cols), cols, axgrp_idx + 1, sharex=prev_ax, sharey=prev_ax)
        if shar is not None:
            prev_axs[prev_ax_key] = ax
        plt.title(axgrp_key)
        pltgrouping = group(axgrp_exs, pgrp, sort=psrt)
        for pltgrp_idx, (pltgrp_key, pltgrp_exs) in enumerate(pltgrouping.items()):
            plot_param_names = plot.__code__.co_varnames[:plot.__code__.co_argcount]
            plot_kwargs = {}
            if "label" in plot_param_names:
                plot_kwargs["label"] = pltgrp_key
            if "exs" in plot_param_names:
                plot_kwargs["exs"] = pltgrp_exs
            elif "ex" in plot_param_names:
                if len(pltgrp_exs) != 1:
                    raise ValueError(
                        f"Expected 1, but found {len(pltgrp_exs)} experiments in axis group '{axgrp_key}' and plot "
                        f"group '{pltgrp_key}'"
                    )
                plot_kwargs["ex"] = pltgrp_exs[0]
            else:
                raise ValueError("Found neither 'ex' nor 'exs' in 'plot' function parameter names")
            plot(**plot_kwargs)
        if conf is not None:
            if conf.__code__.co_argcount == 1:
                conf(axgrp_idx)
            else:
                conf()
    plt.tight_layout()


@typechecked
def model_slug(ex: Munch) -> str:
    if "to_model_slug" in ex.config:
        return ex.config.to_model_slug
    arch = ex.config.model.arch
    if arch == "gcn":
        return "gcn" + ("_1" if ex.config.model.dropout == 0.5 else "_2")
    elif arch == "jaccard_gcn":
        return "jaccard_gcn" + ("_faith" if ex.config.model.dropout == 0.5 else "_tuned")
    elif arch in ("svd_gcn", "svd_gcn_feats"):
        return arch + f"_rank{ex.config.model.rank}" + ("_faith" if ex.config.model.dropout == 0.5 else "_tuned")
    elif arch == "rgcn":
        return "rgcn" + ("_faith" if ex.config.training.weight_decay < 0.01 else "_tuned")
    elif arch == "pro_gnn":
        return "pro_gnn" + ("_faith" if ex.config.training.reg_adj_l1 < 0.01 else "_tuned")
    elif arch == "gnn_guard":
        return "gnn_guard_faith" + ("_refimpl" if ex.config.model.mimic_ref_impl else "_paper") + \
               ("_prune" if ex.config.model.prune_edges else "")
    elif arch == "grand":
        return "grand_tuned"
    elif arch == "soft_median_gdc":
        return "soft_median_gdc" + ("_faith" if ex.config.model.teleport_proba == 0.15 else "_tuned")
    else:
        raise ValueError(f"Unknown model arch: {arch}")


model_slug_order = [
    "gcn_2", "gcn_1",
    "jaccard_gcn_faith", "jaccard_gcn_tuned",
    "svd_gcn_rank10_faith", "svd_gcn_rank10_tuned", "svd_gcn_rank50_faith", "svd_gcn_rank50_tuned",
    "svd_gcn_feats_rank10_faith", "svd_gcn_feats_rank10_tuned",
    "svd_gcn_feats_rank50_faith", "svd_gcn_feats_rank50_tuned",
    "rgcn_faith", "rgcn_tuned",
    "pro_gnn_faith", "pro_gnn_tuned",
    "gnn_guard_faith_paper", "gnn_guard_faith_refimpl",
    "grand_tuned",
    "soft_median_gdc_faith", "soft_median_gdc_tuned"
]


@typechecked
def project_model_slugs(slugs: Union[str, Iterable[str]], dataset_name: str) -> Union[str, List[str]]:
    """If a model slug isn't available on a certain dataset, yields the most similar available slug."""
    if isinstance(slugs, str):
        return "soft_median_gdc_faith" if slugs == "soft_median_gdc_tuned" and dataset_name != "citeseer" else slugs
    else:
        return [project_model_slugs(slug, dataset_name) for slug in slugs]


@typechecked
def model_label(ex_or_slug: Union[Munch, str]) -> str:
    return (ex_or_slug if isinstance(ex_or_slug, str) else model_slug(ex_or_slug)) \
        .replace("jaccard_gcn", "Jaccard-GCN") \
        .replace("svd_gcn", "SVD-GCN") \
        .replace("_feats", "-Features") \
        .replace("_rank", " Rank ") \
        .replace("rgcn", "RGCN") \
        .replace("pro_gnn", "ProGNN") \
        .replace("gnn_guard", "GNNGuard") \
        .replace("_refimpl", " RefImpl") \
        .replace("_paper", " Paper") \
        .replace("_prune", " Prune") \
        .replace("grand", "GRAND") \
        .replace("soft_median_gdc", "Soft-Median-GDC") \
        .replace("gcn", "GCN") \
        .replace("_1", " 1") \
        .replace("_2", " 2") \
        .replace("_faith", " Faithful") \
        .replace("_tuned", " Tuned")


@typechecked
def model_slug_config(slug: str, exs_repository: List[Munch]) -> Optional[Munch]:
    def amend_config(cfg: Munch) -> Munch:
        if cfg.model.arch == "soft_median_gdc":
            cfg.model = cfg.model.copy()
            cfg.model.setdefault("only_weight_neighbors", True)
        cfg.training = cfg.training.copy()
        cfg.training.setdefault("repetitions", 1)
        return cfg

    slug_exs = [ex for ex in exs_repository if model_slug(ex) == slug]
    if len(slug_exs) == 0:
        return None

    # Sanity-check that all experiments with the same slug share the same model-specific config.
    proto_config = amend_config(slug_exs[0].config)
    for ex in slug_exs:
        ex_config = amend_config(ex.config)
        if ex_config.model != proto_config.model or ex_config.training != proto_config.training:
            raise ValueError(f"Experiments with IDs {slug_exs[0]._id} and {ex._id} have different model and/or "
                             f"training configurations even though they share the same model slug '{slug}'")

    return Munch(model=proto_config.model, training=proto_config.training)


@typechecked
def attack_tags(ex: Munch) -> Dict[str, Any]:
    arch = ex.config.model.arch
    attack = ex.config.attack
    method = attack.method
    tags = {"method": attack.method}
    if method == "nettack_edges":
        tags["surrogate"] = attack.surrogate is not None
    elif attack.scope == "global":
        tags["aggregation"] = attack.loss.aggregation
    if method.startswith("pgd"):
        tags["init"] = attack.get("init_from_run_id") is not None
    if method == "greedy_meta_edges":
        ma = attack.meta_adjustment
        tags["sgd"] = ma is not None and ma.training.get("optimizer") == "sgd"
    if arch == "jaccard_gcn":
        if method != "brute_force_edges":
            dl = attack.get("loss", {}).get("drop_layers")
            tags["mask"] = len(dl) == 0 if dl is not None else len(attack.edge_diff_masks) != 0
    elif arch == "svd_gcn":
        if method != "brute_force_edges":
            tags["freeze"] = len(attack.get("loss", {}).get("freeze_layers", {})) != 0
            mask = attack.get("edge_diff_masks", {}).get("e1")
            if mask is None:
                mask = attack.get("loss", {}).get("edge_diff_masks", {}).get("e1")
            tags["mask"] = None if mask is None else "survivor" if mask.symmetrize == "survivor_avg" else "proj_len"
    elif arch == "pro_gnn":
        if method == "pgd_edges":
            tags["multi"] = attack.num_auxiliaries != 1
    elif arch == "grand":
        if method == "pgd_meta_edges":
            tags["unlim_epochs_and_pro_gnn_init"] = attack.meta_adjustment.training.get("max_epochs") is None
    return tags


@typechecked
class Curve:

    def __init__(self, xs: np.ndarray, ys_per_split: np.ndarray):
        if xs.ndim != 1:
            raise ValueError(f"Curve xs array must have 1 dimension, not {xs.ndim}")
        if ys_per_split.ndim != 2:
            raise ValueError(f"Curve xs array must have 2 dimensions, not {ys_per_split.ndim}")
        if not np.all(xs[:-1] <= xs[1:]):
            raise ValueError("Curve xs array must be monotonically increasing")
        self.xs = xs
        self.ys_per_split = ys_per_split

    @property
    def ys(self) -> np.ndarray:
        if self.ys_per_split.shape[0] != 1:
            raise ValueError("Can only retrieve ys if they have been aggregated across splits")
        return self.ys_per_split[0]

    @cached_property
    def mean(self) -> Float:
        return Curve(self.xs, np.mean(self.ys_per_split, axis=0, keepdims=True))

    @cached_property
    def std(self) -> Float:
        return Curve(self.xs, np.std(self.ys_per_split, axis=0, keepdims=True))

    @cached_property
    def alc_per_split(self) -> np.ndarray:
        """ALC = area to the left of the curve"""
        return np.trapz(self.xs, self.ys_per_split)

    def at(self, x: Float) -> np.ndarray:
        return _interp(x, self.xs, self.ys_per_split)

    def where(self, y: Float) -> np.ndarray:
        slc = slice(None) if self.ys_per_split[0][0] < self.ys_per_split[0][-1] else slice(None, None, -1)
        if not np.all(np.diff(self.ys_per_split[:, slc]) >= 0):
            raise ValueError("Can only call where() for monotonically increasing or decreasing curves")
        # We have a special case for when y is found exactly in ys_per_split to get around inconsistency issues when
        # a certain y occurs multiple times, i.e., the curve is not strictly monotonic.
        eq = self.ys_per_split == y
        exact = np.any(eq, axis=1)
        return np.where(exact, self.xs[np.argmax(eq, axis=1)], _interp(y, self.ys_per_split[:, slc], self.xs[slc]))

    def resampled(self, xs: FloatSeq) -> Curve:
        return Curve(np.array(xs), _interp(xs, self.xs, self.ys_per_split))

    def scaled(self, x: Optional[Float] = None, y: Optional[Float] = None) -> Curve:
        scl_xs = self.xs if x is None else self.xs * x
        scl_ys_per_split = self.ys_per_split if y is None else self.ys_per_split * y
        return Curve(scl_xs, scl_ys_per_split)

    def terminated_at_x(self, limit: Float) -> Curve:
        """Cuts off the curve at the given x-coordinate, or extends it horizontally to reach that coordinate."""

        cutoff = next((i for i, x in enumerate(self.xs) if x > limit), len(self.xs))
        lim_xs = self.xs[:cutoff]
        lim_ys_per_split = self.ys_per_split[:, :cutoff]
        if lim_xs[-1] != limit:
            lim_xs = np.r_[lim_xs, limit]
            lim_ys_per_split = np.c_[lim_ys_per_split, _interp(limit, self.xs, self.ys_per_split)]
        return Curve(lim_xs, lim_ys_per_split)

    def terminated_at_y(self, limit: Float) -> Curve:
        """
        Cuts off the curve where it reaches the given y-coordinate for the first time, or extends it vertically to
        reach that coordinate at the end. If there are multiple splits in the curve, the resulting curve might have
        horizontal flats at the end of some ys.
        """

        lim_xs = self.xs.copy()
        lim_ys_per_split = self.ys_per_split.copy()
        for split_idx in range(len(lim_ys_per_split)):
            split_ys = lim_ys_per_split[split_idx]
            cutoff = next((i for i, y in enumerate(split_ys) if y >= limit), None)
            if cutoff is None:
                if lim_xs[-1] == lim_xs[-2]:
                    split_ys[-1] = limit
                else:
                    lim_xs = np.r_[lim_xs, lim_xs[-1]]
                    lim_ys_per_split = np.c_[lim_ys_per_split, lim_ys_per_split[:, -1]]
                    lim_ys_per_split[split_idx][-1] = limit
            elif cutoff == 0:
                split_ys[:] = limit
            else:
                y_bef, y_aft = split_ys[cutoff - 1], split_ys[cutoff]
                if y_aft == limit:
                    split_ys[cutoff + 1:] = limit
                else:
                    x_bef, x_aft = lim_xs[cutoff - 1], lim_xs[cutoff]
                    new_x = x_bef + (x_aft - x_bef) * ((limit - y_bef) / (y_aft - y_bef))
                    lim_ys_per_split = np.c_[
                        lim_ys_per_split[:, :cutoff],
                        _interp(new_x, lim_xs, lim_ys_per_split),
                        lim_ys_per_split[:, cutoff:]
                    ]
                    lim_ys_per_split[split_idx][cutoff + 1:] = limit
                    lim_xs = np.r_[lim_xs[:cutoff], new_x, lim_xs[cutoff:]]
        trim = min(sum(takewhile(lambda b: b, split_ys[-2::-1] == split_ys[:0:-1])) for split_ys in lim_ys_per_split)
        if trim != 0:
            lim_xs = lim_xs[:-trim]
            lim_ys_per_split = lim_ys_per_split[:, :-trim]
        return Curve(lim_xs, lim_ys_per_split)

    def monotonic(self) -> Curve:
        return Curve(self.xs, np.maximum.accumulate(self.ys_per_split, axis=1))

    def __neg__(self):
        return Curve(self.xs, -self.ys_per_split)

    def __add__(self, other: Union[Float, Curve]) -> Curve:
        return self._bi_op(lambda a, b: a + b, other)

    def __sub__(self, other: Union[Float, Curve]) -> Curve:
        return self._bi_op(lambda a, b: a - b, other)

    def __mul__(self, other: Union[Float, Curve]) -> Curve:
        return self._bi_op(lambda a, b: a * b, other)

    def __truediv__(self, other: Union[Float, Curve]) -> Curve:
        return self._bi_op(lambda a, b: a / b, other)

    def _bi_op(self, op: callable, other: Union[Float, Curve]):
        if isinstance(other, Curve):
            return Curve._fold(lambda ys_per_curve: op(ys_per_curve[0], ys_per_curve[1]), [self, other])
        else:
            return Curve(self.xs, op(self.ys_per_split, other))

    @staticmethod
    def sum(curves: Iterable[Curve]) -> Curve:
        return Curve._fold(lambda ys_per_curve: np.sum(ys_per_curve, axis=0), curves)

    @staticmethod
    def max(curves: Iterable[Curve]) -> Curve:
        return Curve._fold(lambda ys_per_curve: np.max(ys_per_curve, axis=0), curves)

    @staticmethod
    def _fold(aggr: callable, curves: Iterable[Curve]) -> Curve:
        curves = list(curves)
        n_splits = {curve.ys_per_split.shape[0] for curve in curves} - {1}
        if len(n_splits) > 1:
            raise ValueError("Not all curves share the same number of splits")
        n_splits = next(iter(n_splits), 1)
        xs = np.array(sorted({x for curve in curves for x in curve.xs}))
        ys_per_split = aggr(np.array([
            np.tile(_interp(xs, curve.xs, curve.ys_per_split), ((1 if len(curve.ys_per_split) != 1 else n_splits), 1))
            for curve in curves
        ]))
        return Curve(xs, ys_per_split)

    @staticmethod
    def align(curves: Iterable[Curve]) -> List[Curve]:
        curves = list(curves)
        xs = np.array(sorted({x for curve in curves for x in curve.xs}))
        return [Curve(xs, _interp(xs, curve.xs, curve.ys_per_split)) for curve in curves]

    def __repr__(self):
        fmt_xs = ", ".join(f"{v:.5g}" for v in self.xs)
        fmt_ys = "[" + "], [".join(", ".join(f"{v:.5g}" for v in split_ys) for split_ys in self.ys_per_split) + "]"
        return f"gb.ana.Curve(np.array([{fmt_xs}]), np.array([{fmt_ys}]))"


def _interp(x, xp, fp):
    if fp.ndim == 2:
        return np.array([np.interp(x, xp, fpp) for fpp in fp])
    elif xp.ndim == 2:
        return np.array([np.interp(x, xpp, fp) for xpp in xp])
    else:
        return np.interp(x, xp, fp)


_edge_count_cache = {}


@typechecked
def get_global_mr_curve(ex: Munch, relative_budget: bool = True) -> Curve:
    if relative_budget:
        dataset_name = ex.config.dataset
        if dataset_name not in _edge_count_cache:
            _edge_count_cache[dataset_name] = get_dataset(dataset_name)[0].triu(diagonal=1).sum().int().item()
        edge_count = _edge_count_cache[dataset_name]

    def x_coord(split_key, budget_key):
        if budget_key.startswith("used_budget"):
            used_budget = int(budget_key[12:])
        elif budget_key == "budget=00000":
            used_budget = 0
        else:
            pert_dict = ex.result.perturbations[split_key]["global"][budget_key]
            used_budget = len(pert_dict.get("edges", [])) + len(pert_dict.get("feats", []))
        return used_budget / edge_count if relative_budget else used_budget

    xs = sorted({x_coord(sk, bk) for sk, sd in ex.result.proba_margins.items() for bk in sd["global"]})

    mrs_per_split = []
    for spli_k, spl_dct in ex.result.proba_margins.items():
        x_to_mr = {x_coord(spli_k, budg_k): np.mean(margins <= 0) for budg_k, margins in spl_dct["global"].items()}
        # Sorting by x-coord is required for np.interp to work properly.
        x_to_mr = dict(sorted(x_to_mr.items()))
        mrs_per_split.append(np.interp(xs, list(x_to_mr.keys()), list(x_to_mr.values())))
    mrs_per_split = np.vstack(mrs_per_split)

    return Curve(np.array(xs), mrs_per_split)


_degree_list_cache = {}


@typechecked
def get_local_break_curves(ex: Munch, relative_budget: bool = True) -> List[Curve]:
    """Returns one curve for each attacked node."""

    if relative_budget:
        # Cache list versions of the degree tensors. This drastically improves performance.
        dataset_name = ex.config.dataset
        if dataset_name not in _degree_list_cache:
            _degree_list_cache[dataset_name] = get_dataset(dataset_name)[0].sum(dim=0).int().tolist()
        deg = _degree_list_cache[dataset_name]

    def for_node_in_split(split_key, node_key):
        div_by = deg[int(node_key[5:])] if relative_budget else 1
        xs = []
        ys = []
        # Note: the "sorted()" sorts by budget key.
        for budget_key, [margin] in sorted(ex.result.proba_margins.get(split_key, {}).get(node_key, {}).items()):
            pert_dict = ex.result.perturbations.get(split_key, {}).get(node_key, {}).get(budget_key, {})
            used_budget = len(pert_dict.get("edges", [])) + len(pert_dict.get("feats", []))
            xs.append(used_budget / div_by)
            ys.append(margin)
        ret = np.array(xs), np.array(ys) <= 0
        return ret

    curves = []
    # By using zip_longest, we get a None and thereby an error if different splits have different numbers of nodes.
    for split_and_node_keys in zip_longest(*(
            [(split_key, node_key) for node_key in split_v.keys()]
            for split_key, split_v in ex.result.proba_margins.items()
    )):
        res = list(starmap(for_node_in_split, split_and_node_keys))
        xs = np.array(sorted({x for split_xs, _ in res for x in split_xs}))
        ys_per_split = np.vstack([np.interp(xs, split_xs, split_ys) for split_xs, split_ys in res])
        curves.append(Curve(xs, ys_per_split))
    return curves


@typechecked
def summarize_breakage(exs: Union[Munch, Sequence[Munch]]) -> Tuple[Curve, np.ndarray]:
    if isinstance(exs, dict):
        exs = [exs]

    dataset_name = set(ex.config.dataset for ex in exs)
    attack_scope = set(ex.config.attack.scope for ex in exs)
    if len(dataset_name) != 1:
        raise ValueError(f"Got experiments for multiple datasets: {dataset_name}")
    if len(attack_scope) != 1:
        raise ValueError(f"Got experiments with different attack scopes: {attack_scope}")
    dataset_name = next(iter(dataset_name))
    attack_scope = next(iter(attack_scope))

    if attack_scope == "global":
        curve = Curve.max(map(Curve.monotonic, map(get_global_mr_curve, exs)))
        y_thresh = 1 - MLP_TEST_ACCURACY[dataset_name]
        alc_curve = curve.scaled(x=1 / 0.15, y=1 / y_thresh).terminated_at_x(1).terminated_at_y(1)
    else:
        all_benchmark_targets = get_all_benchmark_targets()
        total_local_target_nodes = len(all_benchmark_targets) * get_num_nodes_per_benchmark_target()

        exs_per_targets = {}
        for ex in exs:
            exs_per_targets.setdefault(ex.config.attack.targets, []).append(ex)

        if exs_per_targets.keys() != set(all_benchmark_targets):
            raise ValueError(f"Expected experiments from all 6 'targets', but only got: {set(exs_per_targets.keys())}")
        dist = {targets: len(grp) for targets, grp in exs_per_targets.items()}
        if len(set(dist.values())) != 1:
            raise ValueError(f"Not all 'targets' groups to have the same number of experiments: {dist}")

        # 3D array with dimensions (targets, attacks, nodes):
        cs_by_targets = [
            [[c.monotonic() for c in get_local_break_curves(ex)] for ex in exs]
            for exs in exs_per_targets.values()
        ]
        curve = Curve.sum(Curve.sum(map(Curve.max, zip(*cs_by_attack))) for cs_by_attack in cs_by_targets)

        alc_curve = curve.scaled(x=1 / 2, y=1 / total_local_target_nodes).terminated_at_x(1).terminated_at_y(1)

    return curve, alc_curve.alc_per_split


@typechecked
def find_perturbations_on_mr_envelope(exs: List[Munch]) \
        -> Dict[str, Dict[str, Tuple[Dict[str, Union[list, np.ndarray]], Munch, str]]]:
    ex_curves = [get_global_mr_curve(ex, relative_budget=False) for ex in exs]
    envelope_curve = Curve.max(map(Curve.monotonic, ex_curves))

    found_perts = {}
    for ex, ex_curve in zip(exs, ex_curves):
        for split_key, split_dict in ex.result.perturbations.items():
            split_idx = int(split_key[6:])
            for budget_key, pert_dct in split_dict["global"].items():
                used_budget = len(pert_dct.get("edges", [])) + len(pert_dct.get("feats", []))
                if abs(ex_curve.at(used_budget)[split_idx] - envelope_curve.at(used_budget)[split_idx]) < 1e-5:
                    found_perts.setdefault(split_key, {})[f"used_budget={used_budget:05d}"] = (pert_dct, ex, budget_key)
    return found_perts


@typechecked
def get_cross_model_envelope_alc_matrix(exs: Iterable[Munch]) -> Tuple[List[str], List[str], np.ndarray]:
    grouping = {}
    for ex in exs:
        if ex.config.attack.scope != "global":
            raise ValueError("Only supports global")
        if "cross_model" in ex.collection:
            key = ex.config.to_model_slug, ex.config.from_model_slug
        else:
            slug = model_slug(ex)
            key = slug, slug
        grouping.setdefault(key, []).append(ex)

    row_slugs = sorted({k[0] for k in grouping.keys()})
    col_slugs = sorted({k[1] for k in grouping.keys()})
    matrix = np.full((len(row_slugs), len(col_slugs)), np.nan)
    for (grp_row_slug, grp_col_slug), grp_exs in grouping.items():
        _, alc_per_split = summarize_breakage(grp_exs)
        matrix[row_slugs.index(grp_row_slug), col_slugs.index(grp_col_slug)] = alc_per_split.mean()
    return row_slugs, col_slugs, matrix


@typechecked
def plot_curve(
        curve: Curve, label: Optional[str] = None, color: Optional[str] = None, ls: Optional[Union[str, tuple]] = None,
        std: bool = True, secondary: bool = False, baseline: Union[bool, str] = False,
        ref_y: Optional[Float] = None, ref_label: Optional[str] = None,
        ax: Optional[plt.Axes] = None, **kwargs
) -> None:
    if ax is None:
        ax = plt.gca()

    if not hasattr(ax, "_gb_plotted_curve_before"):
        ax._gb_plotted_curve_before = True
        if baseline == "relative":
            ax._relative_baseline = curve
        if ref_y is not None:
            ref_kwargs = dict(label=ref_label, color="gray", lw=2, ls=(0, (3, 1)))
            if baseline == "relative":
                ax.plot(curve.xs, np.array(ref_y) - curve.mean.ys, zorder=2.8, **ref_kwargs)
                # ax.fill_between(curve.xs, np.array(ref_y) - curve.mean.ys, -105, alpha=0.2, zorder=0, **ref_kwargs)
            else:
                ax.axhline(ref_y, **ref_kwargs)
    elif baseline == "relative":
        raise ValueError("If there is a relative baseline, it must be plotted first")

    if hasattr(ax, "_relative_baseline"):
        curve = curve - ax._relative_baseline.mean

    if color is None:
        color = "black" if baseline else _cur_axis_color(advance=not secondary)
    if ls is None:
        ls = ":" if secondary else "-"
    if baseline:
        ax.plot(curve.xs, curve.mean.ys, **{**dict(ls=ls, color=color, label=label, zorder=2.5), **kwargs})
    else:
        ax.plot(curve.xs, curve.mean.ys, **{**dict(ls=ls, color=color, label=label), **kwargs})
        if std:
            ax.fill_between(
                curve.xs, curve.mean.ys - curve.std.ys, curve.mean.ys + curve.std.ys,
                **{**dict(ls=ls, color=color, alpha=0.1), **kwargs}
            )


@typechecked
def plot_global_mr_curve(ex_or_curve: Union[Munch, Curve], *, dataset_name: Optional[str] = None, **kwargs) -> None:
    if isinstance(ex_or_curve, Curve):
        curve = ex_or_curve
    else:
        curve = summarize_breakage(ex_or_curve)[0]
        dataset_name = ex_or_curve.config.dataset

    if dataset_name in MLP_TEST_ACCURACY:
        ref_label, ref_y = "MLP", 1 - MLP_TEST_ACCURACY[dataset_name]

    plot_curve(curve, ref_y=ref_y, ref_label=ref_label, **kwargs)

    plt.xlabel("Budget ÷ Edges")
    if hasattr(plt.gca(), "_gb_principal_baseline"):
        plt.ylabel("Test Set MR Difference (±σ)")
    else:
        plt.ylabel("Test Set MR (±σ)")


@typechecked
def plot_envelope_of_global_mr_curves(exs: Sequence[Munch], **kwargs) -> None:
    plot_global_mr_curve(summarize_breakage(exs)[0], dataset_name=exs[0].config.dataset, **kwargs)


@typechecked
def plot_sum_of_local_break_curves(exs: Sequence[Munch], **kwargs) -> None:
    if len(exs) != 6:
        raise ValueError(f"Expected 6 experiments, got {len(exs)}")
    plot_sum_of_envelopes_of_local_break_curves(exs, **kwargs)


@typechecked
def plot_sum_of_envelopes_of_local_break_curves(exs: Sequence[Munch], **kwargs) -> None:
    total_local_target_nodes = len(get_all_benchmark_targets()) * get_num_nodes_per_benchmark_target()

    plot_curve(summarize_breakage(exs)[0], **kwargs)

    plt.xlabel("Budget ÷ Degree")
    plt.xlim(left=0)
    if hasattr(plt.gca(), "_gb_principal_baseline"):
        plt.ylabel("Broken Nodes Difference (±σ)")
    else:
        plt.ylabel("Broken Nodes (±σ)")
        plt.ylim(bottom=0, top=total_local_target_nodes)


@typechecked
def plot_local_break_curve_per_targets(exs: Iterable[Munch]) -> None:
    nodes_pbt = get_num_nodes_per_benchmark_target()

    exs = sorted(exs, key=lambda ex: get_all_benchmark_targets().index(ex["config"]["attack"]["targets"]))

    for ex, color, shift in zip(exs, cycle(COLOR_CYCLE), range((len(exs) - 1) * nodes_pbt, -1, -nodes_pbt)):
        if shift != 0:
            plt.axhline(shift, c="gray", lw=1, ls="--")
        curve = Curve.sum(map(Curve.monotonic, get_local_break_curves(ex)))
        label = ex["config"]["attack"]["targets"]
        plt.plot(curve.xs, curve.mean.ys + shift, color=color, label=label)
        plt.fill_between(
            curve.xs, curve.mean.ys - curve.std.ys + shift, curve.mean.ys + curve.std.ys + shift,
            color=color, alpha=0.3
        )

    plt.legend()
    plt.xlabel("Budget ÷ Degree")
    plt.ylabel("Broken Nodes (±σ)")
    plt.yticks(range(0, len(exs) * nodes_pbt + 1, nodes_pbt), ["0"] + ["0\n20"] * (len(exs) - 1) + ["20"], fontsize=7)
    plt.xlim(left=0)
    plt.ylim(0, len(exs) * nodes_pbt)


@typechecked
def _cur_axis_color(advance: bool) -> str:
    ax = plt.gca()
    if not hasattr(ax, "gb_color_idx"):
        ax.gb_color_idx = 0
    elif advance:
        ax.gb_color_idx += 1
    return COLOR_CYCLE[ax.gb_color_idx % len(COLOR_CYCLE)]


@typechecked
def plot_matrix(
        values: np.ndarray, row_labels: Optional[List[str]] = None, col_labels: Optional[List[str]] = None,
        *, row_seps: Union[None, Int, Collection[Int]] = None, col_seps: Union[None, Int, Collection[Int]] = None,
        cmap: Union[str, Colormap] = "viridis", fontcolor_thresh: Float = 0.5, norm: Optional[plt.Normalize] = None,
        text_len: Int = 4, omit_leading_zero: bool = False, trailing_zeros: bool = False,
        highlight: Optional[str] = None, highlight_axis: Int = 1, highlight_color: str = "C3",
        grid: bool = True, angle_left: bool = False, cbar: bool = True, cbar_label: Optional[str] = None,
        ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[int, int]] = None, cellsize: Float = 0.65,
        title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None
) -> None:
    cmap = get_cmap(cmap)

    # Create figure if necessary.
    if ax is None:
        if figsize is None:
            # Note the extra width factor for the colorbar.
            figsize = (cellsize * values.shape[1] * (1.2 if cbar else 1), cellsize * values.shape[0])
        ax = plt.figure(figsize=figsize).gca()

    # Set title and axis labels if applicable.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if row_seps is not None:
        values = np.insert(values, row_seps, np.nan, axis=0)
        if row_labels is not None:
            row_labels = np.insert(row_labels, row_seps, "")
    if col_seps is not None:
        values = np.insert(values, col_seps, np.nan, axis=1)
        if col_labels is not None:
            col_labels = np.insert(col_labels, col_seps, "")

    # Plot the heatmap.
    im = ax.matshow(values, cmap=cmap, norm=norm)

    # Plot the text annotations showing each cell's value.
    norm_values = im.norm(values)
    for row, col in product(range(values.shape[0]), range(values.shape[1])):
        val = values[row, col]
        if not np.isnan(val):
            # Find text color.
            bg_color = cmap(norm_values[row, col])[:3]
            luma = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            color = "white" if luma < fontcolor_thresh else "black"

            # Plot cell text.
            annotation = _format_value(val, text_len, omit_leading_zero, trailing_zeros)
            ax.text(col, row, annotation, ha="center", va="center", color=color)

    # Add ticks and labels.
    if col_labels is None:
        ax.set_xticks([])
    else:
        col_labels = np.asarray(col_labels)
        labeled_cols = np.where(col_labels)[0]
        ax.set_xticks(labeled_cols)
        ax.set_xticklabels(col_labels[labeled_cols])
    if row_labels is None:
        ax.set_yticks([])
    else:
        row_labels = np.asarray(row_labels)
        labeled_rows = np.where(row_labels)[0]
        ax.set_yticks(labeled_rows)
        ax.set_yticklabels(row_labels[labeled_rows])

    ax.tick_params(which="major", bottom=False)

    plt.setp(ax.get_xticklabels(), rotation=40, ha="left", rotation_mode="anchor")

    # Turn off spines.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Rotate the left labels if applicable.
    if angle_left:
        plt.setp(ax.get_yticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    # Create the white grid if applicable.
    if grid:
        # Extra ticks required to avoid glitch.
        xticks = np.concatenate([[-0.56], np.arange(values.shape[1] + 1) - 0.5, [values.shape[1] - 0.44]])
        yticks = np.concatenate([[-0.56], np.arange(values.shape[0] + 1) - 0.5, [values.shape[0] - 0.44]])
        ax.set_xticks(xticks, minor=True)
        ax.set_yticks(yticks, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, top=False, left=False)

    # Highlight cells if applicable.
    if highlight is not None:
        if highlight == "lowest":
            func = np.nanargmin
        elif highlight == "highest":
            func = np.nanargmax
        else:
            raise ValueError(f"Unknown highlight: {highlight}")
        for row, col in (zip(func(values, 0), count()) if highlight_axis == 0 else zip(count(), func(values, 1))):
            ax.add_patch(Rectangle(
                (col - 0.45, row - 0.45), 0.91, 0.91, lw=2, edgecolor=highlight_color, facecolor="none", zorder=2
            ))

    # Create the colorbar if applicable.
    if cbar:
        bar = ax.figure.colorbar(im, ax=ax)
        bar.ax.set_ylabel(cbar_label)
        fmt = bar.ax.yaxis.get_major_formatter()
        if isinstance(fmt, FixedFormatter):
            fmt.seq = [_format_value(eval(re.sub(r"[a-z$\\{}]", "", label.replace("times", "*").replace("^", "**"))),
                                     text_len, omit_leading_zero, trailing_zeros)
                       if label else "" for label in fmt.seq]


def _format_value(val, text_len, omit_leading_zero, trailing_zeros):
    whole, fractional = str(float(val)).split(".")
    if fractional == "0":
        return whole
    else:
        fractional_len = text_len - len(whole) - 1
        if omit_leading_zero and whole in ("0", "-0"):
            fractional_len += 1
        if fractional_len <= 0:
            return whole
        whole, fractional = f"{val:.{fractional_len}f}".split(".")
        if not trailing_zeros:
            fractional = fractional.rstrip("0")
            if len(fractional) == 0:
                return whole
        if omit_leading_zero:
            if whole in ("0", "-0"):
                whole = whole[:-1]
            fractional = fractional[:text_len - len(whole) - 1]
        return f"{whole}.{fractional}"
