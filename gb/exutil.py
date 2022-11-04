import logging
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from lzma import LZMAFile
from tempfile import NamedTemporaryFile
from typing import Any, Union, Optional, Sequence, Tuple, List, Dict

import numpy as np
import seml
import torch
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb import metric, preprocess
from gb.data import get_dataset, get_splits, get_benchmark_target_nodes
from gb.model import GraphSequential, PreprocessA, PreprocessX, PreprocessAUsingXMetric, GCN, RGCN, ProGNN, GNNGuard, \
    GRAND, MLP, SoftMedianPropagation
from gb.pert import sp_edge_diff_matrix, sp_feat_diff_matrix
from gb.torchext import mul
from gb.typing import Int

__all__ = [
    "single", "ensure_contains", "sub_dict", "NonPrintingDict", "recursive_tensors_to_lists", "set_seed",
    "prep_data", "prep_target", "perturb_A_X", "make_attacked_model",
    "filter_model_args", "full_input_submodel_with_args",
    "make_experiment", "add_npz_artifact", "make_metric_cb", "run_poisoning"
]

patch_typeguard()
nodes = None
features = None


@typechecked
def single(seq: Sequence[Any]) -> Any:
    if len(seq) != 1:
        raise ValueError("Expected one value, but got multiple")
    return seq[0]


@typechecked
def ensure_contains(dct: dict, *keys: Any) -> None:
    for key in keys:
        if key not in dct:
            raise KeyError(f"Missing key: {key}")


@typechecked
def sub_dict(dct: dict, *filter_keys: Any) -> dict:
    # Note: This method raises a KeyError if a desired key is not found, and that is exactly what we want.
    return {key: dct[key] for key in filter_keys}


class NonPrintingDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "[NOT PRINTED]"


@typechecked
def recursive_tensors_to_lists(tree: dict) -> dict:
    for key in list(tree):
        if isinstance(tree[key], dict):
            recursive_tensors_to_lists(tree[key])
        else:
            tree[key] = tree[key].tolist()
    return tree


@typechecked
def set_seed(*stuff: Any) -> None:
    h = sha256()
    for x in stuff:
        if isinstance(x, int):
            x = str(x)
        h.update(x.encode())
    torch.manual_seed(int.from_bytes(h.digest()[:8], "big", signed=False))


@typechecked
def prep_data(dataset_name: str):
    logging.info(f"Loading dataset {dataset_name} and its train-val-test splits...")
    A, X, y = get_dataset(dataset_name)
    N, D = X.shape
    C = y.max().item() + 1
    splits = get_splits(y)
    split_keys = [f"split={split_idx}" for split_idx in range(len(splits))]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Transferring dataset to {device} device...")
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    return A, X, y, N, D, C, split_keys, splits


@typechecked
def prep_target(
        dataset_name: str, scope: str, targets_key: Optional[str], test_nodes_per_split: List[TensorType[-1]]
) -> List[List[Tuple[str, TensorType[-1]]]]:
    if scope == "global":
        return [[("global", test_nodes)] for test_nodes in test_nodes_per_split]
    elif scope == "local":
        return [
            [(f"node={target_node:05d}", target_node[None]) for target_node in split_target_nodes]
            for split_target_nodes in get_benchmark_target_nodes(dataset_name)[targets_key]
        ]
    else:
        raise ValueError(f"Unknown attack scope: {scope}")


@typechecked
def perturb_A_X(
        pert_dict: Dict[str, Union[list, np.ndarray]],
        A: TensorType["nodes", "nodes"],
        X: TensorType["nodes", "features"]
) -> Tuple[TensorType["nodes", "nodes"], TensorType["nodes", "features"]]:
    A_pert = A
    X_pert = X
    if "edges" in pert_dict:
        A_pert = A + sp_edge_diff_matrix(torch.tensor(pert_dict["edges"], device=A.device), A)
    if "feats" in pert_dict:
        X_pert = X + sp_feat_diff_matrix(torch.tensor(pert_dict["feats"], device=X.device), X)
    if len(set(pert_dict) - {"edges", "feats"}) != 0:
        raise ValueError(f"Unknown perturbation key: {set(pert_dict.keys())}")
    return A_pert, X_pert


@typechecked
def make_attacked_model(A: TensorType["nodes", "nodes"], D: Int, C: Int, params: Dict[str, Any]) -> nn.Module:
    arch = params["arch"]
    if arch == "gcn":
        return GCN(n_feat=D, n_class=C, bias=True, activation="relu", **sub_dict(params, "hidden_dims", "dropout"))
    elif arch == "jaccard_gcn":
        thresh = params["threshold"]
        return GraphSequential(OrderedDict(
            jaccard=PreprocessAUsingXMetric(
                lambda X: metric.pairwise_jaccard(X) > thresh,
                lambda A, A_mask: mul(A, A_mask)
            ),
            gcn=GCN(n_feat=D, n_class=C, bias=True, activation="relu", **sub_dict(params, "hidden_dims", "dropout"))
        ))
    elif arch == "svd_gcn":
        rank = params["rank"]
        return GraphSequential(OrderedDict(
            low_rank=PreprocessA(lambda A: preprocess.low_rank(A, rank)),
            gcn=GCN(n_feat=D, n_class=C, bias=True, activation="relu", **sub_dict(params, "hidden_dims", "dropout"))
        ))
    elif arch == "svd_gcn_feats":
        rank = params["rank"]
        return GraphSequential(OrderedDict(
            low_rank_A=PreprocessA(lambda A: preprocess.low_rank(A, rank)),
            low_rank_X=PreprocessX(lambda X: preprocess.low_rank(X, rank)),
            gcn=GCN(n_feat=D, n_class=C, bias=True, activation="relu", **sub_dict(params, "hidden_dims", "dropout"))
        ))
    elif arch == "rgcn":
        return RGCN(n_feat=D, n_class=C, **sub_dict(params, "hidden_dims", "gamma", "dropout", "sqrt_eps"))
    elif arch == "pro_gnn":
        gcn_kwargs = sub_dict(params, "hidden_dims", "dropout")
        return ProGNN(A, GCN(n_feat=D, n_class=C, bias=True, activation="relu", **gcn_kwargs))
    elif arch == "gnn_guard":
        kwargs = sub_dict(params, "hidden_dims", "mimic_ref_impl", "prune_edges", "dropout", "div_limit")
        return GNNGuard(n_feat=D, n_class=C, **kwargs)
    elif arch == "grand":
        return GRAND(
            MLP(n_feat=D, n_class=C, bias=True, **sub_dict(params, "hidden_dims", "dropout")),
            **sub_dict(params, "dropnode", "order", "mlp_input_dropout")
        )
    elif arch == "soft_median_gdc":
        ppr_kwargs = sub_dict(params, "teleport_proba", "neighbors")
        sm_kwargs = sub_dict(params, "temperature", "only_weight_neighbors")
        return GraphSequential(OrderedDict(
            ppr=PreprocessA(lambda A: preprocess.personalized_page_rank(A, **ppr_kwargs)),
            gcn=GCN(
                n_feat=D, n_class=C, bias=True, propagation=SoftMedianPropagation(**sm_kwargs),
                loops=False, activation="relu", **sub_dict(params, "hidden_dims", "dropout")
            )
        ))
    else:
        raise ValueError(f"Unknown model arch: {arch}")


@typechecked
def filter_model_args(
        model: nn.Module, A: TensorType["nodes", "nodes"], X: TensorType["nodes", "features"]
) -> Tuple[TensorType["nodes", -1], ...]:
    return (X,) if isinstance(model, ProGNN) else (A, X)


@typechecked
def full_input_submodel_with_args(
        model: nn.Module, A: TensorType["nodes", "nodes"], X: TensorType["nodes", "features"]
) -> Tuple[nn.Module, TensorType["nodes", "nodes"], TensorType["nodes", "features"]]:
    return (model.gnn, model.S, X) if isinstance(model, ProGNN) else (model, A, X)


_ex: Optional[Experiment] = None
_metrics: Dict[str, List[float]] = {}


@typechecked
def make_experiment(name: str) -> Experiment:
    global _ex
    if _ex is not None:
        raise RuntimeError("Cannot create multiple experiments")

    _ex = Experiment(name)
    _ex.captured_out_filter = apply_backspaces_and_linefeeds  # just in case sth uses tqdm

    seml.setup_logger(_ex)
    handler = _ex.logger.handlers[0]
    fmt = "(%(process)s %(threadName)s) " + handler.formatter._fmt
    handler.setFormatter(logging.Formatter(fmt, handler.formatter.datefmt))

    @_ex.post_run_hook
    @typechecked
    def collect_stats(_run: Run):
        seml.collect_exp_stats(_run)
        if len(_metrics) != 0:
            add_npz_artifact("metrics", {k: np.array(lst, dtype=np.float16) for k, lst in _metrics.items()})

    _ex.config(_seml_config)

    # Allow the process to be remotely debugged via the ptrace syscall.
    # First argument: PR_SET_PTRACER; second argument: PR_SET_PTRACER_ANY; both taken from <linux/prctl.h>
    import ctypes
    ctypes.CDLL(None).prctl(0x59616d61, -1, 0, 0, 0)
    # If debugging is still not possible for some reason, at least enable us to print all stack traces by sending
    # a USR1 signal to the offending process.
    import faulthandler
    from signal import SIGUSR1
    faulthandler.register(SIGUSR1)

    return _ex


def _seml_config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        _ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@typechecked
def add_npz_artifact(name: str, tree: dict) -> None:
    arrays_by_filename = {}

    def traverse(node: dict, prefix: str):
        for key, child in node.items():
            child_prefix = key if len(prefix) == 0 else f"{prefix}/{key}"
            if isinstance(child, dict):
                traverse(child, child_prefix)
            elif isinstance(child, torch.Tensor):
                arrays_by_filename[child_prefix] = child.numpy()
            elif isinstance(child, np.ndarray):
                arrays_by_filename[child_prefix] = child
            else:
                raise ValueError(f"Encountered child of unknown type: {type(child)}")

    traverse(tree, "")

    with NamedTemporaryFile(suffix=".npz.xz") as file:
        npz_buf = BytesIO()
        np.savez(npz_buf, **arrays_by_filename)
        with LZMAFile(file, "wb") as xz:
            xz.write(npz_buf.getvalue())
        file.flush()
        _ex.add_artifact(file.name, f"{name}.npz.xz", content_type="application/x-xz")


@typechecked
def make_metric_cb(*metric_names: str) -> callable:
    metric_lists = []
    for name in metric_names:
        if name in _metrics:
            raise ValueError(f"Metric defined 2 times: {name}")
        lst = []
        _metrics[name] = lst
        metric_lists.append(lst)

    def cb(*metric_values):
        if len(metric_lists) != len(metric_values):
            raise ValueError("Number of provided metric names doesn't match number of actual metrics")
        for lst, value in zip(metric_lists, metric_values):
            lst.append(value)

    return cb


@typechecked
def run_poisoning(
        dataset: str, attack_scope: str, attack_targets: Optional[str], model_params: Dict[str, Any],
        training_params: Dict[str, Any],
        perturbations: Dict[str, Dict[str, Dict[str, Dict[str, Union[list, np.ndarray]]]]],
        use_evasion_seeds: bool = False
) -> Tuple[dict, dict, dict]:
    A, X, y, N, D, C, split_keys, splits = prep_data(dataset)
    target_nodes_all = prep_target(dataset, attack_scope, attack_targets, [s[2] for s in splits])

    out_test_acc = {}
    out_scores = {}
    out_margins = {}

    budget_key_prefix = next(iter(next(iter(next(iter(perturbations.values())).values())).keys())).split("=")[0]
    for (split_idx, split_key), (train_nodes, val_nodes, test_nodes), split_target_nodes \
            in zip(enumerate(split_keys), splits, target_nodes_all):
        for target_key, target_nodes in split_target_nodes:
            target_perts = perturbations[split_key].get(target_key, {})
            for budget_key, budget_dict in [(f"{budget_key_prefix}=00000", {})] + list(target_perts.items()):
                A_pert, X_pert = perturb_A_X(budget_dict, A, X)

                logging.info(f"{split_key} {target_key} {budget_key}: Training a model with poisoned data...")
                if use_evasion_seeds:
                    set_seed("clean_model", split_idx, 0)
                else:
                    set_seed("poisoned_model", split_key)
                poisoned_model = make_attacked_model(A_pert, D, C, model_params).to(A.device)
                model_args = filter_model_args(poisoned_model, A_pert, X_pert)

                ensure_contains(training_params, "max_epochs", "patience")
                metric_cbs = []
                reps = training_params.get("repetitions", 1)
                for rep in range(reps):
                    metric_cb_base = f"{split_key}/{target_key}/{budget_key}/model" + \
                                     (f"/repetition={rep}" if reps != 1 else "")
                    metric_cbs.append(make_metric_cb(f"{metric_cb_base}/train_cost", f"{metric_cb_base}/val_cost"))
                if use_evasion_seeds:
                    set_seed("clean_train", split_key, 0)
                else:
                    set_seed("poisoned_train", split_key)
                poisoned_model.fit(
                    model_args, y, train_nodes, val_nodes, **training_params, progress=False, metric_cbs=metric_cbs
                )

                if use_evasion_seeds:
                    if len(budget_dict) == 0:
                        set_seed("clean_eval", split_key, 0)
                    else:
                        set_seed("evasion_eval", split_key)
                else:
                    set_seed("poisoned_eval", split_key)
                scores = poisoned_model(*model_args).detach()
                out_test_acc.setdefault(split_key, {}).setdefault(target_key, {})[budget_key] \
                    = metric.accuracy(scores[test_nodes], y[test_nodes]).item()
                out_scores.setdefault(split_key, {}).setdefault(target_key, {})[budget_key] \
                    = scores[target_nodes].cpu()
                out_margins.setdefault(split_key, {}).setdefault(target_key, {})[budget_key] \
                    = metric.margin(scores[target_nodes].softmax(dim=-1), y[target_nodes]).cpu()

                del poisoned_model
                del scores

    return out_test_acc, out_scores, out_margins
