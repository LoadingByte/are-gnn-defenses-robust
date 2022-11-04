import logging
from itertools import zip_longest
from typing import Any, Union, Optional, Tuple, List, Dict, Callable

import torch
import torch.nn.functional as F
from munch import Munch
from torch import autograd, nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gb import metric
from gb.attack import brute_force, nettack, greedy_grad_descent, proj_grad_descent, repeat
from gb.exutil import *
from gb.model import GraphSequential, GCN, changed_fields
from gb.pert import edge_diff_matrix, sp_edge_diff_matrix, sp_feat_diff_matrix
from gb.typing import Int, Float
from gb.util import fetch

patch_typeguard()
nodes = None
classes = None
AdjMat = TensorType["nodes", "nodes"]
FeatMat = TensorType["nodes", "features"]
LabelVec = TensorType["nodes"]

SVD_GCN_ARCHS = ("svd_gcn", "svd_gcn_feats")
NETTACK_ATTACK_METHODS = ("nettack_edges", "nettack_feats", "nettack_edges_feats")
META_ATTACK_METHODS = ("greedy_meta_edges", "pgd_meta_edges")
NO_PERT: TensorType[0, 2] = torch.empty(0, 2, dtype=torch.int64)

ex = make_experiment("evasion")


@ex.config
def config():
    dataset = "cora"

    model = {
        "arch": "gcn",
        "hidden_dims": [64],
        "dropout": 0.5
    }
    training = {
        "repetitions": 1,
        "max_epochs": 3000,
        "patience": 50,
    }

    if model["arch"] in ("gcn", "jaccard_gcn", *SVD_GCN_ARCHS, "rgcn", "gnn_guard", "grand", "soft_median_gdc"):
        training["lr"] = 1e-2
        training["weight_decay"] = 1e-2
        if model["arch"] == "jaccard_gcn":
            model["threshold"] = 0.0
        elif model["arch"] in SVD_GCN_ARCHS:
            model["rank"] = 50
        elif model["arch"] == "rgcn":
            model["gamma"] = 1.0
            model["sqrt_eps"] = "auto"
            training["reg_kl"] = 5e-4
        elif model["arch"] == "gnn_guard":
            model["mimic_ref_impl"] = False
            model["prune_edges"] = False
            model["div_limit"] = "auto"
        elif model["arch"] == "grand":
            model["dropnode"] = 0.5
            model["order"] = 2
            model["mlp_input_dropout"] = 0.5
            training["n_samples"] = 2
            training["reg_consistency"] = 1.0
            training["sharpening_temperature"] = 0.5
        elif model["arch"] == "soft_median_gdc":
            model["teleport_proba"] = 0.15
            model["neighbors"] = 64
            model["temperature"] = 0.5
            model["only_weight_neighbors"] = True
    elif model["arch"] == "pro_gnn":
        training["adj_optim_interval"] = 2
        training["gnn_lr"] = 1e-2
        training["gnn_weight_decay"] = 5e-4
        training["adj_lr"] = 1e-2
        training["adj_momentum"] = 0.9
        training["reg_adj_deviate"] = 1.0
        training["reg_adj_l1"] = 5e-4
        training["reg_adj_nuclear"] = 1.5
        training["reg_feat_smooth"] = 1e-3

    attack = {
        "scope": "local",
        "method": "fga_edges"
    }

    if attack["scope"] == "global":
        if dataset == "citeseer":
            attack["budgets"] = [27, 55, 82, 110, 137, 165, 192, 220, 247, 275, 302, 330, 357, 385, 412, 440, 467, 495,
                                 522, 550]
        elif dataset == "cora":
            attack["budgets"] = [38, 76, 114, 152, 190, 228, 266, 304, 342, 380, 418, 456, 494, 532, 570, 608, 646, 684,
                                 722, 760]
    elif attack["scope"] == "local":
        attack["targets"] = "degree_1"
        if attack["targets"] in ("degree_1", "degree_2", "degree_3"):
            attack["budgets"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif attack["targets"] == "degree_5":
            attack["budgets"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
        elif attack["targets"] == "degree_8_to_10":
            attack["budgets"] = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
        elif attack["targets"] == "degree_15_to_25":
            attack["budgets"] = [1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
        else:
            raise ValueError(f"Unknown local attack targets: {attack['targets']}")

    if attack["method"] in ("brute_force_edges", "fga_edges", "pgd_edges", *META_ATTACK_METHODS):
        attack["loss"] = {
            "aggregation": "score_margin",
            "drop_layers": [],
            "freeze_layers": []
        }

    if model["arch"] == "pro_gnn":
        attack.setdefault("loss", {})["use_learned_adj"] = True

    if attack["method"] in ("brute_force_edges", *NETTACK_ATTACK_METHODS):
        attack["edge_diff_masks"] = {}
    elif attack["method"] in ("fga_edges", "pgd_edges", *META_ATTACK_METHODS):
        attack["loss"]["edge_diff_masks"] = {}

    if attack["method"] in ("fga_edges", "pgd_edges"):
        attack["loss"]["model_alteration"] = {}

    if attack["method"] == "brute_force_edges":
        attack["early_stop_loss"] = None if attack["scope"] == "global" else -5.0
        attack["edge_set_size"] = 1
    elif attack["method"] in NETTACK_ATTACK_METHODS:
        attack["surrogate"] = {
            "model": {
                "arch": "linear_unbiased_gcn",
                "hidden_dims": [64],
                "dropout": 0.0
            },
            "training": {
                "lr": 1e-2,
                "weight_decay": 0.0,
                "max_epochs": 3000,
                "patience": 50
            }
        }
    elif attack["method"] == "pgd_edges":
        attack["early_stop_loss"] = None if attack["scope"] == "global" else -5.0
        attack["repetitions"] = 1
        attack["iterations"] = 200
        attack["base_lr"] = 1e-1
        attack["xi"] = 1e-5
        attack["grad_clip"] = None
        attack["sampling_tries"] = 100
        attack["init_from_run_id"] = None
        attack["num_auxiliaries"] = 1
    elif attack["method"] in META_ATTACK_METHODS:
        attack["approximation"] = False
        attack["meta_adjustment"] = None
        attack["steady_training_seed"] = False
        if attack["method"] == "pgd_meta_edges":
            attack["iterations"] = 200
            attack["base_lr"] = 1e-2
            attack["xi"] = 1e-5
            attack["grad_clip"] = 1.0
            attack["sampling_tries"] = 100
            attack["init_from_run_id"] = None

    if not (attack["method"] in META_ATTACK_METHODS or
            (attack["method"] in NETTACK_ATTACK_METHODS and attack["surrogate"] is not None)):
        attack["auxiliary_adjustment"] = None

    for edge_masks in (attack.get("loss", {}).get("edge_diff_masks", {}), attack.get("edge_diff_masks", {})):
        for k in (f"e{n}" for n in range(1, 6)):
            edge_masks[k] = {"type": None}
            if edge_masks[k]["type"] is None:
                edge_masks.pop(k)
            else:
                edge_masks[k]["invert"] = False
                if edge_masks[k]["type"] == "receptive_field":
                    edge_masks[k]["steps"] = 2
                elif edge_masks[k]["type"] == "eigenspace_alignment":
                    edge_masks[k]["rank"] = 10
                    edge_masks[k]["symmetrize"] = "survivor_avg"
                if edge_masks[k]["type"] in ("jaccard", "eigenspace_alignment"):
                    edge_masks[k]["binarize"] = None
                    edge_masks[k]["cutoff"] = None
            del k
        del edge_masks


@ex.capture
@typechecked
def do_run(dataset: str, attack: Dict[str, Any]) -> NonPrintingDict:
    A, X, y, N, D, C, split_keys, splits = prep_data(dataset)
    target_nodes_all = prep_target(dataset, attack["scope"], attack.get("targets"), [s[2] for s in splits])

    # === Make models ===
    clean_models = make_clean_models(A, D, C, splits)
    auxiliary_models = make_auxiliary_models(A, D, C, splits)
    surrogate_models = make_surrogate_models(A, D, C, splits)

    # === Fit and evaluate models ===
    clean_scores = fit_and_evaluate_clean_models(A, X, y, split_keys, splits, clean_models)
    auxiliary_scores = fit_and_evaluate_auxiliary_models(A, X, y, split_keys, splits, auxiliary_models)
    surrogate_scores = fit_and_evaluate_surrogate_models(A, X, y, split_keys, splits, surrogate_models)

    # === Retrieve the initialization experiment's results ===
    init_ex = None
    init_from_run_id = attack.get("init_from_run_id")
    if init_from_run_id is not None:
        logging.info(f"Loading result of evasion run with ID {init_from_run_id}...")
        init_exs = fetch("evasion", ["result"], filter={"_id": init_from_run_id}, incl_files={"perturbations"})
        if len(init_exs) == 0:
            raise ValueError(f"There is no evasion experiment with ID {init_from_run_id}")
        init_ex = init_exs[0]

    # === Run evasion attacks ===
    perts = []
    for split_key, (train_nodes, val_nodes, _), split_clean_model, split_clean_scores, split_aux_models, \
        split_aux_scores, split_surr_model, split_target_nodes \
            in zip_longest(split_keys, splits, clean_models, clean_scores, auxiliary_models, auxiliary_scores,
                           surrogate_models, target_nodes_all):
        split_perts = []
        perts.append(split_perts)
        if split_aux_models is not None:
            models_and_margins = list(zip(split_aux_models, (metric.margin(s, y) for s in split_aux_scores)))
        else:
            models_and_margins = [(split_clean_model, metric.margin(split_clean_scores, y))]
        for target_key, target_nodes in split_target_nodes:
            logging.info(f"{split_key} {target_key}: Finding attacks...")
            set_seed("evasion_attack", split_key)
            split_perts.append([
                (budget, edge_pert.detach(), feat_pert.detach())  # detach so that the GC can free the memory
                for budget, edge_pert, feat_pert
                in run_attack(
                    split_key, target_key, A, X, y, C, train_nodes, val_nodes, target_nodes, models_and_margins,
                    split_surr_model, init_ex
                )
                if edge_pert.shape[0] != 0 or feat_pert.shape[0] != 0  # filter out empty perturbations
            ])

    # === Evaluate evasion attacks ===
    evas_scores = []
    for split_key, split_clean_model, split_target_nodes, split_perts \
            in zip(split_keys, clean_models, target_nodes_all, perts):
        split_evas_scores = []
        evas_scores.append(split_evas_scores)
        for (target_key, _), target_perts in zip(split_target_nodes, split_perts):
            target_evas_scores = []
            split_evas_scores.append(target_evas_scores)
            for budget, budget_edge_pert, budget_feat_pert in target_perts:
                budget_key = f"budget={budget:05d}"
                logging.info(f"{split_key} {target_key} {budget_key}: Getting model's predictions under attack...")
                submodel, A_s, X_s = full_input_submodel_with_args(split_clean_model, A, X)
                A_pert = A_s if budget_edge_pert.shape[0] == 0 else A_s + sp_edge_diff_matrix(budget_edge_pert, A_s)
                X_pert = X_s if budget_feat_pert.shape[0] == 0 else X_s + sp_feat_diff_matrix(budget_feat_pert, X_s)
                set_seed("evasion_eval", split_key)
                target_evas_scores.append(submodel(A_pert, X_pert).detach().cpu())

    # === Collect results ===
    logging.info("Done! Collecting results...")
    out_test_acc_clean = {}
    out_test_acc_aux = {}
    out_test_acc_surr = {}
    out_perts = {}
    out_scores = {}
    out_margins = {}
    y = y.cpu()  # required for computing the accuracy, since the scores are only on the CPU
    for split_key, (_, _, test_nodes), split_target_nodes, split_clean_scores, split_aux_scores, split_surr_scores, \
        split_perts, split_evas_scores \
            in zip_longest(split_keys, splits, target_nodes_all, clean_scores, auxiliary_scores, surrogate_scores,
                           perts, evas_scores):
        out_test_acc_clean[split_key] = metric.accuracy(split_clean_scores[test_nodes], y[test_nodes]).item()
        if split_aux_scores is not None:
            out_test_acc_aux[split_key] = [
                metric.accuracy(aux_scores[test_nodes], y[test_nodes]).item() for aux_scores in split_aux_scores
            ]
        if split_surr_scores is not None:
            out_test_acc_surr[split_key] = metric.accuracy(split_surr_scores[test_nodes], y[test_nodes]).item()
        for (target_key, target_nodes), target_perts, target_evas_scores \
                in zip(split_target_nodes, split_perts, split_evas_scores):
            target_clean_scores = split_clean_scores[target_nodes]
            target_clean_margins = metric.margin(target_clean_scores.softmax(dim=-1), y[target_nodes])
            out_scores.setdefault(split_key, {})[target_key] = {"budget=00000": target_clean_scores}
            out_margins.setdefault(split_key, {})[target_key] = {"budget=00000": target_clean_margins}
            for (budget, budget_edge_pert, budget_feat_pert), budget_evas_scores \
                    in zip(target_perts, target_evas_scores):
                budget_key = f"budget={budget:05d}"
                budget_pert_dict = {}
                if budget_edge_pert.shape[0] != 0:
                    budget_pert_dict["edges"] = budget_edge_pert.cpu()
                if budget_feat_pert.shape[0] != 0:
                    budget_pert_dict["feats"] = budget_feat_pert.cpu()
                target_budget_evas_scores = budget_evas_scores[target_nodes]
                target_budget_evas_margins = metric.margin(target_budget_evas_scores.softmax(dim=-1), y[target_nodes])
                out_perts.setdefault(split_key, {}).setdefault(target_key, {})[budget_key] = budget_pert_dict
                out_scores[split_key][target_key][budget_key] = target_budget_evas_scores
                out_margins[split_key][target_key][budget_key] = target_budget_evas_margins

    result_dict = {"test_accuracy_clean": out_test_acc_clean}
    if len(out_test_acc_aux) != 0:
        result_dict["test_accuracy_auxiliary"] = out_test_acc_aux
    if len(out_test_acc_surr) != 0:
        result_dict["test_accuracy_surrogate"] = out_test_acc_surr
    if attack["scope"] == "global":
        add_npz_artifact("perturbations", out_perts)
        add_npz_artifact("scores", out_scores)
        add_npz_artifact("proba_margins", out_margins)
    else:
        result_dict["perturbations"] = recursive_tensors_to_lists(out_perts)
        result_dict["scores"] = recursive_tensors_to_lists(out_scores)
        result_dict["proba_margins"] = recursive_tensors_to_lists(out_margins)
    return NonPrintingDict(result_dict)


@ex.capture
@typechecked
def make_clean_models(A: AdjMat, D: Int, C: Int, splits, model: Dict[str, Any]) -> List[nn.Module]:
    models = []
    for split_idx in range(len(splits)):
        set_seed("clean_model", split_idx, 0)
        models.append(make_attacked_model(A, D, C, model).to(A.device))
    return models


@ex.capture
@typechecked
def make_auxiliary_models(A: AdjMat, D: Int, C: Int, splits, model: Dict[str, Any], attack: Dict[str, Any]) \
        -> List[List[nn.Module]]:
    models = []
    num_aux = attack.get("num_auxiliaries", 1)
    aux_adjustment = attack.get("auxiliary_adjustment") or {}
    if num_aux != 1 or len(aux_adjustment) != 0:
        models = [[] for _ in range(len(splits))]
        aux_model_params = {**model, **aux_adjustment.get("model", {})}
        for n in range(num_aux):
            for split_idx in range(len(splits)):
                set_seed("clean_model", split_idx, n)
                models[split_idx].append(make_attacked_model(A, D, C, aux_model_params).to(A.device))
    return models


@ex.capture
@typechecked
def make_surrogate_models(A: AdjMat, D: Int, C: Int, splits, attack: Dict[str, Any]) -> List[nn.Module]:
    models = []
    if attack.get("surrogate") is not None:
        for split_idx in range(len(splits)):
            set_seed("surrogate_model", split_idx)
            models.append(make_surrogate_model(D, C).to(A.device))
    return models


@ex.capture
@typechecked
def make_surrogate_model(D: Int, C: Int, attack: Dict[str, Any]) -> nn.Module:
    params = attack["surrogate"]["model"]
    arch = params["arch"]
    if arch == "linear_unbiased_gcn":
        return GCN(n_feat=D, n_class=C, bias=False, activation="none", **sub_dict(params, "hidden_dims", "dropout"))
    else:
        raise ValueError(f"Unknown surrogate model arch: {arch}")


# Note: Quick experiments have shown that for us, training with sparse matrices would be slower than using dense ones.

@ex.capture
@typechecked
def fit_and_evaluate_clean_models(
        A: AdjMat, X: FeatMat, y: LabelVec, split_keys, splits, models: List[nn.Module],
        training: Dict[str, Any]
) -> List[TensorType]:
    scores = []
    for split_key, (train_nodes, val_nodes, _), split_model in zip(split_keys, splits, models):
        logging.info(f"{split_key}: Training clean model and getting its predictions...")
        ensure_contains(training, "repetitions", "max_epochs", "patience")
        model_args = filter_model_args(split_model, A, X)
        # Make metric callbacks.
        metric_cbs = []
        reps = training["repetitions"]
        for rep in range(reps):
            metric_cb_base = f"{split_key}/clean_model" + (f"/repetition={rep}" if reps != 1 else "")
            metric_cbs.append(make_metric_cb(f"{metric_cb_base}/train_cost", f"{metric_cb_base}/val_cost"))
        # Fit model.
        set_seed("clean_train", split_key, 0)
        split_model.fit(model_args, y, train_nodes, val_nodes, **training, progress=False, metric_cbs=metric_cbs)
        # Get model predictions.
        set_seed("clean_eval", split_key, 0)
        scores.append(split_model(*model_args).detach().cpu())
    return scores


@ex.capture
@typechecked
def fit_and_evaluate_auxiliary_models(
        A: AdjMat, X: FeatMat, y: LabelVec, split_keys, splits, models: List[List[nn.Module]],
        training: Dict[str, Any], attack: Dict[str, Any]
) -> List[List[TensorType]]:
    scores = []
    aux_training_params = {**training, **(attack.get("auxiliary_adjustment") or {}).get("training", {})}
    for split_key, (train_nodes, val_nodes, _), split_models in zip(split_keys, splits, models):
        split_aux_scores = []
        scores.append(split_aux_scores)
        for aux_idx, aux_model in enumerate(split_models):
            logging.info(f"{split_key}: Training auxiliary model {aux_idx} and getting its predictions...")
            model_args = filter_model_args(aux_model, A, X)
            # Make metric callbacks.
            metric_cbs = []
            reps = aux_training_params["repetitions"]
            for rep in range(reps):
                metric_cb_base = f"{split_key}/auxiliary={aux_idx}_model" + (f"/repetition={rep}" if reps != 1 else "")
                metric_cbs.append(make_metric_cb(f"{metric_cb_base}/train_cost", f"{metric_cb_base}/val_cost"))
            # Fit model.
            set_seed("clean_train", split_key, aux_idx)
            aux_model.fit(
                model_args, y, train_nodes, val_nodes, **aux_training_params, progress=False, metric_cbs=metric_cbs
            )
            # Get model predictions.
            set_seed("clean_eval", split_key, aux_idx)
            split_aux_scores.append(aux_model(*model_args).detach().cpu())
    return scores


@ex.capture
@typechecked
def fit_and_evaluate_surrogate_models(
        A: AdjMat, X: FeatMat, y: LabelVec, split_keys, splits, models: List[nn.Module],
        attack: Dict[str, Any]
) -> List[TensorType]:
    scores = []
    for split_key, (train_nodes, val_nodes, _), split_model in zip(split_keys, splits, models):
        logging.info(f"{split_key}: Training surrogate model and getting its predictions...")
        ensure_contains(attack["surrogate"]["training"], "lr", "weight_decay", "max_epochs", "patience")
        # Fit model.
        # We use the same split for the main and the surrogate model; this is in accordance to what Nettack does.
        set_seed("surrogate_train", split_key)
        split_model.fit(
            (A, X), y, train_nodes, val_nodes, **attack["surrogate"]["training"], progress=False,
            metric_cb=make_metric_cb(f"{split_key}/surrogate_model/train_cost", f"{split_key}/surrogate_model/val_cost")
        )
        # Get model predictions.
        set_seed("surrogate_eval", split_key)
        scores.append(split_model(A, X).detach().cpu())
    return scores


@ex.capture
@typechecked
def run_attack(
        split_key: str, target_key: str, A_root: AdjMat, X_root: FeatMat, y: LabelVec, C: Int,
        train_nodes: TensorType[-1], val_nodes: TensorType[-1], target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]], surrogate_model: Optional[nn.Module],
        init_ex: Optional[Munch],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    method = attack["method"]
    if method == "brute_force_edges":
        return run_brute_force_edges_attack(A_root, X_root, y, target_nodes, models_and_margins)
    elif method in NETTACK_ATTACK_METHODS:
        return run_nettack_attack(A_root, X_root, y, target_nodes, models_and_margins, surrogate_model)
    elif method == "fga_edges":
        return run_fga_edges_attack(A_root, X_root, y, target_nodes, models_and_margins)
    elif method == "pgd_edges":
        return run_pgd_edges_attack(split_key, target_key, A_root, X_root, y, target_nodes, models_and_margins, init_ex)
    elif method == "greedy_meta_edges":
        return run_greedy_meta_edges_attack(
            split_key, target_key, A_root, X_root, y, C, train_nodes, val_nodes, target_nodes, models_and_margins
        )
    elif method == "pgd_meta_edges":
        return run_pgd_meta_edges_attack(
            split_key, target_key, A_root, X_root, y, C, train_nodes, val_nodes, target_nodes, models_and_margins,
            init_ex
        )
    else:
        raise ValueError(f"Unknown attack method: {method}")


@ex.capture
@typechecked
def run_brute_force_edges_attack(
        A_root: AdjMat, X_root: FeatMat, y: LabelVec, target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    budgets = attack["budgets"]
    model, margins = single(models_and_margins)

    model, A_model, X_model = get_edge_attack_submodel_with_args(model, A_root, X_root)
    A_model, X_model = make_edge_attack_A_X_model(model, A_model, X_model)
    model = make_edge_attack_model(model)

    score_loss_fn = make_score_loss_fn(y, target_nodes, margins)
    raw_perts = brute_force(
        A_model,
        symmetric=True,
        loss_fn=lambda A_pert: score_loss_fn(model(A_pert, X_model)),
        mask=make_edge_mask(A_root, X_root, A_model, target_nodes, attack["edge_diff_masks"]),
        budgets=budgets, early_stop_loss=attack["early_stop_loss"], flip_set_size=attack["edge_set_size"],
        progress=False
    )
    return [(budget, pert, NO_PERT) for budget, pert in zip(budgets, raw_perts)]


@ex.capture
@typechecked
def run_nettack_attack(
        A_root: AdjMat, X_root: FeatMat, y: LabelVec, target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]], surrogate_model: Optional[nn.Module],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    if target_nodes.shape[0] != 1:
        raise ValueError("Nettack cannot be used for attacks targeting more than 1 node")

    src_model = surrogate_model if attack["surrogate"] is not None else single(models_and_margins)[0]
    src_model, A_model, X_model = get_edge_attack_submodel_with_args(src_model, A_root, X_root)
    if isinstance(src_model, GraphSequential):
        src_model = src_model.gcn
    if len(src_model.convs) != 2:
        raise ValueError("Nettack can only use surrogate models that have 2 layers")
    W1 = src_model.convs[0].weight
    W2 = src_model.convs[1].weight

    pert_edges = True
    pert_feats = True
    method = attack["method"]
    if method == "nettack_edges":
        pert_feats = False
    elif method == "nettack_feats":
        pert_edges = False

    budgets = attack["budgets"]
    max_edge_pert, max_feat_pert = nettack(
        A_model, X_model, y, W1, W2, target_nodes[0], max(budgets), pert_edges, pert_feats,
        A_mask=make_edge_mask(A_root, X_root, A_model, target_nodes, attack["edge_diff_masks"])
    )
    if len(max_edge_pert) != len(max_feat_pert):
        raise ValueError(f"Length of Nettack edge pert list ({len(max_edge_pert)}) doesn't match length of feat "
                         f"pert list ({len(max_feat_pert)})")
    # If the attack didn't find any perturbations, return an empty list.
    if len(max_edge_pert) == 0:
        return []

    perts = []
    for budget in budgets:
        perts.append((budget, torch.vstack(max_edge_pert[:budget]), torch.vstack(max_feat_pert[:budget])))
        # Stop early if we've exhausted the total number of found perturbations.
        if budget >= len(max_edge_pert):
            break
    return perts


@ex.capture
@typechecked
def run_fga_edges_attack(
        A_root: AdjMat, X_root: FeatMat, y: LabelVec, target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    budgets = attack["budgets"]
    model, margins = single(models_and_margins)
    loss_fn = make_grad_evasion_edge_attack_loss_fn(A_root, X_root, y, target_nodes, model, margins)
    raw_perts = greedy_grad_descent(
        A_root.shape, True, A_root.device, budgets,
        grad_fn=lambda A_flip: autograd.grad(loss_fn(A_flip), A_flip)[0],
        flips_per_iteration=max(budgets), max_iterations=1, progress=False
    )
    return [(budget, pert, NO_PERT) for budget, pert in zip(budgets, raw_perts)]


@ex.capture
@typechecked
def run_pgd_edges_attack(
        split_key: str, target_key: str, A_root: AdjMat, X_root: FeatMat, y: LabelVec, target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]], init_ex: Optional[Munch],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    pgd_kwargs = sub_dict(attack, "iterations", "base_lr", "xi", "grad_clip", "sampling_tries")
    repetitions = attack["repetitions"]
    early_stop_loss = attack["early_stop_loss"]

    loss_fns = [
        make_grad_evasion_edge_attack_loss_fn(A_root, X_root, y, target_nodes, model, margins)
        for model, margins in models_and_margins
    ]

    perts = []
    for budget in attack["budgets"]:
        budget_key = f"budget={budget:05d}"

        A_flip_init = None
        if init_ex is not None:
            init_perts = init_ex.result.perturbations[split_key][target_key]
            # If the attack from which we take init perts has been stopped early, also stop this attack early.
            if budget_key not in init_perts:
                break
            init_pert = init_perts[budget_key]["edges"]
            A_flip_init = edge_diff_matrix(torch.tensor(init_pert, device=A_root.device), torch.zeros_like(A_root))

        def attack_fn(repetition: Int):
            metric_cb = make_metric_cb(
                f"{split_key}/{target_key}/{budget_key}" +
                (f"/repetition={repetition}" if repetitions != 1 else "") + "/pgd_loss"
            )

            def loss_fn(A_flip):
                n_models = len(models_and_margins)
                model_idx = 0 if n_models == 1 else torch.randint(n_models, ())
                return loss_fns[model_idx](A_flip)

            def grad_fn(A_flip):
                loss = loss_fn(A_flip)
                metric_cb(loss.item())
                return autograd.grad(loss, A_flip)[0]

            return proj_grad_descent(
                A_root.shape if A_flip_init is None else A_flip_init, True, A_root.device, budget, grad_fn, loss_fn,
                **pgd_kwargs, progress=False
            )

        pert, loss = repeat(attack_fn, repetitions, progress=False)
        if pert.shape[0] != 0:
            perts.append((budget, pert, NO_PERT))
        # If an early stopping loss is configured and the loss surpasses it, stop early.
        if early_stop_loss is not None and loss <= early_stop_loss:
            break
    return perts


@ex.capture
@typechecked
def make_grad_evasion_edge_attack_loss_fn(
        A_root: AdjMat, X_root: FeatMat, y: LabelVec, target_nodes: TensorType[-1],
        model: nn.Module, margins: TensorType["nodes"],
        attack: Dict[str, Any]
) -> Callable[[AdjMat], TensorType[()]]:
    model, A_model, X_model = get_edge_attack_submodel_with_args(model, A_root, X_root)
    A_model, X_model = make_edge_attack_A_X_model(model, A_model, X_model)
    model = make_edge_attack_model(model)
    score_loss_fn = make_score_loss_fn(y, target_nodes, margins)
    A_mask = make_edge_mask(A_root, X_root, A_model, target_nodes, attack["loss"]["edge_diff_masks"])
    alter = attack["loss"]["model_alteration"]

    A_flipper = 1 - 2 * A_model

    def loss_fn(A_flip):
        A_diff = A_flip * A_flipper
        if A_mask is not None:
            A_diff = A_diff * A_mask
        with changed_fields(model, **alter):
            scores = model(A_model + A_diff, X_model)
        return score_loss_fn(scores)

    return loss_fn


@ex.capture
@typechecked
def run_greedy_meta_edges_attack(
        split_key: str, target_key: str, A_root: AdjMat, X_root: FeatMat, y: LabelVec, C: Int,
        train_nodes: TensorType[-1], val_nodes: TensorType[-1], target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    budgets = attack["budgets"]
    training_fn = make_meta_training_fun(
        split_key, A_root, X_root, y, C, train_nodes, val_nodes, target_nodes, models_and_margins
    )
    metric_cb = make_metric_cb(f"{split_key}/{target_key}/greedy_loss")

    def grad_fn(A_flip):
        loss, grad = training_fn(A_flip, True)
        metric_cb(loss.item())
        return grad

    raw_perts = greedy_grad_descent(
        A_root.shape, True, A_root.device, budgets, grad_fn,
        flips_per_iteration=1, max_iterations=2 * max(budgets), progress=False
    )

    return [(budget, pert, NO_PERT) for budget, pert in zip(budgets, raw_perts)]


@ex.capture
@typechecked
def run_pgd_meta_edges_attack(
        split_key: str, target_key: str, A_root: AdjMat, X_root: FeatMat, y: LabelVec, C: Int,
        train_nodes: TensorType[-1], val_nodes: TensorType[-1], target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]], init_ex: Optional[Munch],
        attack: Dict[str, Any]
) -> List[Tuple[int, TensorType[-1, 2], TensorType[-1, 2]]]:
    pgd_kwargs = sub_dict(attack, "iterations", "base_lr", "xi", "grad_clip", "sampling_tries")
    training_fn = make_meta_training_fun(
        split_key, A_root, X_root, y, C, train_nodes, val_nodes, target_nodes, models_and_margins
    )

    perts = []
    for budget in attack["budgets"]:
        budget_key = f"budget={budget:05d}"
        metric_cb = make_metric_cb(f"{split_key}/{target_key}/{budget_key}/pgd_loss")

        A_flip_init = None
        if init_ex is not None:
            init_perts = init_ex.result.perturbations[split_key][target_key]
            # If the attack from which we take init perts has been stopped early, also stop this attack early.
            if budget_key not in init_perts:
                break
            init_pert = init_perts[budget_key]["edges"]
            A_flip_init = edge_diff_matrix(torch.tensor(init_pert, device=A_root.device), torch.zeros_like(A_root))

        def grad_fn(A_flip):
            loss, grad = training_fn(A_flip, True)
            metric_cb(loss.item())
            return grad

        def loss_fn(A_flip):
            return training_fn(A_flip, False)

        pert, _ = proj_grad_descent(
            A_root.shape if A_flip_init is None else A_flip_init, True, A_root.device, budget, grad_fn, loss_fn,
            **pgd_kwargs, progress=False
        )
        if pert.shape[0] != 0:
            perts.append((budget, pert, NO_PERT))
    return perts


@ex.capture
@typechecked
def make_meta_training_fun(
        split_key: str, A_root: AdjMat, X_root: FeatMat, y: LabelVec, C: Int,
        train_nodes: TensorType[-1], val_nodes: TensorType[-1], target_nodes: TensorType[-1],
        models_and_margins: List[Tuple[nn.Module, TensorType["nodes"]]],
        model: Dict[str, Any], training: Dict[str, Any], attack: Dict[str, Any]
) -> Callable[[AdjMat, bool], Union[Float, Tuple[Float, AdjMat]]]:
    approx = attack["approximation"]
    steady_training_seed = attack["steady_training_seed"]
    meta_adjustment = attack.get("meta_adjustment") or {}
    meta_model_params = {**model, **meta_adjustment.get("model", {})}
    meta_training_params = {**training, **meta_adjustment.get("training", {})}

    ref_model, ref_margins = single(models_and_margins)
    A_model, X_model = make_edge_attack_A_X_model(ref_model, A_root, X_root)
    score_loss_fn = make_score_loss_fn(y, target_nodes, ref_margins)
    A_mask = make_edge_mask(A_root, X_root, A_model, target_nodes, attack["loss"]["edge_diff_masks"])

    A_flipper = 1 - 2 * A_model

    def training_fn(A_flip, comp_grad):
        if steady_training_seed:
            set_seed("meta_train", split_key)
        A_diff = A_flip * A_flipper
        if A_mask is not None:
            A_diff = A_diff * A_mask
        A_pert = A_model + A_diff
        att_model = make_attacked_model(A_pert, X_root.shape[1], C, meta_model_params).to(A_pert.device)
        att_model = make_edge_attack_model(att_model)
        model_args = filter_model_args(att_model, A_pert, X_model)
        if not comp_grad:
            att_model.fit(model_args, y, train_nodes, val_nodes, **meta_training_params, progress=False)
            return score_loss_fn(att_model(*model_args))
        elif approx:
            grad = torch.zeros_like(A_root)
            att_model.fit(
                model_args, y, train_nodes, val_nodes, **meta_training_params, progress=False,
                scores_cb=lambda scor: grad.add_(torch.autograd.grad(score_loss_fn(scor), A_flip, retain_graph=True)[0])
            )
            final_loss = score_loss_fn(att_model(*model_args))
            grad += torch.autograd.grad(final_loss, A_flip)[0]
            return final_loss, grad
        else:
            att_model.fit(
                model_args, y, train_nodes, val_nodes, **meta_training_params, differentiable=True, progress=False
            )
            loss = score_loss_fn(att_model(*model_args))
            return loss, autograd.grad(loss, A_flip)[0]

    return training_fn


@ex.capture
@typechecked
def get_edge_attack_submodel_with_args(
        model: nn.Module, A_root: AdjMat, X_root: FeatMat, attack: Dict[str, Any]
) -> Tuple[nn.Module, AdjMat, FeatMat]:
    model, A_model, X_model = full_input_submodel_with_args(model, A_root, X_root)
    if not attack.get("loss", {}).get("use_learned_adj", True):
        A_model = A_root
    return model, A_model, X_model


@ex.capture(prefix="attack")
@typechecked
def make_edge_attack_model(model: nn.Module, loss: Dict[str, Any]) -> nn.Module:
    # Drop all requested layers. All frozen preprocessing must naturally also be dropped to avoid it being duplicated.
    eff_drop_layers = loss["drop_layers"] + loss["freeze_layers"]
    if eff_drop_layers:
        if not isinstance(model, GraphSequential):
            raise ValueError("Can only drop or freeze layers if the model is a GraphSequential")
        model = model.sub(exclude=eff_drop_layers)
    return model


@ex.capture(prefix="attack")
@typechecked
def make_edge_attack_A_X_model(
        model: nn.Module, A_model: AdjMat, X_model: FeatMat, loss: Dict[str, Any]
) -> Tuple[AdjMat, FeatMat]:
    # If requested, apply certain preprocessing to A and X (_model) and use that as base (where the diff is added to).
    freeze_layers = loss["freeze_layers"]
    if freeze_layers:
        if not isinstance(model, GraphSequential):
            raise ValueError("Can only freeze layers if the model is a GraphSequential")
        frozen_layers = model.sub(include=freeze_layers, return_A_X=True)
        A_model, X_model = frozen_layers(A_model, X_model)
    return A_model, X_model


@ex.capture(prefix="attack")
@typechecked
def make_score_loss_fn(
        y: LabelVec, target_nodes: TensorType[-1], margins: TensorType["nodes"], loss: Dict[str, Any]
) -> Callable[[TensorType[..., "nodes", "classes"]], TensorType[()]]:
    sel = target_nodes  # Shortcut for the list of nodes which should play into the loss.
    agg = loss["aggregation"]
    if agg == "cross_entropy":
        return lambda scores: -F.cross_entropy(scores[..., sel, :], y[sel])
    elif agg == "masked_cross_entropy":
        sel = sel[margins[sel] > 0]
        return lambda scores: -F.cross_entropy(scores[..., sel, :], y[sel])
    elif agg == "score_margin":
        return lambda scores: metric.margin(scores[..., sel, :], y[sel]).mean(dim=-1)
    elif agg == "relu_score_margin":
        return lambda scores: metric.margin(scores[..., sel, :], y[sel]).relu().mean(dim=-1)
    elif agg == "elu_score_margin":
        return lambda scores: F.elu(metric.margin(scores[..., sel, :], y[sel])).mean(dim=-1)
    elif agg == "tanh_score_margin":
        return lambda scores: metric.margin(scores[..., sel, :], y[sel]).tanh().mean(dim=-1)
    elif agg == "proba_margin":
        return lambda scores: metric.margin(scores[..., sel, :].softmax(dim=-1), y[sel]).mean(dim=-1)
    elif agg == "relu_proba_margin":
        return lambda scores: metric.margin(scores[..., sel, :].softmax(dim=-1), y[sel]).relu().mean(dim=-1)
    else:
        raise ValueError(f"Unknown loss aggregation: {agg}")


@typechecked
def make_edge_mask(
        A_root: AdjMat, X_root: FeatMat, A_model: AdjMat, target_nodes: TensorType[-1],
        edge_masks: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[AdjMat]:
    if edge_masks is None or len(edge_masks) == 0:
        return None
    A_mask = torch.ones_like(A_root)
    for m in edge_masks.values():
        typ = m["type"]
        if typ == "receptive_field":
            A_cur_mask = metric.receptive_field(A_model, target_nodes, m["steps"])
        elif typ == "jaccard":
            A_cur_mask = metric.pairwise_jaccard(X_root)
        elif typ == "eigenspace_alignment":
            A_cur_mask = metric.eigenspace_alignment(A_root, m["rank"], m["symmetrize"])
        else:
            raise ValueError(f"Unknown edge mask type: {typ}")
        if m["invert"] is True:
            A_cur_mask = 1 - A_cur_mask
        binarize = m.get("binarize")
        cutoff = m.get("cutoff")
        if binarize is not None:
            A_cur_mask = A_cur_mask > binarize
        if cutoff is not None:
            A_cur_mask[A_cur_mask <= cutoff] = 0
        A_mask *= A_cur_mask
    return A_mask


@ex.automain
def run() -> NonPrintingDict:
    return do_run()
