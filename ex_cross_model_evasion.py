import logging
from typing import Any, Dict

from torchtyping import patch_typeguard
from typeguard import typechecked

from gb import metric
from gb.ana import model_slug, model_slug_config, find_perturbations_on_mr_envelope
from gb.exutil import *
from gb.util import fetch

patch_typeguard()

ex = make_experiment("cross_model_evasion")


@ex.config
def config():
    dataset = "cora"
    from_model_slug = "gcn_1"
    to_model_slug = "gcn_2"
    attack = {
        "scope": "global",
        "methods": [
            "brute_force_edges", "nettack_edges", "nettack_feats", "nettack_edges_feats", "fga_edges", "pgd_edges",
            "greedy_meta_edges", "pgd_meta_edges"
        ]
    }


@ex.capture
@typechecked
def do_run(dataset: str, from_model_slug: str, to_model_slug: str, attack: Dict[str, Any]) -> NonPrintingDict:
    if from_model_slug == to_model_slug:
        raise ValueError(f"The 'from' and 'to' model slugs are equal: {from_model_slug}")
    if attack["scope"] != "global":
        raise ValueError(f"Only 'global' is allowed as attack scope at the moment, not '{attack['scope']}'")

    logging.info("Loading configs of all relevant evasion runs...")
    evasion_exs = fetch(
        "evasion", ["config"], filter={
            "config.dataset": dataset,
            "config.attack.scope": attack["scope"],
            "config.attack.method": {"$in": attack["methods"]}
        }
    )
    logging.info(f"Selecting config for model slug '{to_model_slug}'...")
    to_config = model_slug_config(to_model_slug, evasion_exs)
    logging.info(f"Selecting IDs of evasion runs with model slug '{from_model_slug}'...")
    from_ex_ids = [ex._id for ex in evasion_exs if model_slug(ex) == from_model_slug]

    logging.info(f"Loading reference results of evasion runs with model slug '{from_model_slug}'...")
    ref_exs = fetch(
        "evasion", ["result"], filter={"_id": {"$in": from_ex_ids}}, incl_files={"perturbations", "proba_margins"}
    )

    logging.info("Finding perturbations lying on the misclassification rate envelope...")
    best_perts = find_perturbations_on_mr_envelope(ref_exs)

    A, X, y, N, D, C, split_keys, splits = prep_data(dataset)
    target_nodes_all = prep_target(dataset, attack["scope"], "global", [s[2] for s in splits])

    models = {}
    clean_scores = {}
    out_test_acc = {}
    for (split_idx, split_key), (train_nodes, val_nodes, test_nodes) in zip(enumerate(split_keys), splits):
        logging.info(f"{split_key}: Training clean model and getting its predictions...")
        set_seed("clean_model", split_idx, 0)
        model = make_attacked_model(A, D, C, to_config.model).to(A.device)
        # Make metric callbacks.
        metric_cbs = []
        reps = to_config.training["repetitions"]
        for rep in range(reps):
            metric_cb_base = f"{split_key}/clean_model" + (f"/repetition={rep}" if reps != 1 else "")
            metric_cbs.append(make_metric_cb(f"{metric_cb_base}/train_cost", f"{metric_cb_base}/val_cost"))
        # Fit model.
        set_seed("clean_train", split_key, 0)
        model_args = filter_model_args(model, A, X)
        ensure_contains(to_config.training, "max_epochs", "patience")
        model.fit(
            model_args, y, train_nodes, val_nodes, **to_config.training, progress=False, metric_cbs=metric_cbs
        )
        # Get model predictions.
        set_seed("clean_eval", split_key, 0)
        scores = model(*model_args).detach()
        # Save results.
        models[split_key] = model
        clean_scores[split_key] = scores
        out_test_acc[split_key] = metric.accuracy(scores[test_nodes], y[test_nodes]).item()

    out_scores = {}
    out_margins = {}
    for split_key, split_target_nodes in zip(split_keys, target_nodes_all):
        split_model = models[split_key]
        pert_dicts = [(ub_key, pert_dict) for ub_key, (pert_dict, _, _) in best_perts[split_key].items()]
        for target_key, target_nodes in split_target_nodes:  # always only "global"
            for ub_key, pert_dict in [("used_budget=00000", {})] + pert_dicts:
                if len(pert_dict) == 0:
                    scores = clean_scores[split_key]
                else:
                    logging.info(f"{split_key} {target_key} {ub_key}: Getting model's predictions under attack...")
                    submodel, A_s, X_s = full_input_submodel_with_args(split_model, A, X)
                    A_pert, X_pert = perturb_A_X(pert_dict, A_s, X_s)
                    set_seed("evasion_eval", split_key)
                    scores = submodel(A_pert, X_pert).detach()

                out_scores.setdefault(split_key, {}).setdefault(target_key, {})[ub_key] = scores[target_nodes].cpu()
                out_margins.setdefault(split_key, {}).setdefault(target_key, {})[ub_key] \
                    = metric.margin(scores[target_nodes].softmax(dim=-1), y[target_nodes]).cpu()

    logging.info("Done! Collecting results...")
    add_npz_artifact("scores", out_scores)
    add_npz_artifact("proba_margins", out_margins)
    return NonPrintingDict({
        "test_accuracy": out_test_acc,
        "perturbation_sources": {
            split_key: {ub_key: {"evasion_run_id": ex._id, "budget_key": bud_k} for ub_key, (_, ex, bud_k) in d.items()}
            for split_key, d in best_perts.items()
        }
    })


@ex.automain
def run() -> NonPrintingDict:
    return do_run()
