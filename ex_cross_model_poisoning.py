import logging
from typing import Any, Dict

from torchtyping import patch_typeguard
from typeguard import typechecked

from gb.ana import model_slug, model_slug_config, find_perturbations_on_mr_envelope
from gb.exutil import *
from gb.util import fetch

patch_typeguard()

ex = make_experiment("cross_model_poisoning")


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

    logging.info(f"Loading reference results of evasion and poisoning runs with model slug '{from_model_slug}'...")
    ref_evas_exs = fetch("evasion", ["result"], filter={"_id": {"$in": from_ex_ids}}, incl_files={"perturbations"})
    ref_exs = fetch(
        "trans_poisoning", ["config", "result"], incl_files={"proba_margins"},
        filter={"config.evasion_run_id": {"$in": from_ex_ids}, "config.use_evasion_seeds": {"$ne": True}}
    )
    if len(ref_exs) != len(from_ex_ids):
        raise ValueError("Could not find a reference poisoning experiment for each evasion experiment")
    for ref_ex in ref_exs:
        ref_evas_ex = next(evas_ex for evas_ex in ref_evas_exs if evas_ex._id == ref_ex.config.evasion_run_id)
        ref_ex.result.perturbations = ref_evas_ex.result.perturbations
    del ref_evas_exs

    logging.info("Finding perturbations lying on the misclassification rate envelope...")
    best_perts = find_perturbations_on_mr_envelope(ref_exs)

    out_test_acc, out_scores, out_margins = run_poisoning(
        dataset, attack["scope"], "global", to_config.model, to_config.training,
        {split_k: {"global": {ub_key: pert for ub_key, (pert, _, _) in d.items()}} for split_k, d in best_perts.items()}
    )

    logging.info("Done! Collecting results...")
    add_npz_artifact("scores", out_scores)
    add_npz_artifact("proba_margins", out_margins)
    return NonPrintingDict({
        "test_accuracy": out_test_acc,
        "perturbation_sources": {
            split_key: {
                ub_key: {"evasion_run_id": ex.config.evasion_run_id, "budget_key": budget_key}
                for ub_key, (_, ex, budget_key) in split_dict.items()
            } for split_key, split_dict in best_perts.items()
        }
    })


@ex.automain
def run() -> NonPrintingDict:
    return do_run()
