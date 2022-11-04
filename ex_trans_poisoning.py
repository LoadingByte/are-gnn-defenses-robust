import logging

from torchtyping import patch_typeguard
from typeguard import typechecked

from gb.exutil import *
from gb.util import fetch

patch_typeguard()

ex = make_experiment("trans_poisoning")


@ex.config
def config():
    evasion_run_id = 0
    use_evasion_seeds = False


@ex.capture
@typechecked
def do_run(evasion_run_id: int, use_evasion_seeds: bool) -> NonPrintingDict:
    logging.info(f"Loading config and result of evasion run with ID {evasion_run_id}...")
    evasion_exs = fetch(
        "evasion", ["config", "result"], filter={"_id": evasion_run_id}, incl_files={"perturbations"}
    )
    if len(evasion_exs) == 0:
        raise ValueError(f"There is no evasion experiment with ID {evasion_run_id}")
    evasion_ex = evasion_exs[0]

    attack = evasion_ex.config.attack
    out_test_acc, out_scores, out_margins = run_poisoning(
        evasion_ex.config.dataset, attack.scope, attack.get("targets"), evasion_ex.config.model,
        evasion_ex.config.training, evasion_ex.result.perturbations, use_evasion_seeds
    )

    logging.info("Done! Collecting results...")
    if attack.scope == "global":
        add_npz_artifact("scores", out_scores)
        add_npz_artifact("proba_margins", out_margins)
        return NonPrintingDict({
            "test_accuracy": out_test_acc
        })
    else:
        return NonPrintingDict({
            "test_accuracy": out_test_acc,
            "scores": recursive_tensors_to_lists(out_scores),
            "proba_margins": recursive_tensors_to_lists(out_margins)
        })


@ex.automain
def run() -> NonPrintingDict:
    return do_run()
