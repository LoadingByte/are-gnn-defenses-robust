# Black-Box Robustness Unit Test
# ==============================
#
# As part of our work, we provide the strongest global perturbations found against each evaluated model. This collection
# can serve as an advanced black-box unit test to evaluate the robustness of novel defenses. Still, we cannot stress
# enough that such a black-box test does not replace a proper adaptive evaluation.
#
# To make the unit test self-contained, it also includes our versions of Coral ML and Citeseer, as well as our exact
# data splits. This Python script demonstrates how to access and interpret the datasets, data splits, and perturbations.
# It can serve as inspiration on how to integrate the unit test into your codebase.


import numpy as np

# All data utilized by the unit test is condensed in this single file. It is just a ZIP archive, so feel free to explore
# its contents in a file browser.
with np.load("unit_test.npz") as loader:
    loader = dict(loader)

# The unit test provides perturbations for both evasion and poisoning. Select just the one scenario you are testing.
scenario_name = "evasion"  # or "poisoning"

# The unit test contains perturbations for two datasets: Cora ML and Citeseer. Because there are multiple versions of
# these datasets in circulation, we supply the right ones as part of the unit test.
for dataset_name in ["cora_ml", "citeseer"]:
    A_edges = loader[f"{dataset_name}/dataset/adjacency"]
    X_coords = loader[f"{dataset_name}/dataset/features"]
    y = loader[f"{dataset_name}/dataset/labels"]

    N = y.shape[0]
    D = X_coords[:, 1].max() + 1

    # The "A" (adjacency) and "X" (feature) matrices are loaded as dense arrays here. Working with dense data is usually
    # fine in the case of Cora ML and Citeseer, as both datasets are comparably small. However, if it better fits your
    # code, you can of course load them as sparse matrices as well. Be aware, however, that "A_edges" only contains half
    # of the 1-entries since the adjacency matrix is symmetric; as such, do not forget to symmetrize it!
    A = np.zeros((N, N))
    A[A_edges[:, 0], A_edges[:, 1]] = 1
    A[A_edges[:, 1], A_edges[:, 0]] = 1
    X = np.zeros((N, D))
    X[X_coords[:, 0], X_coords[:, 1]] = 1

    # Perturbations are available for five different train-val-test splits, all of them using 10-10-80 ratios. Each
    # split naturally has a different set of test nodes which are under attack. Once again, we recommend testing against
    # all splits.
    for split_number in [0, 1, 2, 3, 4]:
        train_nodes = loader[f"{dataset_name}/splits/{split_number}/train"]
        val_nodes = loader[f"{dataset_name}/splits/{split_number}/val"]
        test_nodes = loader[f"{dataset_name}/splits/{split_number}/test"]

        # Perturbations found against all of these models are available. For the most comprehensive evaluation, you
        # should test against all of them.
        for model_name in ["gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"]:

            # For each combination of the aforementioned variables, there are structure perturbations available across a
            # series of budgets. You probably want to evaluate against all budgets.
            prefix = f"{dataset_name}/perturbations/{scenario_name}/{model_name}/split_{split_number}/budget_"
            for pert_edges in (p for (key, p) in loader.items() if key.startswith(prefix)):
                # The perturbation "pert_edges" as it is stored in the unit test file is just a list of edges that must
                # be flipped. Once again, if you prefer to work with sparse matrices, feel free to do so instead, but
                # remain aware that the list of edges must be symmetrized!
                flipped = 1 - A[pert_edges[:, 0], pert_edges[:, 1]]
                A_perturbed = A.copy()
                A_perturbed[pert_edges[:, 0], pert_edges[:, 1]] = flipped
                A_perturbed[pert_edges[:, 1], pert_edges[:, 0]] = flipped

                # At this point, you have a dataset, a train-val-test split, and a perturbed adjacency matrix to your
                # disposal. It is now time to confront a defense with the perturbed adjacency matrix and record how much
                # its test set accuracy drops.
                # Depending on the chosen scenario, either feed the perturbed adjacency matrix into a model trained on
                # the clean adjacency matrix "A" (evasion), or train a poisoned model on the perturbed adjacency matrix
                # "A_perturbed" (poisoning).
