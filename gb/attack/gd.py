import math
from itertools import count
from typing import Union, Optional, Tuple, List, Callable

import numpy as np
import torch
from torch import autograd
from torchtyping import TensorType, patch_typeguard
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from gb.typing import Int, Float, IntSeq

patch_typeguard()
dim_1 = None
dim_2 = None
_GradFn = Callable[[TensorType["dim_1", "dim_2"]], TensorType["dim_1", "dim_2"]]
_LossFn = Callable[[TensorType["dim_1", "dim_2"]], Float]


@typechecked
def greedy_grad_descent(
        flip_shape: Tuple[Int, Int],
        symmetric: bool,
        device: torch.device,
        budgets: IntSeq,
        grad_fn: _GradFn,
        *, flips_per_iteration: Int = 1,
        max_iterations: Optional[Int] = None,
        progress: bool = True
) -> List[TensorType[-1, 2]]:
    """
    Each iteration, the entries with the largest gradient indicating the descent direction are greedily flipped. This is
    continued until either all the given budgets have been reached or the given max iterations have been exhausted. In
    the latter case, the perturbation that is "halfway there" to reaching the next higher budget is returned for that
    budget, but for the remaining budgets, no perturbations are returned. The gradients are obtained through a
    user-provided grad_fn.

    :param grad_fn: Takes a matrix indicating with 1s which entries should be flipped and computes its gradient w.r.t.
                    the loss that should be minimized.
    :return: The found perturbation for each budget. Not every budget necessarily receives a perturbation.
    """

    if list(budgets) != sorted(set(budgets)):
        raise ValueError("Budgets must be sorted and contain no duplicates")
    budgets = list(budgets)  # will modify this list, so copy it

    flip = torch.zeros(flip_shape, device=device, requires_grad=True)
    used_budget = 0
    perts = []

    pbar = tqdm(total=budgets[-1], leave=False) if progress and max_iterations != 1 else None
    for _ in range(max_iterations) if max_iterations is not None else count():
        if symmetric:
            flip_sym = _sym(flip)
            grad = autograd.grad(flip_sym, flip, grad_outputs=grad_fn(flip_sym))[0]
        else:
            grad = grad_fn(flip)

        with torch.no_grad():
            # Note: If we wanted to maximize the loss, the != would be a ==, but as we want to minimize it, we have to
            # take the "opposite" gradient.
            grad[(grad < 0) != (flip == 0)] = 0
            flt = grad.abs().flatten()
            # Note: When we only look for one entry to flip, use max() instead of topk() as it's a lot faster.
            for v, linear_idx in [flt.max(dim=0)] if flips_per_iteration == 1 else zip(*flt.topk(flips_per_iteration)):
                if v == 0:
                    break
                linear_idx = linear_idx.item()
                idx_2d = (linear_idx // flip.shape[1], linear_idx % flip.shape[1])
                # Case 1: The edge has not been flipped previously.
                if flip[idx_2d] == 0:
                    flip[idx_2d] = 1
                    used_budget += 1
                    # If we have reached the next higher budget, save its perturbation and drop the budget.
                    if used_budget == budgets[0]:
                        del budgets[0]
                        perts.append(flip.detach().nonzero())
                        # Stop if we have found perturbations for all budgets.
                        if len(budgets) == 0:
                            break
                # Case 2: The edge has been flipped previously, so flip it back.
                else:
                    flip[idx_2d] = 0
                    used_budget -= 1
        if pbar:
            pbar.update(used_budget - pbar.n)
        # Stop if we have found perturbations for all budgets.
        if len(budgets) == 0:
            break
    if pbar:
        pbar.close()

    # If there are still budgets left after having exhausted the maximum number of iterations, and we have found more
    # edges compared to the last perturbation we've emitted, i.e., we are "closer" to the next requested budget, emit
    # this "halfway there" perturbation as well.
    if len(budgets) != 0 and (len(perts) == 0 or perts[-1].shape[0] < used_budget):
        perts.append(flip.detach().nonzero())

    return perts


@typechecked
def proj_grad_descent(
        flip_shape_or_init: Union[Tuple[Int, Int], TensorType[-1, -1]],
        symmetric: bool,
        device: torch.device,
        budget: Int,
        grad_fn: _GradFn,
        loss_fn: _LossFn,
        *,
        iterations: Int = 200,
        base_lr: Float = 1e-1,
        xi: Float = 1e-5,
        grad_clip: Optional[Float] = None,
        sampling_tries: Int = 100,
        progress: bool = True
) -> Tuple[TensorType[-1, 2], Float]:
    """
    As proposed in: Topology attack and defense for graph neural networks: An optimization perspective.
    This is not a greedy attack, but instead tries to utilize the specified budget as optimally as possible.

    :param grad_fn: Takes a matrix indicating with larger numbers how much each entry should be flipped and computes
                    its gradient w.r.t. the loss that should be minimized.
    :param loss_fn: Takes a matrix indicating with 1s which entries should be flipped and returns a loss value that
                    should be minimized.
    :return: Both the found perturbation and its loss value.
    """

    if isinstance(flip_shape_or_init, TensorType):
        flip = flip_shape_or_init.triu(diagonal=1) if symmetric else flip_shape_or_init
        flip.requires_grad_()
    else:
        flip = torch.zeros(flip_shape_or_init, device=device, requires_grad=True)

    for itr in trange(iterations, leave=False) if progress else range(iterations):
        if symmetric:
            flip_sym = _sym(flip)
            grad = autograd.grad(flip_sym, flip, grad_outputs=grad_fn(flip_sym))[0]
        else:
            grad = grad_fn(flip)

        if grad_clip is not None:
            grad_len_sq = grad.square().sum()
            if grad_len_sq > grad_clip * grad_clip:
                grad *= grad_clip / grad_len_sq.sqrt()

        with torch.no_grad():
            lr = base_lr * budget / math.sqrt(itr + 1)
            flip -= lr * grad
            if flip.clamp(0, 1).sum() <= budget:
                flip.clamp_(0, 1)
            else:
                # Bisect to find mu up to a maximal error of xi.
                top = flip.max().item()
                # We clamp the bottom boundary since according to the paper, mu must be > 0. If mu were < 0, then we
                # would uniformly add some probability to EVERY edge, which derails the algorithm.
                bot = (flip.min() - 1).clamp_min(0).item()
                mu = (top + bot) / 2
                while (top - bot) / 2 > xi:
                    used_budget = (flip - mu).clamp(0, 1).sum()
                    if used_budget == budget:
                        break
                    elif used_budget > budget:
                        bot = mu
                    else:
                        top = mu
                    mu = (top + bot) / 2
                flip.sub_(mu).clamp_(0, 1)

    best_loss = np.inf
    best_pert = None

    flip.detach_()
    k = 0
    pbar = tqdm(total=sampling_tries, leave=False) if progress else None
    while k < sampling_tries:
        flip_sample = flip.bernoulli()
        if flip_sample.count_nonzero() <= budget:
            k += 1
            if pbar:
                pbar.update(1)
            loss = loss_fn(_sym(flip_sample) if symmetric else flip_sample)
            if loss < best_loss:
                best_loss = loss
                best_pert = flip_sample.nonzero()
    if pbar:
        pbar.close()

    return best_pert, best_loss


@typechecked
def _sym(triu: TensorType["dim_1", "dim_1"]) -> TensorType["dim_1", "dim_1"]:
    """
    Constructs a matrix whose values only depend on the upper triangle of "triu", and whose diagonal is
    always zero. This way, each edge in the graph is only parameterized by one number and not two, and the
    diagonal is not parameterized at all. Thereby, we avoid non-symmetric and diagonal adjustments of "triu".
    """

    triu = triu.triu(diagonal=1)
    return triu + triu.T
