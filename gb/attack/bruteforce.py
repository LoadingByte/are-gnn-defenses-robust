import gc
import math
from itertools import islice, chain
from typing import Optional, Tuple, List, Callable, Generator

import torch
from torchtyping import TensorType, patch_typeguard
from tqdm.auto import tqdm
from typeguard import typechecked

from gb.torchext import sp_tile
from gb.typing import Int, Float, IntSeq

patch_typeguard()
dim_1 = None
dim_2 = None


@typechecked
def brute_force(
        clean: TensorType["dim_1", "dim_2", torch.strided],
        symmetric: bool,
        loss_fn: Callable[[TensorType[..., "dim_1", "dim_2", torch.sparse_coo]], TensorType[...]],
        budgets: IntSeq,
        *, early_stop_loss: Optional[Float] = None,
        flip_set_size: Int = 1,
        mask: Optional[TensorType["dim_1", "dim_2", torch.strided]] = None,
        progress: bool = True
) -> List[TensorType[-1, 2]]:
    # Avoid unnecessary caches for backpropagation if, for some reason, the clean matrix requires gradient.
    clean = clean.detach()

    clean_sp = clean.to_sparse()

    if mask is None:
        mask = torch.ones_like(clean)
    if symmetric:
        mask = mask.triu(diagonal=1)
    n_all_flips = mask.count_nonzero()

    if list(budgets) != sorted(set(budgets)):
        raise ValueError("Budgets must be sorted and contain no duplicates")
    # Drop budgets requiring more flips than we have available.
    budgets = [b for b in budgets if b <= n_all_flips]
    if len(budgets) == 0:
        return []

    # Find the number of flip sets required in total to display a correct progress bar.
    n_flip_sets = 0
    # Start with the flip sets required by the steps between "bases".
    for base_budget in range(flip_set_size, budgets[-1] + 1, flip_set_size):
        n_flips = n_all_flips - (base_budget - flip_set_size)
        n_flip_sets += math.comb(n_flips, flip_set_size)
    # Add the flip sets required by the steps from "bases" to specific budgets.
    for budget in budgets:
        if budget % flip_set_size != 0:
            n_flips = n_all_flips - budget // flip_set_size
            n_flip_sets += math.comb(n_flips, budget % flip_set_size)

    perts = []

    pbar = tqdm(total=n_flip_sets, leave=False) if progress else None
    base_flips = torch.empty(0, 2, dtype=torch.int64, device=clean.device)
    for budget in budgets:
        # Advance to the next "base" if necessary.
        while base_flips.shape[0] <= budget - flip_set_size:
            if pbar:
                pbar.set_postfix({"budget": base_flips.shape[0] + flip_set_size})
            loss, new_base_flips = _round(
                clean, clean_sp, mask, symmetric, loss_fn, early_stop_loss, pbar, base_flips, flip_set_size
            )
            base_flips = torch.cat([base_flips, new_base_flips])
            mask[tuple(new_base_flips.T)] = 0
        # Find the edge set taking us from the last "base" to the specific budget if necessary.
        if budget == base_flips.shape[0]:
            pert = base_flips
        else:
            if pbar:
                pbar.set_postfix({"budget": budget})
            extra_flip_set_size = budget - base_flips.shape[0]
            loss, extra_flips = _round(
                clean, clean_sp, mask, symmetric, loss_fn, early_stop_loss, pbar, base_flips, extra_flip_set_size
            )
            pert = torch.cat([base_flips, extra_flips])
        perts.append(pert)
        # If the early stopping loss has been surpassed, stop the whole attack.
        if early_stop_loss is not None and loss <= early_stop_loss:
            break
    if pbar:
        pbar.close()

    return perts


@typechecked
def _round(
        clean: TensorType["dim_1", "dim_2", torch.strided],
        clean_sp: TensorType["dim_1", "dim_2", torch.sparse_coo],
        mask: TensorType["dim_1", "dim_2", torch.strided],
        symmetric: bool, loss_fn: callable, early_stop_loss: Optional[Float], pbar: Optional[tqdm],
        base_pert: TensorType[-1, 2], flip_set_size: Int
) -> Tuple[TensorType[()], TensorType[-1, 2]]:
    flips = mask.nonzero()
    # Prioritize flipping entries with larger mask values.
    flips = flips[mask[tuple(flips.T)].argsort(descending=True)]

    batch_size = 128
    bs_change_dir = 1
    bs_inc_factor = 2.0
    bs_regression_cnt = 0

    flip_comb_gen = _combinations(flips.shape[0], flip_set_size)
    best_loss = None
    best_flip_set = None
    while True:
        # If the change direction is set accordingly, adjust the batch size.
        if bs_change_dir == 1:
            batch_size = round(batch_size * bs_inc_factor)
        elif bs_change_dir == -1:
            batch_size = round(batch_size / bs_inc_factor)

        batch_flip_combs = list(islice(flip_comb_gen, batch_size))
        if len(batch_flip_combs) == 0:
            break

        try:
            min_loss, min_flip_set = _iteration(clean, clean_sp, symmetric, loss_fn, base_pert, flips, batch_flip_combs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Out of memory! First put this iteration's flip combinations at the front of the
                # flip combination generator such that they will be consumed in the next iterations.
                flip_comb_gen = chain(batch_flip_combs, flip_comb_gen)
                # Deallocate all unused memory on the GPU.
                gc.collect()
                torch.cuda.empty_cache()
                # Decrease the batch size with the same rate at which we have increased it.
                # Also remember that we have once again hit a limit.
                if bs_change_dir != -1:
                    bs_change_dir = -1
                    bs_regression_cnt += 1
                continue
            else:
                raise e
        if bs_change_dir == -1:
            # No longer out of memory even though we have been before. If we have hit a limit 4 times or more,
            # stay at the current batch size forever to avoid hitting limits over and over again. Otherwise,
            # go back to increasing the batch size, but with half of the previous rate.
            if bs_regression_cnt >= 4:
                bs_change_dir = 0
            else:
                bs_change_dir = 1
                bs_inc_factor = 0.5 + bs_inc_factor / 2

        if best_loss is None or min_loss < best_loss:
            if pbar:
                pbar.write(f"New best loss for budget={base_pert.shape[0] + flip_set_size}: {min_loss.item():.5f}")
            best_loss = min_loss
            best_flip_set = min_flip_set
        if pbar:
            pbar.update(len(batch_flip_combs))
        if early_stop_loss is not None and best_loss <= early_stop_loss:
            break

    return best_loss, best_flip_set


@typechecked
def _iteration(
        clean: TensorType["dim_1", "dim_2", torch.strided],
        clean_sp: TensorType["dim_1", "dim_2", torch.sparse_coo],
        symmetric: bool, loss_fn: callable, base_flips: TensorType[-1, 2],
        candidate_flips: TensorType[-1, 2], batch_flip_combs: List[tuple]
) -> Tuple[TensorType[()], TensorType[-1, 2]]:
    batch_flip_sets = candidate_flips[torch.tensor(batch_flip_combs)]
    cur_batch_size, flip_set_size, _ = batch_flip_sets.shape

    flip_list = torch.hstack([
        base_flips.expand(cur_batch_size, -1, -1),
        batch_flip_sets
    ]).view(-1, 2).T  # (2, cur_batch_size * (n_base_flips + flips_set_size))
    del batch_flip_sets  # free memory
    diff_indices = torch.vstack([
        torch.arange(cur_batch_size, device=clean.device).repeat_interleave(base_flips.shape[0] + flip_set_size),
        flip_list
    ])
    diff_values = 1 - clean[tuple(flip_list)]
    del flip_list  # free memory
    if symmetric:
        diff_indices = torch.hstack([diff_indices, diff_indices[[0, 2, 1]]])
        diff_values = diff_values.tile(2)
    diff = torch.sparse_coo_tensor(indices=diff_indices, values=diff_values, size=(cur_batch_size, *clean.shape))
    del diff_indices  # free memory
    del diff_values  # free memory
    # Note: We have observed a performance boost from coalescence, also it's required for some models.
    pert = (sp_tile(clean_sp, diff.shape[0]) + diff).coalesce()
    del diff  # free memory

    # Avoid unnecessary caches for backpropagation.
    with torch.no_grad():
        losses = loss_fn(pert)
    del pert  # free memory
    min_idx = losses.argmin().detach()  # detach to be able to free loss memory in a moment
    min_loss = losses[min_idx].detach()  # and once more
    del losses  # free memory
    min_edge_set = candidate_flips[torch.tensor(batch_flip_combs[min_idx])]
    return min_loss, min_edge_set


@typechecked
def _combinations(n: int, r: int) -> Generator[tuple, None, None]:
    """
    Returns the same combinations as itertools.combinations, but sorted such that all combinations with certain low
    numbers appear before any combination with a higher number.
    """
    if r <= 0:
        pass
    elif r == 1:
        for v1 in range(n):
            yield v1,
    # Special case for r=2 to improve performance.
    elif r == 2:
        for v2 in range(1, n):
            for v1 in range(v2):
                yield v1, v2
    else:
        for v_last in range(r - 1, n):
            for vs in _combinations(v_last, r - 1):
                yield *vs, v_last
