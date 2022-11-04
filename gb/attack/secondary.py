import operator
from typing import Callable, Tuple

import numpy as np
from torchtyping import TensorType, patch_typeguard
from tqdm.auto import trange
from typeguard import typechecked

from gb.typing import Int, Float

patch_typeguard()
nodes = None


@typechecked
def repeat(
        attack_fn: Callable[[Int], Tuple[TensorType[-1, 2], Float]],
        repetitions: Int,
        *, progress: bool = True
) -> Tuple[TensorType[-1, 2], Float]:
    best_pert = None
    best_loss = np.inf
    for rep in trange(repetitions, leave=False) if progress else range(repetitions):
        pert, loss = attack_fn(rep)
        if loss < best_loss:
            best_pert = pert
            best_loss = loss
    return best_pert, best_loss


@typechecked
def reduce(
        loss_fn: Callable[[TensorType[-1, 2]], TensorType[()]],
        pert: TensorType[-1, 2],
        *, max_loss: Float, reduction_factor: Float = 0.95, gen_size: Int = 10, max_tries_per_gen: Int = 100
) -> TensorType[-1, 2]:
    curr_gen = [(pert, 0)]
    while curr_gen[0][0].shape[0] > 1:
        next_gen = []
        next_budget = int(curr_gen[0][0].shape[0] * reduction_factor)  # floor
        for _ in range(max_tries_per_gen):
            try_pert = curr_gen[np.random.randint(0, len(curr_gen))][0]
            try_pert = try_pert[np.random.choice(np.arange(try_pert.shape[0]), next_budget, replace=False)]
            loss = loss_fn(try_pert).item()
            if loss <= max_loss:
                next_gen.append((try_pert, loss))
                if len(next_gen) == gen_size:
                    break
        if not next_gen:
            break
        curr_gen = next_gen
    return min(curr_gen, key=operator.itemgetter(1))[0]
