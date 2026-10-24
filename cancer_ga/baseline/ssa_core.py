"""Generic Simulated Annealing loop, problem-agnostic."""

from __future__ import annotations
import math
import time
from typing import Any, Callable
import numpy as np


def anneal(
    x0: Any,
    evaluate: Callable[[Any], float],
    sample_neighbor: Callable[[Any, np.random.Generator], Any | None],
    n_evaluations: int,
    T0: float = 1.0,
    Tf: float = 1e-3,
    rng: np.random.Generator | None = None,
    log_every: int | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    if n_evaluations <= 0:
        raise ValueError("n_evaluations must be > 0")
    if not (0 < Tf < T0):
        raise ValueError("Require 0 < Tf < T0")

    alpha = (Tf / T0) ** (1.0 / n_evaluations)
    x_curr = x0
    cost_curr = float(evaluate(x_curr))
    x_best, cost_best, best_eval = x_curr, cost_curr, 1
    T = T0
    log_every = log_every or max(1, n_evaluations // 50)
    trajectory = [{"evaluation": 1, "cost": cost_curr, "temperature": T}]
    n_eval = 1
    t0 = time.time()

    while n_eval < n_evaluations and T > Tf:
        neighbor = sample_neighbor(x_curr, rng)
        if neighbor is None:
            T *= alpha
            continue
        cost_new = float(evaluate(neighbor))
        n_eval += 1
        delta = cost_new - cost_curr
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            x_curr, cost_curr = neighbor, cost_new
            if cost_curr < cost_best:
                x_best, cost_best, best_eval = x_curr, cost_curr, n_eval
        T *= alpha
        if n_eval % log_every == 0:
            trajectory.append({"evaluation": n_eval, "cost": cost_curr, "temperature": T})

    trajectory.append({"evaluation": n_eval, "cost": cost_best, "temperature": T})
    return {
        "best_x": x_best,
        "best_cost": cost_best,
        "best_eval": best_eval,
        "trajectory": trajectory,
        "n_evaluations": n_eval,
        "elapsed_s": time.time() - t0,
    }