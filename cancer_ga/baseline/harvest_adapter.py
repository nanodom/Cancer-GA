"""Adapter entre el modelo CP de cosecha y el SSA genérico.
No necesita `ga_instance`: usa la API funcional pura del benchmark."""

from __future__ import annotations
import numpy as np

from cancer_ga.benchmarks.harvest import (
    simulate_plan,
    make_contract_function,
    harvest_bounds,
    harvest_is_integer,
    plan_variability_index,
    make_oil_pvi_normalizer,
)


def _x0_from_plan(instance, plan, totals):
    n_lots = instance.n_lots
    team_gene = np.zeros(n_lots)
    for j in range(n_lots):
        per_k = np.zeros(instance.n_resources)
        for (jj, t, k), amt in plan.items():
            if jj == j:
                per_k[k] += amt
        team_gene[j] = int(np.argmax(per_k)) if per_k.sum() > 0 else 0

    first_day = np.full(n_lots, instance.horizon_days, dtype=float)
    for (j, t, k), amt in plan.items():
        if amt > 0:
            first_day[j] = min(first_day[j], t)
    priority = first_day / max(1, instance.horizon_days - 1)

    for j in range(n_lots):
        elig = instance.eligible_resources(j)
        if len(elig) and team_gene[j] not in elig:
            team_gene[j] = elig[np.argmin(np.abs(elig - team_gene[j]))]

    return np.concatenate([team_gene, priority])


def make_harvest_callbacks(
    instance, plan, totals, contract_type: str,
    n_contracts: int = 1, disabled_days=None,
    neighbor_scale: float = 0.15, neighbor_mask_prob: float = 0.3,
):
    n = instance.n_lots
    bounds = harvest_bounds(instance)
    is_int = harvest_is_integer(instance)
    f2_fn = make_contract_function(instance, plan, contract_type, n_contracts)
    baseline_plan = plan
    x0 = _x0_from_plan(instance, plan, totals)

    def evaluate_point(x):
        x = np.asarray(x, dtype=float)
        plan_new, _, oil = simulate_plan(instance, x[:n], x[n:], disabled_days=disabled_days)
        f2 = float(f2_fn(plan_new))
        pvi = float(plan_variability_index(instance, plan_new, baseline_plan))
        return {"oil": float(oil), "f2": f2, "pvi": pvi}

    def sample_neighbor(x, rng):
        x_new = np.asarray(x, dtype=float).copy()
        mask = rng.random(2 * n) < neighbor_mask_prob
        if not mask.any():
            mask[rng.integers(2 * n)] = True
        cont = (~is_int) & mask
        if cont.any():
            x_new[cont] += rng.normal(0.0, neighbor_scale, int(cont.sum()))
        int_mask = is_int & mask
        idx = np.where(int_mask)[0]
        if idx.size:
            x_new[idx] += rng.integers(-1, 2, idx.size)
        return np.clip(x_new, bounds[:, 0], bounds[:, 1])

    return x0, evaluate_point, sample_neighbor


def make_common_normalizer(instance, plan):
    """Normalizador común (aceite, PVI) — el mismo que usa CancerGA."""
    return make_oil_pvi_normalizer(instance, plan)