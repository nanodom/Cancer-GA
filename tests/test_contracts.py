import numpy as np
import pytest

from cancer_ga.benchmarks.harvest import (
    CONTRACT_TYPES,
    generate_baseline,
    generate_paper_instance,
    make_contract_function,
    plan_variability_index,
    simulate_plan,
)


@pytest.fixture(scope="module")
def setup():
    inst = generate_paper_instance(n_lots=14, n_varieties=4, horizon_days=25, seed=3)
    plan, totals, oil, team_gene, priority = generate_baseline(inst)
    return inst, plan


def _random_plan(inst, seed):
    rng = np.random.default_rng(seed)
    n = inst.n_lots
    team = rng.integers(0, inst.n_resources, size=n).astype(float)
    prio = rng.random(n)
    plan, _, _ = simulate_plan(inst, team, prio)
    return plan


def _group_of(inst, k, n_contracts):
    from cancer_ga.benchmarks.harvest import CONTRACT_GROUPS

    kind = inst.resources[k]["kind"]
    return next(g for g, kinds in enumerate(CONTRACT_GROUPS[n_contracts]) if kind in kinds)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
@pytest.mark.parametrize("n_contracts", [1, 2, 3])
def test_baseline_plan_has_zero_discrepancy(setup, contract_type, n_contracts):
    inst, plan = setup
    f2 = make_contract_function(inst, plan, contract_type, n_contracts)
    assert f2(plan) == pytest.approx(0.0, abs=1e-9)


def test_pvi_is_zero_for_baseline_and_positive_otherwise(setup):
    inst, plan = setup
    assert plan_variability_index(inst, plan, plan) == pytest.approx(0.0)
    other = _random_plan(inst, 1)
    assert plan_variability_index(inst, other, plan) > 0


def test_pvi_matches_hand_computation(setup):
    inst, plan = setup
    shifted = {(j, t, k): amt for (j, t, k), amt in plan.items()}
    key = next(iter(shifted))
    shifted[key] += 2.5  # una sola desviación de 2.5 t
    assert plan_variability_index(inst, shifted, plan) == pytest.approx(2.5 / inst.horizon_days)


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_discrepancy_is_nonnegative(setup, contract_type):
    inst, plan = setup
    f2 = make_contract_function(inst, plan, contract_type, 2)
    for seed in range(5):
        assert f2(_random_plan(inst, seed)) >= 0.0


def test_aggregated_never_exceeds_punctual(setup):
    """Desigualdad triangular: agrupar recursos sólo puede reducir la discrepancia."""
    inst, plan = setup
    punctual = make_contract_function(inst, plan, "punctual", 1)
    for n_contracts in (1, 2, 3):
        agg = make_contract_function(inst, plan, "aggregated", n_contracts)
        for seed in range(6):
            other = _random_plan(inst, seed)
            assert agg(other) <= punctual(other) + 1e-9


def test_more_contracts_means_less_flexibility(setup):
    """Una partición más fina de recursos nunca baja la discrepancia agregada."""
    inst, plan = setup
    f1c = make_contract_function(inst, plan, "aggregated", 1)
    f2c = make_contract_function(inst, plan, "aggregated", 2)
    f3c = make_contract_function(inst, plan, "aggregated", 3)
    for seed in range(6):
        other = _random_plan(inst, seed)
        assert f1c(other) <= f2c(other) + 1e-9 <= f3c(other) + 2e-9


def test_minimum_guaranteed_ignores_surplus(setup):
    inst, plan = setup
    f2 = make_contract_function(inst, plan, "minimum_guaranteed", 1)
    surplus = {key: amt * 1.3 for key, amt in plan.items()}
    assert f2(surplus) == pytest.approx(0.0, abs=1e-9)
    shortfall = {key: amt * 0.5 for key, amt in plan.items()}
    expected = 0.5 * sum(plan.values())
    assert f2(shortfall) == pytest.approx(expected)


def test_by_work_hours_hand_computation(setup):
    """Mover `a` toneladas del recurso p al recurso q cambia las horas en
    a * H * (1/cap_q - 1/cap_p) (con 1 solo contrato)."""
    inst, plan = setup
    f2 = make_contract_function(inst, plan, "by_work_hours", 1)
    (j, t, k_from), amt = next(iter(plan.items()))
    k_to = next(k for k in range(inst.n_resources) if k != k_from and (j, t, k) not in plan)
    moved = dict(plan)
    del moved[(j, t, k_from)]
    moved[(j, t, k_to)] = amt
    expected = abs(amt * 8.0 * (1.0 / inst.capacity[k_to] - 1.0 / inst.capacity[k_from]))
    assert f2(moved) == pytest.approx(expected)


def test_by_surface_hand_computation(setup):
    """Pasar un lote completo a otro grupo cuesta surface_j * b_j."""
    inst, plan = setup
    n_contracts = 3
    f2 = make_contract_function(inst, plan, "by_surface", n_contracts)

    # lote cuya cosecha base está íntegramente en un único grupo
    by_lot = {}
    for (j, t, k), amt in plan.items():
        by_lot.setdefault(j, []).append(((j, t, k), amt))
    target = next(
        j for j, items in by_lot.items() if len({_group_of(inst, k, n_contracts) for (_, _, k), _ in items}) == 1
    )
    g_from = _group_of(inst, by_lot[target][0][0][2], n_contracts)
    k_other = next(k for k in range(inst.n_resources) if _group_of(inst, k, n_contracts) != g_from)

    moved = {}
    for (j, t, k), amt in plan.items():
        if j == target:
            moved[(j, t, k_other)] = moved.get((j, t, k_other), 0.0) + amt
        else:
            moved[(j, t, k)] = amt

    b = sum(a for _, a in by_lot[target]) / inst.tons[target]
    assert f2(moved) == pytest.approx(inst.surface[target] * b)


def test_invalid_arguments_raise(setup):
    inst, plan = setup
    with pytest.raises(ValueError):
        make_contract_function(inst, plan, "no_existe", 1)
    with pytest.raises(ValueError):
        make_contract_function(inst, plan, "punctual", 7)
