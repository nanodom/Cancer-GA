import numpy as np
from pathlib import Path
from cancer_ga import CancerGA
from cancer_ga.benchmarks.harvest import (
    generate_paper_instance,
    generate_baseline,
    simulate_plan,
    make_harvest_objective,
    harvest_bounds,
    harvest_is_integer,
    generate_tabali_instance,
    read_tabali_lots,
)

_TABALI_XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


def _small_instance(seed=1):
    return generate_paper_instance(n_lots=8, n_varieties=3, horizon_days=15, seed=seed)


def test_baseline_is_feasible_and_nonzero():
    inst = _small_instance()
    plan, totals, oil, team_gene, priority = generate_baseline(inst)

    assert oil > 0
    assert len(plan) > 0
    # capacidad diaria de la planta nunca se excede
    daily = {}
    for (j, t, k), amt in plan.items():
        daily[t] = daily.get(t, 0.0) + amt
    for t, amt in daily.items():
        assert amt <= inst.mill_capacity[t] + 1e-6
    # los lotes de ladera nunca son asignados a la máquina
    slope_lots = [j for j, lot in enumerate(inst.lots) if lot["slope"]]
    machine_idx = [i for i, r in enumerate(inst.resources) if r["kind"] == "Machine"]
    for j, t, k in plan:
        if j in slope_lots:
            assert k not in machine_idx


def test_simulate_plan_respects_lot_tonnage():
    inst = _small_instance()
    _, _, _, team_gene, priority = generate_baseline(inst)
    plan, _, _ = simulate_plan(inst, team_gene, priority)

    harvested_per_lot = np.zeros(inst.n_lots)
    for (j, t, k), amt in plan.items():
        harvested_per_lot[j] += amt

    assert np.all(harvested_per_lot <= inst.tons + 1e-6)


def test_cancerga_runs_on_harvest_cp_punctual():
    inst = _small_instance()
    plan, totals, oil, _, _ = generate_baseline(inst)
    obj = make_harvest_objective(inst, plan, totals, contract_type="punctual")

    ga = CancerGA(
        objective_function=obj,
        n_dim=2 * inst.n_lots,
        bounds=harvest_bounds(inst),
        n_obj=2,
        is_integer=harvest_is_integer(inst),
        n_cancer=12,
        n_lymph=5,
        seed=3,
    )
    result = ga.run(n_generations=4, verbose=False)

    assert result is not None
    best_x, best_objs, feasible = result
    assert len(best_objs) == 2
    assert feasible is True
    # f1 devuelto negado (minimización) => oil producida = -f1 >= 0
    assert -best_objs[0] >= 0


def test_cancerga_runs_on_harvest_cp_aggregated():
    inst = _small_instance(seed=2)
    plan, totals, oil, _, _ = generate_baseline(inst)
    obj = make_harvest_objective(inst, plan, totals, contract_type="aggregated")

    ga = CancerGA(
        objective_function=obj,
        n_dim=2 * inst.n_lots,
        bounds=harvest_bounds(inst),
        n_obj=2,
        is_integer=harvest_is_integer(inst),
        n_cancer=12,
        n_lymph=5,
        seed=3,
    )
    result = ga.run(n_generations=4, verbose=False)

    assert result is not None
    _, best_objs, feasible = result
    assert len(best_objs) == 2
    assert feasible is True


def test_read_tabali_lots_matches_real_scale():
    lots = read_tabali_lots(_TABALI_XLSX)
    assert len(lots) == 28
    varieties = {lot["variety"] for lot in lots}
    assert varieties == {"Arbequina", "Biancolilla", "Coratina", "Frantoio", "Leccino", "Noccelara"}
    # cuarteles 1..28 sin huecos
    assert sorted(lot["cuartel"] for lot in lots) == list(range(1, 29))


def test_generate_tabali_instance_calibrated():
    inst = generate_tabali_instance(_TABALI_XLSX, seed=1)

    assert inst.n_lots == 28
    # las toneladas por lote deben coincidir con kilos reales / 1000
    raw = read_tabali_lots(_TABALI_XLSX)
    expected_tons = sorted(r["kilos"] / 1000.0 for r in raw)
    assert np.allclose(sorted(inst.tons), expected_tons)

    # tasa de extracción real calibrada (~18.1%)
    assert 0.17 < inst.extraction_ratio < 0.19


def test_cancerga_runs_on_tabali_instance():
    inst = generate_tabali_instance(_TABALI_XLSX, horizon_days=20, seed=1)
    plan, totals, oil, _, _ = generate_baseline(inst)
    obj = make_harvest_objective(inst, plan, totals, contract_type="aggregated")

    ga = CancerGA(
        objective_function=obj,
        n_dim=2 * inst.n_lots,
        bounds=harvest_bounds(inst),
        n_obj=2,
        is_integer=harvest_is_integer(inst),
        n_cancer=15,
        n_lymph=6,
        seed=3,
    )
    result = ga.run(n_generations=4, verbose=False)

    assert result is not None
    _, best_objs, feasible = result
    assert len(best_objs) == 2
    assert feasible is True
