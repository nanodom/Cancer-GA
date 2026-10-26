from pathlib import Path

import numpy as np
import pytest

from cancer_ga.baseline.harvest_adapter import make_common_normalizer, make_harvest_callbacks
from cancer_ga.baseline.ssa import run_ssa
from cancer_ga.benchmarks.harvest import (
    generate_baseline,
    generate_tabali_instance,
    make_harvest_normalizer,
    make_harvest_objective,
)

XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


@pytest.fixture(scope="module")
def tabali():
    instance = generate_tabali_instance(XLSX)
    plan, totals, _, _, _ = generate_baseline(instance)
    return instance, plan, totals, make_common_normalizer(instance, plan)


def _ssa(tabali, contract, n_contracts=1, seeds=(0,), n_evaluations=300, K=3):
    instance, plan, totals, common = tabali
    obj = make_harvest_objective(instance, plan, contract_type=contract, n_contracts=n_contracts)
    norm_contract = make_harvest_normalizer(instance, obj, n_samples=40)
    x0, ev, sn = make_harvest_callbacks(instance, plan, totals, contract, n_contracts=n_contracts)
    return run_ssa(x0, ev, sn, norm_contract, common, seeds=list(seeds), n_evaluations=n_evaluations, K=K)


@pytest.mark.parametrize("contract", ["punctual", "aggregated", "minimum_guaranteed", "by_surface", "by_work_hours"])
def test_adapter_smoke(tabali, contract):
    out = _ssa(tabali, contract, n_contracts=2 if contract == "by_surface" else 1, n_evaluations=500)
    r = out["runs"][0]
    assert len(r["front"]) >= 1
    assert r["n_evaluations_actual"] <= 500 + 3
    assert r["hv_oil_pvi"] is not None and 0.0 <= r["hv_oil_pvi"] <= 1.0 + 1e-9


def test_budget_is_respected_across_seeds(tabali):
    out = _ssa(tabali, "aggregated", seeds=(0, 1, 2), n_evaluations=300)
    assert all(r["n_evaluations_actual"] <= 300 + 3 for r in out["runs"])
    assert out["summary"]["n_runs"] == 3
    assert out["summary"]["evaluations"]["mean"] <= 303


def test_ssa_optimizes_the_contract_not_just_pvi(tabali):
    """Regresión: el SSA escalarizaba (aceite, PVI) e ignoraba f2, así que daba lo mismo en todos los
    contratos. Con la misma semilla y presupuesto, contratos distintos deben producir resultados distintos."""
    a = _ssa(tabali, "punctual", n_evaluations=600)["runs"][0]
    b = _ssa(tabali, "by_work_hours", n_evaluations=600)["runs"][0]
    assert [p["x"] for p in a["front"]] != [p["x"] for p in b["front"]]


def test_ssa_representative_is_not_the_baseline_plan(tabali):
    r = _ssa(tabali, "punctual", n_evaluations=600)["runs"][0]
    assert r["representative"]["pvi"] > 1e-9 or r["representative_is_baseline"]


def test_ssa_std_is_sample_std(tabali):
    out = _ssa(tabali, "aggregated", seeds=(0, 1, 2), n_evaluations=300)
    vals = [r["representatives"]["knee"]["oil"] for r in out["runs"]]
    assert out["summary"]["oil"]["std"] == pytest.approx(np.std(vals, ddof=1))
