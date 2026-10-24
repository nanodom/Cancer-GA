import numpy as np
import pytest
from pathlib import Path

from cancer_ga.benchmarks.harvest import (
    generate_tabali_instance, generate_baseline,
)
from cancer_ga.baseline.harvest_adapter import make_harvest_callbacks, make_common_normalizer
from cancer_ga.baseline.ssa import run_ssa
from cancer_ga.utils.metrics import ObjectiveNormalizer


XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


@pytest.fixture(scope="module")
def tabali():
    instance = generate_tabali_instance(XLSX)
    plan, totals, _, _, _ = generate_baseline(instance)
    return instance, plan, totals


@pytest.mark.parametrize("contract", ["punctual", "aggregated", "minimum_guaranteed"])
def test_adapter_smoke(tabali, contract):
    instance, plan, totals = tabali
    x0, ev, sn = make_harvest_callbacks(instance, plan, totals, contract, n_contracts=1)
    norm = make_common_normalizer(instance, plan)

    d = ev(x0)
    assert set(d) == {"oil", "f2", "pvi"}
    assert d["oil"] >= 0 and d["pvi"] >= 0

    out = run_ssa(
        x0=x0, evaluate_point=ev, sample_neighbor=sn,
        common_normalizer=norm, seeds=[0],
        n_evaluations=500, K=3,
    )
    r = out["runs"][0]
    assert len(r["front"]) >= 1
    assert r["n_evaluations_actual"] <= 500 + 3
    # HV y spacing deben estar definidos y acotados
    assert r["hv_oil_pvi"] is not None
    assert 0.0 <= r["hv_oil_pvi"] <= 1.0 + 1e-9
    if len(r["front"]) >= 2:
        assert r["spacing"]["spacing"] is not None


def test_adapter_presupuesto_across_seeds(tabali):
    instance, plan, totals = tabali
    x0, ev, sn = make_harvest_callbacks(instance, plan, totals, "aggregated", n_contracts=1)
    norm = make_common_normalizer(instance, plan)
    out = run_ssa(x0, ev, sn, norm, seeds=[0, 1, 2], n_evaluations=300, K=3)
    for r in out["runs"]:
        assert r["n_evaluations_actual"] <= 300 + 3
    assert out["summary"]["n_runs"] == 3


def test_hv_decreases_with_fewer_points(tabali):
    """Verificación indirecta: HV de un punto debe ser <= HV del frente completo."""
    instance, plan, totals = tabali
    x0, ev, sn = make_harvest_callbacks(instance, plan, totals, "punctual", n_contracts=1)
    norm = make_common_normalizer(instance, plan)
    out = run_ssa(x0, ev, sn, norm, seeds=[0], n_evaluations=400, K=4)
    r = out["runs"][0]
    objs = np.array([[-p["oil"], p["pvi"]] for p in r["front"]])
    hv_full = norm.hypervolume(objs)
    hv_one = norm.hypervolume(objs[:1])
    assert hv_one <= hv_full + 1e-12