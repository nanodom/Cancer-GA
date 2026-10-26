import json
from pathlib import Path

import pytest

from cancer_ga.experiment import run_experiment, summarize_runs
from cancer_ga.utils.dashboard import build_dashboard_html

XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"
INSTANCE = {"kind": "tabali", "xlsx_path": str(XLSX), "horizon_days": 20, "seed": 42}
FAST = dict(ga_params={"n_cancer": 10, "n_lymph": 4}, max_generations=3, patience=None, norm_samples=30, progress=None)


@pytest.fixture(scope="module")
def results():
    return run_experiment(INSTANCE, contract_types=["punctual", "aggregated"], seeds=[0, 1], **FAST)


def test_structure_and_json_serializable(results):
    json.dumps(results, allow_nan=False)  # lanza si hubiera NaN/Inf
    assert [c["id"] for c in results["contracts"]] == ["punctual_1", "aggregated_1"]
    for c in results["contracts"]:
        assert [r["seed"] for r in c["runs"]] == [0, 1]
        assert c["summary"]["n_runs"] == 2
        for r in c["runs"]:
            assert r["log"][0]["gen"] == 0 and len(r["log"]) == 4
            assert len(r["front"]) >= 1
            assert len(r["representative"]["gantt"]) == results["meta"]["n_resources"]


def test_meta_records_reproducibility_info(results):
    meta = results["meta"]
    for key in ("seeds", "ga_params", "stop", "python", "numpy", "baseline_oil", "generated_at"):
        assert key in meta
    assert meta["seeds"] == [0, 1]


def test_same_configuration_is_reproducible(results):
    again = run_experiment(INSTANCE, contract_types=["punctual", "aggregated"], seeds=[0, 1], **FAST)
    for c1, c2 in zip(results["contracts"], again["contracts"]):
        for r1, r2 in zip(c1["runs"], c2["runs"]):
            assert [e["hv"] for e in r1["log"]] == [e["hv"] for e in r2["log"]]
            assert r1["representative"]["x"] == r2["representative"]["x"]


def test_parallel_matches_sequential(results):
    par = run_experiment(INSTANCE, contract_types=["punctual", "aggregated"], seeds=[0, 1], workers=2, **FAST)
    for c1, c2 in zip(results["contracts"], par["contracts"]):
        for r1, r2 in zip(c1["runs"], c2["runs"]):
            assert [e["hv"] for e in r1["log"]] == [e["hv"] for e in r2["log"]]


def test_punctual_runs_once_regardless_of_n_contracts():
    res = run_experiment(
        INSTANCE, contract_types=["punctual", "aggregated"], n_contracts_list=(1, 2, 3), seeds=[0], **FAST
    )
    assert [c["id"] for c in res["contracts"]] == ["punctual_1", "aggregated_1", "aggregated_2", "aggregated_3"]


def test_oil_pvi_hypervolume_uses_common_bounds_across_contracts(results):
    a, b = results["contracts"]
    assert a["common_normalizer"] == b["common_normalizer"]


def test_summarize_runs_statistics():
    runs = [
        {
            "representative": {"oil": o, "f2": 1.0, "pvi": 2.0},
            "hv": 0.5,
            "hv_oil_pvi": 0.4,
            "spacing": {"spacing": 0.1, "n_points": 5, "reason": None},
            "n_evaluations": 1000,
            "front": [{"nd": True}, {"nd": False}, {"nd": True}],
            "generations_run": g,
            "elapsed_s": 1.0,
            "stop_reason": s,
        }
        for o, g, s in [(10.0, 5, "patience"), (20.0, 7, "max_generations")]
    ]
    s = summarize_runs(runs)
    assert s["oil"]["mean"] == 15.0 and s["oil"]["min"] == 10.0 and s["oil"]["max"] == 20.0
    assert s["oil"]["std"] == pytest.approx(7.0710678, rel=1e-6)  # desv. muestral (ddof=1)
    assert s["stopped_by_patience"] == 1


def test_dashboard_is_self_contained(results, tmp_path):
    out = build_dashboard_html(results, tmp_path / "d.html")
    html = Path(out).read_text(encoding="utf-8")
    assert "cdnjs" not in html and "jsdelivr" not in html and 'src="http' not in html
    assert "Chart.js v4" in html
    assert "const DATA = " in html
    assert "f2 no es comparable entre tipos de contrato" in html


def test_representative_keeps_detail_needed_by_dashboard(results):
    """Regresión: el representante perdió gantt/x/resource_totals y el dashboard no podía dibujarlos."""
    for c in results["contracts"]:
        for r in c["runs"]:
            rep = r["representative"]
            for key in ("oil", "f2", "pvi", "gantt", "x", "resource_totals"):
                assert key in rep
            assert len(rep["resource_totals"]) == results["meta"]["n_resources"]


def test_representative_baseline_fallback_is_flagged(results):
    """El representante excluye el plan base; si el archivo sólo lo contiene, debe quedar marcado."""
    for c in results["contracts"]:
        for r in c["runs"]:
            assert r["representative_is_baseline"] == (r["representative"]["pvi"] <= 1e-9)
        assert c["summary"]["representative_is_baseline"] == sum(r["representative_is_baseline"] for r in c["runs"])


def test_evaluations_are_counted_and_gradient_is_off_by_default(results):
    assert results["meta"]["use_gradient"] is False
    for c in results["contracts"]:
        for r in c["runs"]:
            # sin gradiente: ~ (población x evaluaciones por generación), no ~(población x n_dim)
            assert 0 < r["n_evaluations"] < 5000
        assert c["summary"]["evaluations"]["mean"] > 0


def test_front_marks_nondominated_points_in_common_space(results):
    from cancer_ga.utils.metrics import nondominated_mask
    import numpy as np

    for c in results["contracts"]:
        for r in c["runs"]:
            pts = np.array([[-f["oil"], f["pvi"]] for f in r["front"]])
            assert [f["nd"] for f in r["front"]] == nondominated_mask(pts).tolist()


def test_degenerate_contract_is_skipped_and_reported():
    inst = {"kind": "tabali", "xlsx_path": str(XLSX), "horizon_days": 45, "seed": 42}
    res = run_experiment(inst, contract_types=["by_surface"], n_contracts_list=(1, 2), seeds=[0], **FAST)
    assert [c["id"] for c in res["contracts"]] == ["by_surface_2"]
    assert [s["id"] for s in res["meta"]["skipped_contracts"]] == ["by_surface_1"]
