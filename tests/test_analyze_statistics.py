import pytest

from analyze_statistics import a12_oriented, collect_rows, holm_adjust


def test_holm_matches_hand_computation():
    # ordenados: 0.01*3=0.03 ; 0.03*2=0.06 ; 0.04*1=0.04 -> monótono: 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # monótono y con tope en 1


def test_a12_is_oriented_by_metric_direction():
    better_when_high = a12_oriented([10, 11, 12], [1, 2, 3], higher_is_better=True)
    better_when_low = a12_oriented([1, 2, 3], [10, 11, 12], higher_is_better=False)
    assert better_when_high == 1.0 and better_when_low == 1.0
    assert a12_oriented([1, 2, 3], [10, 11, 12], higher_is_better=True) == 0.0
    assert a12_oriented([1, 1], [1, 1]) == 0.5  # empates


def _run(oil, pvi, hv, sp):
    return {"representative": {"oil": oil, "pvi": pvi}, "hv_oil_pvi": hv, "spacing": {"spacing": sp}}


def _ssa(oil, pvi, hv, sp):
    return {"representatives": {"knee": {"oil": oil, "pvi": pvi}}, "hv_oil_pvi": hv, "spacing": {"spacing": sp}}


def test_collect_rows_orientation_and_independence():
    data = {
        "contracts": [
            {
                "label": "X",
                "runs": [_run(200 + i, 10 + i, 0.5 + i / 100, 0.1) for i in range(8)],
                "baseline_ssa": {"runs": [_ssa(100 + i, 50 + i, 0.2 + i / 100, 0.3) for i in range(8)]},
            }
        ]
    }
    rows = {r["metric"]: r for r in collect_rows(data)}
    # CancerGA claramente mejor en las 4 métricas -> A12 orientado = 1 en todas
    assert all(r["a12"] == pytest.approx(1.0) for r in rows.values())
    assert all(r["p_holm"] < 0.05 for r in rows.values())


def test_spacing_none_values_are_dropped_and_constant_samples_are_not_nan():
    data = {
        "contracts": [
            {
                "label": "X",
                "runs": [_run(5.0, 5.0, 0.5, None) for _ in range(6)],
                "baseline_ssa": {"runs": [_ssa(5.0, 5.0, 0.5, None) for _ in range(6)]},
            }
        ]
    }
    rows = collect_rows(data)
    assert "Spacing" not in {r["metric"] for r in rows}  # sin datos válidos: no se prueba
    assert all(r["p_holm"] == pytest.approx(1.0) for r in rows)  # constantes idénticas -> p = 1
