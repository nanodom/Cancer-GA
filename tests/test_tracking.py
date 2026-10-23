import numpy as np
import pytest

from cancer_ga import CancerGA
from cancer_ga.benchmarks.harvest import (
    generate_baseline,
    generate_paper_instance,
    harvest_bounds,
    harvest_is_integer,
    make_harvest_normalizer,
    make_harvest_objective,
)
from cancer_ga.utils.tracking import run_tracked


def _problem(contract_type="aggregated", seed_instance=2):
    inst = generate_paper_instance(n_lots=8, n_varieties=3, horizon_days=15, seed=seed_instance)
    plan, _, _, _, _ = generate_baseline(inst)
    obj = make_harvest_objective(inst, plan, contract_type=contract_type)
    norm = make_harvest_normalizer(inst, obj, n_samples=60)
    return inst, obj, norm


def _ga(inst, obj, seed):
    return CancerGA(
        objective_function=obj,
        n_dim=2 * inst.n_lots,
        bounds=harvest_bounds(inst),
        n_obj=2,
        is_integer=harvest_is_integer(inst),
        n_cancer=12,
        n_lymph=5,
        seed=seed,
    )


def test_run_tracked_matches_cancerga_run_for_same_seed():
    """Garantiza que el ciclo instrumentado no se desvía de CancerGA.run()."""
    inst, obj, norm = _problem()
    n_gen = 4

    x_ref, obj_ref, feas_ref = _ga(inst, obj, seed=11).run(n_generations=n_gen, verbose=False)
    out = run_tracked(_ga(inst, obj, seed=11), norm, max_generations=n_gen, patience=None)
    x_trk, obj_trk, feas_trk = out["best"]

    assert np.array_equal(x_ref, x_trk)
    assert np.allclose(obj_ref, obj_trk)
    assert feas_ref == feas_trk


def test_log_has_one_entry_per_generation_plus_initial():
    inst, obj, norm = _problem()
    out = run_tracked(_ga(inst, obj, 1), norm, max_generations=5, patience=None)
    assert [e["gen"] for e in out["log"]] == [0, 1, 2, 3, 4, 5]
    for key in ("hv", "best_f1", "min_f2", "archive_size", "cancer_size", "lymph_size", "elapsed_s"):
        assert all(key in e for e in out["log"])


def test_hypervolume_is_monotone_nondecreasing():
    inst, obj, norm = _problem()
    out = run_tracked(_ga(inst, obj, 3), norm, max_generations=8, patience=None)
    hvs = [e["hv"] for e in out["log"]]
    assert all(b >= a - 1e-12 for a, b in zip(hvs, hvs[1:]))
    assert hvs[-1] > 0


def test_best_f1_and_min_f2_never_get_worse():
    inst, obj, norm = _problem()
    out = run_tracked(_ga(inst, obj, 4), norm, max_generations=8, patience=None)
    f1 = [e["best_f1"] for e in out["log"]]
    f2 = [e["min_f2"] for e in out["log"]]
    assert all(b <= a + 1e-9 for a, b in zip(f1, f1[1:]))
    assert all(b <= a + 1e-9 for a, b in zip(f2, f2[1:]))


def test_early_stopping_triggers_after_patience_without_improvement():
    inst, obj, norm = _problem()
    # min_delta enorme => ninguna generación cuenta como mejora
    out = run_tracked(_ga(inst, obj, 5), norm, max_generations=50, patience=3, min_delta=10.0)
    assert out["stop_reason"] == "patience"
    assert out["generations_run"] == 3


def test_runs_until_max_generations_when_patience_disabled():
    inst, obj, norm = _problem()
    out = run_tracked(_ga(inst, obj, 5), norm, max_generations=4, patience=None)
    assert out["stop_reason"] == "max_generations"
    assert out["generations_run"] == 4


def test_same_seed_is_reproducible_and_different_seeds_differ():
    inst, obj, norm = _problem()

    def hv_curve(seed):
        out = run_tracked(_ga(inst, obj, seed), norm, max_generations=4, patience=None)
        return [e["hv"] for e in out["log"]], [e["best_f1"] for e in out["log"]]

    a1, a2 = hv_curve(7), hv_curve(7)
    b = hv_curve(8)
    assert a1 == a2
    assert a1 != b


def test_archive_is_nondominated():
    from cancer_ga.utils.metrics import nondominated_mask

    inst, obj, norm = _problem()
    out = run_tracked(_ga(inst, obj, 6), norm, max_generations=4, patience=None)
    objs = np.array([m["obj"] for m in out["archive"]])
    assert nondominated_mask(objs).all()


def test_rejects_non_biobjective():
    inst, obj, norm = _problem()
    ga = CancerGA(objective_function=lambda x: float(np.sum(x**2)), n_dim=3, bounds=(-1, 1), n_obj=1, seed=1)
    with pytest.raises(ValueError):
        run_tracked(ga, norm)
