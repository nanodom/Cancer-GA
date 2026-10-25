"""
SSA baseline para el CP de cosecha, con escalarización por pesos.

El SSA optimiza el objetivo del contrato (f2) pero reporta sus métricas
en el espacio común de (aceite, PVI) para que sea directamente comparable
con CancerGA.
"""

from __future__ import annotations
import time
import numpy as np

from cancer_ga.baseline.ssa_core import anneal
from cancer_ga.utils.metrics import ObjectiveNormalizer, nondominated_mask
from cancer_ga.utils.spacing import spacing_to_front


def generate_weights(K: int) -> list[tuple[float, float]]:
    if K < 2:
        raise ValueError("K must be >= 2")
    return [(k / (K - 1), 1.0 - k / (K - 1)) for k in range(K)]


def _to_normalizer(n) -> ObjectiveNormalizer:
    if isinstance(n, ObjectiveNormalizer):
        return n
    return ObjectiveNormalizer(n["ideal"], n["nadir"])


def _scalarized_cost(oil: float, f2: float, w1: float, w2: float,
                     norm_contract: ObjectiveNormalizer) -> float:
    """
    Costo escalarizado en el espacio del contrato: [-oil, f2].
    norm = norm_contract([-oil, f2]) ∈ [0, 1]².
    cost = w1 * norm[0] + w2 * norm[1].
    """
    n = norm_contract(np.array([[-oil, f2]]))[0]
    return float(w1 * n[0] + w2 * n[1])


def _pareto_filter_f2(points: list[dict]) -> list[dict]:
    """Filtra los puntos no dominados maximizando oil y minimizando f2."""
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q["oil"] >= p["oil"] and q["f2"] <= p["f2"]:
                if q["oil"] > p["oil"] or q["f2"] < p["f2"]:
                    dominated = True
                    break
        if not dominated:
            keep.append(p)
    return keep


def _run_one_seed(x0, evaluate_point, sample_neighbor,
                  norm_contract: ObjectiveNormalizer,
                  common_normalizer: ObjectiveNormalizer,
                  n_evaluations, K, seed, T0, Tf):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    weights = generate_weights(K)
    n_per_w = max(1, n_evaluations // K)

    all_pts, ssa_runs = [], []
    total_evals = 0

    for (w1, w2) in weights:
        rng_w = np.random.default_rng(rng.integers(0, 2**63 - 1))

        def make_cost(w1=w1, w2=w2):
            def cost(x):
                d = evaluate_point(x)
                return _scalarized_cost(d["oil"], d["f2"], w1, w2, norm_contract)
            return cost

        res = anneal(
            x0=x0, evaluate=make_cost(),
            sample_neighbor=lambda x, r: sample_neighbor(x, r),
            n_evaluations=n_per_w, T0=T0, Tf=Tf, rng=rng_w,
        )

        d = evaluate_point(res["best_x"])
        all_pts.append({
            "oil": float(d["oil"]), "f2": float(d["f2"]), "pvi": float(d["pvi"]),
            "weight": [float(w1), float(w2)],
            "x": list(np.asarray(res["best_x"], dtype=float)),
            "evaluations": res["n_evaluations"],
        })
        total_evals += res["n_evaluations"]
        ssa_runs.append({
            "weight": [float(w1), float(w2)],
            "best_cost": float(res["best_cost"]),
            "n_evaluations": res["n_evaluations"],
            "trajectory": res["trajectory"],
        })

    # 1. Frente de Pareto en el espacio del contrato [-oil, f2]
    front_f2 = _pareto_filter_f2(all_pts)
    
    # 2. Proyectar al espacio común [-oil, pvi] y re-filtrar dominancia
    # (Igual que en CancerGA, para que el Spacing y el HV sean comparables)
    pvi_objs = np.array([[-p["oil"], p["pvi"]] for p in front_f2])
    nd_mask_pvi = nondominated_mask(pvi_objs)
    front = [front_f2[i] for i in range(len(front_f2)) if nd_mask_pvi[i]]
    
    front.sort(key=lambda p: p["oil"])

    # 3. Representantes (Knee) en el espacio común, excluyendo el plan base (pvi=0)
    valid_reps = [i for i, p in enumerate(front) if p["pvi"] > 1e-9]
    if not valid_reps:
        valid_reps = list(range(len(front)))

    objs_pvi = np.array([[-front[i]["oil"], front[i]["pvi"]] for i in valid_reps])
    norm_pvi = common_normalizer(objs_pvi)
    i_knee_valid = int(np.argmin(np.linalg.norm(norm_pvi, axis=1)))
    i_knee = valid_reps[i_knee_valid]

    reps = {
        "knee": front[i_knee],
        "min_pvi": min(front, key=lambda p: p["pvi"]),
        "max_oil": max(front, key=lambda p: p["oil"])
    }

    # 4. Spacing y HV sobre el frente en el espacio común
    sp = spacing_to_front(
        [{"oil": p["oil"], "pvi": p["pvi"]} for p in front],
        common_normalizer,
    )
    
    hv = None
    if front:
        hv_objs = np.array([[-p["oil"], p["pvi"]] for p in front])
        hv = float(common_normalizer.hypervolume(hv_objs))

    return {
        "seed": seed,
        "front": front,
        "representative": reps["knee"],
        "representatives": reps,
        "spacing": sp,
        "hv_oil_pvi": hv,
        "weights": weights,
        "ssa_runs": ssa_runs,
        "n_evaluations_actual": total_evals,
        "n_evaluations_target": n_evaluations,
        "elapsed_s": time.time() - t0,
        "stop_reason": "evaluations_budget",
        "generations_run": None,
        "log": [],
        "common_normalizer": common_normalizer.to_dict(),
    }


def run_ssa(x0, evaluate_point, sample_neighbor,
            norm_contract, common_normalizer, seeds: list[int],
            n_evaluations: int, K: int = 11,
            T0: float = 1.0, Tf: float = 1e-3,
            contract_meta: dict | None = None) -> dict:
    """Ejecuta SSA para cada seed y agrega resultados."""
    norm_c = _to_normalizer(norm_contract)
    common_norm = _to_normalizer(common_normalizer)
    
    runs = [
        _run_one_seed(x0, evaluate_point, sample_neighbor, norm_c, common_norm,
                      n_evaluations, K, s, T0, Tf)
        for s in seeds
    ]

    def _stats(vals):
        a = np.asarray(vals, dtype=float)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "min": float(np.min(a)), "max": float(np.max(a))}

    sp_arr = np.array([r["spacing"]["spacing"] if r["spacing"]["spacing"] is not None else np.nan
                       for r in runs], dtype=float)
    hv_arr = np.array([r["hv_oil_pvi"] if r["hv_oil_pvi"] is not None else np.nan
                       for r in runs], dtype=float)
    valid_sp = sp_arr[~np.isnan(sp_arr)]
    valid_hv = hv_arr[~np.isnan(hv_arr)]

    def _rep_stats(key):
        return {
            "oil": _stats([r["representatives"][key]["oil"] for r in runs]),
            "pvi": _stats([r["representatives"][key]["pvi"] for r in runs]),
            "f2": _stats([r["representatives"][key]["f2"] for r in runs]),
        }

    summary = {
        "n_runs": len(runs),
        "oil": _stats([r["representatives"]["knee"]["oil"] for r in runs]),
        "pvi": _stats([r["representatives"]["knee"]["pvi"] for r in runs]),
        "f2": _stats([r["representatives"]["knee"]["f2"] for r in runs]),
        "hv_oil_pvi": (
            {"mean": float(valid_hv.mean()), "std": float(valid_hv.std()),
             "min": float(valid_hv.min()), "max": float(valid_hv.max())}
            if len(valid_hv) else {"mean": None, "std": None, "min": None, "max": None}
        ),
        "spacing": {
            "mean": float(valid_sp.mean()) if len(valid_sp) else None,
            "std": float(valid_sp.std()) if len(valid_sp) else None,
            "n_valid": int(len(valid_sp)),
            "n_nan": int(len(sp_arr) - len(valid_sp)),
        },
        "elapsed_s": _stats([r["elapsed_s"] for r in runs]),
        "representatives": {
            "min_pvi": _rep_stats("min_pvi"),
            "max_oil": _rep_stats("max_oil"),
            "knee":    _rep_stats("knee"),
        },
    }

    out = {
        "runs": runs,
        "summary": summary,
        "common_normalizer": common_norm.to_dict(),
        "method": "ssa",
    }
    if contract_meta:
        out["contract_meta"] = contract_meta
    return out