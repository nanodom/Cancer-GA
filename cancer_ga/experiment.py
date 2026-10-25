"""
Ejecutor de experimentos para el modelo CP de re-planificación de cosecha.

Corre CancerGA para cada combinación (tipo de contrato × semilla), con
criterio de parada por estancamiento del hipervolumen y registro por
generación, y devuelve un diccionario 100% serializable a JSON.
"""

import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from cancer_ga import CancerGA
from cancer_ga.benchmarks.harvest import (
    generate_baseline,
    generate_paper_instance,
    generate_tabali_instance,
    harvest_bounds,
    harvest_is_integer,
    make_harvest_normalizer,
    make_harvest_objective,
    make_oil_pvi_normalizer,
    plan_variability_index,
    simulate_plan,
)
from cancer_ga.utils.metrics import nondominated_mask
from cancer_ga.utils.spacing import spacing_to_front
from cancer_ga.utils.tracking import run_tracked

CONTRACT_INFO = {
    "punctual": {"label": "Puntual", "unit": "t"},
    "aggregated": {"label": "Agregado", "unit": "t"},
    "minimum_guaranteed": {"label": "Mínimo garantizado", "unit": "t"},
    "by_surface": {"label": "Por superficie", "unit": "ha"},
    "by_work_hours": {"label": "Por horas de trabajo", "unit": "h"},
}


def contract_id(contract_type, n_contracts):
    return f"{contract_type}_{n_contracts}"


def contract_label(contract_type, n_contracts):
    base = CONTRACT_INFO[contract_type]["label"]
    return base if n_contracts == 1 else f"{base} ({n_contracts} contratos)"


# --------------------------------------------------------------------------
# Construcción del problema (determinista a partir de la configuración)
# --------------------------------------------------------------------------
def build_instance(instance_cfg):
    kind = instance_cfg["kind"]
    horizon = instance_cfg.get("horizon_days", 45)
    seed = instance_cfg.get("seed", 42)
    if kind == "tabali":
        return generate_tabali_instance(instance_cfg["xlsx_path"], horizon_days=horizon, seed=seed)
    if kind == "paper":
        return generate_paper_instance(horizon_days=horizon, seed=seed)
    raise ValueError("instance.kind debe ser 'tabali' o 'paper'")


def plan_to_gantt(plan, n_resources, n_days):
    """Plan disperso {(j,t,k): t} -> matriz densa [recurso][día] de [lote, toneladas] | None."""
    gantt = [[None for _ in range(n_days)] for _ in range(n_resources)]
    for (j, t, k), amt in plan.items():
        gantt[k][t] = [int(j), round(float(amt), 3)]
    return gantt


# --------------------------------------------------------------------------
# Una corrida (contrato, semilla)
# --------------------------------------------------------------------------
def run_single(task):
    """Ejecuta una corrida y devuelve un dict serializable."""
    t_start = time.time()
    instance = build_instance(task["instance"])
    baseline_plan, _, baseline_oil, base_team, base_priority = generate_baseline(instance)

    ctype, n_c = task["contract_type"], task["n_contracts"]
    objective = make_harvest_objective(instance, baseline_plan, contract_type=ctype, n_contracts=n_c)

    # Cotas fijas del HV: independientes de la semilla de la corrida
    norm = make_harvest_normalizer(instance, objective, n_samples=task["norm_samples"])
    common_norm = make_oil_pvi_normalizer(instance, baseline_plan, n_samples=task["norm_samples"])

    ga = CancerGA(
        objective_function=objective,
        n_dim=2 * instance.n_lots,
        bounds=harvest_bounds(instance),
        n_obj=2,
        is_integer=harvest_is_integer(instance),
        seed=task["seed"],
        use_gradient=False,  # FIX 2: Apagar gradiente para igualar presupuesto
        **task["ga"],
    )
    out = run_tracked(ga, norm, **task["stop"])

    # FIX 3: Inyectar el plan base en el archivo de CancerGA
    base_x = np.concatenate([base_team, base_priority])
    base_obj, _ = objective(base_x)
    
    is_duplicate = any(np.allclose(m["obj"], base_obj, atol=1e-9) for m in out["archive"])
    if not is_duplicate:
        out["archive"].append({"x": base_x, "obj": base_obj})

    # FIX 3: Re-filtrar dominancia tras la inyección
    arch_objs = np.array([m["obj"] for m in out["archive"]])
    keep_mask = nondominated_mask(arch_objs)
    out["archive"] = [m for i, m in enumerate(out["archive"]) if keep_mask[i]]

    # Post-proceso del archivo élite: PVI y aceite
    n = instance.n_lots
    front, details = [], []
    for member in out["archive"]:
        x = member["x"]
        plan, totals, oil = simulate_plan(instance, x[:n], x[n:])
        pvi = plan_variability_index(instance, plan, baseline_plan)
        front.append({"oil": float(oil), "f2": float(member["obj"][1]), "pvi": pvi})
        details.append((x, plan, totals))

    # FIX 4: Representante (Knee) en espacio común (aceite, PVI) excluyendo plan base (pvi=0)
    valid_reps = [i for i, f in enumerate(front) if f["pvi"] > 1e-9]
    if not valid_reps:
        valid_reps = list(range(len(front)))

    objs_pvi = np.array([[-front[i]["oil"], front[i]["pvi"]] for i in valid_reps])
    norm_pvi = common_norm(objs_pvi)
    rep_i_valid = int(np.argmin(np.linalg.norm(norm_pvi, axis=1)))
    rep_i = valid_reps[rep_i_valid]

    rep_x, rep_plan, rep_totals = details[rep_i]

    # FIX 5: Spacing sobre los puntos no dominados en el espacio (aceite, PVI)
    pvi_objs = np.array([[-f["oil"], f["pvi"]] for f in front])
    nd_mask_pvi = nondominated_mask(pvi_objs)
    front_nd_pvi = [front[i] for i in range(len(front)) if nd_mask_pvi[i]]
    
    sp = spacing_to_front(
        [{"oil": f["oil"], "pvi": f["pvi"]} for f in front_nd_pvi],
        common_norm,
    )

    # HV en (aceite, PVI) con todos los puntos (incluyendo el baseline)
    proj = np.array([[-f["oil"], f["pvi"]] for f in front])
    hv_oil_pvi = float(common_norm.hypervolume(proj))

    # Representantes para el análisis estadístico
    reps = {
        "knee": front[rep_i],
        "min_pvi": min(front, key=lambda p: p["pvi"]),
        "max_oil": max(front, key=lambda p: p["oil"])
    }

    front.sort(key=lambda f: f["oil"])
    
    return {
        "contract_id": contract_id(ctype, n_c),
        "seed": task["seed"],
        "generations_run": out["generations_run"],
        "stop_reason": out["stop_reason"],
        "hv": out["best_hv"],
        "hv_oil_pvi": hv_oil_pvi,
        "spacing": sp,
        "elapsed_s": float(time.time() - t_start),
        "log": out["log"],
        "front": front,
        "representative": reps["knee"],
        "representatives": reps,
        "normalizer": norm.to_dict(),
        "common_normalizer": common_norm.to_dict(),
        "_baseline_oil": float(baseline_oil),
    }


# --------------------------------------------------------------------------
# Resumen estadístico
# --------------------------------------------------------------------------
def _stats(values):
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _stats_spacing(runs):
    """Stats de spacing entre semillas. Maneja corridas con frente <2 puntos."""
    vals = [r["spacing"]["spacing"] for r in runs if r["spacing"]["spacing"] is not None]
    n_nan = len(runs) - len(vals)
    if not vals:
        return {
            "mean": None, "std": None, "min": None, "max": None,
            "n_valid": 0, "n_nan": n_nan,
        }
    s = _stats(vals)
    s["n_valid"] = len(vals)
    s["n_nan"] = n_nan
    return s


def summarize_runs(runs):
    """Estadísticos (media, desv. estándar muestral, mín, máx) entre semillas."""
    return {
        "n_runs": len(runs),
        "oil": _stats([r["representative"]["oil"] for r in runs]),
        "f2": _stats([r["representative"]["f2"] for r in runs]),
        "pvi": _stats([r["representative"]["pvi"] for r in runs]),
        "hv": _stats([r["hv"] for r in runs]),
        "hv_oil_pvi": _stats([r["hv_oil_pvi"] for r in runs]),
        "spacing": _stats_spacing(runs),
        "generations": _stats([r["generations_run"] for r in runs]),
        "elapsed_s": _stats([r["elapsed_s"] for r in runs]),
        "stopped_by_patience": int(sum(r["stop_reason"] == "patience" for r in runs)),
    }


# --------------------------------------------------------------------------
# Experimento completo
# --------------------------------------------------------------------------
def run_experiment(
    instance_cfg,
    contract_types,
    n_contracts_list=(1,),
    seeds=(0, 1, 2, 3, 4),
    ga_params=None,
    max_generations=100,
    patience=15,
    min_delta=1e-4,
    norm_samples=200,
    workers=1,
    progress=print,
):
    """Corre todas las combinaciones (contrato × n_contratos × semilla)."""
    ga_params = dict(ga_params or {"n_cancer": 40, "n_lymph": 15})
    stop = {"max_generations": max_generations, "patience": patience, "min_delta": min_delta}

    specs = []
    for ctype in contract_types:
        for n_c in ((1,) if ctype == "punctual" else n_contracts_list):
            specs.append((ctype, n_c))

    tasks = [
        {
            "instance": instance_cfg,
            "contract_type": ctype,
            "n_contracts": n_c,
            "seed": seed,
            "ga": ga_params,
            "stop": stop,
            "norm_samples": norm_samples,
        }
        for ctype, n_c in specs
        for seed in seeds
    ]

    t0 = time.time()
    results = []
    total = len(tasks)

    def _report(done, res):
        if progress:
            sp = res["spacing"]["spacing"]
            sp_txt = f"{sp:.4f}" if sp is not None else "  --  "
            progress(
                f"[{done:3d}/{total}] {res['contract_id']:<24} semilla {res['seed']:>3} | "
                f"gens {res['generations_run']:3d} ({res['stop_reason']:<15}) | "
                f"HV {res['hv']:.4f} | SP {sp_txt} | {res['elapsed_s']:5.1f}s"
            )

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            res = run_single(task)
            results.append(res)
            _report(i, res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                _report(i, res)

    instance = build_instance(instance_cfg)
    baseline_oil = results[0]["_baseline_oil"]

    contracts = []
    for ctype, n_c in specs:
        cid = contract_id(ctype, n_c)
        runs = sorted((r for r in results if r["contract_id"] == cid), key=lambda r: r["seed"])
        contracts.append(
            {
                "id": cid,
                "type": ctype,
                "n_contracts": n_c,
                "label": contract_label(ctype, n_c),
                "unit": CONTRACT_INFO[ctype]["unit"],
                "normalizer": runs[0]["normalizer"],
                "common_normalizer": runs[0]["common_normalizer"],
                "runs": runs,
                "summary": summarize_runs(runs),
            }
        )
    for r in results:
        r.pop("_baseline_oil", None)

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instance": instance_cfg,
            "instance_label": {"tabali": "Fundo Tabalí (datos reales)", "paper": "Instancia sintética del paper"}[
                instance_cfg["kind"]
            ],
            "n_lots": instance.n_lots,
            "n_days": instance.horizon_days,
            "n_resources": instance.n_resources,
            "resource_labels": [r["id"] for r in instance.resources],
            "baseline_oil": float(baseline_oil),
            "seeds": list(seeds),
            "ga_params": ga_params,
            "stop": stop,
            "norm_samples": norm_samples,
            "total_runtime_s": float(time.time() - t0),
            "workers": workers,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "contracts": contracts,
    }