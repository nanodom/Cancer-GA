"""
Experimento completo: CancerGA + baseline SSA sobre el modelo CP de
re-planificación de cosecha, con múltiples semillas, todos los tipos de
contrato, criterio de parada por hipervolumen, logging por generación,
métricas de spacing-to-front y dashboard HTML.

Ejemplos
--------
    # Comparación por defecto (CancerGA + SSA, Tabalí, 5 contratos × 5 semillas)
    python examples/run_harvest_experiment.py

    # Solo CancerGA (comportamiento previo)
    python examples/run_harvest_experiment.py --baseline cancer

    # Solo SSA (scaffolding mínimo de CancerGA para la estructura)
    python examples/run_harvest_experiment.py --baseline ssa

    # Prueba rápida
    python examples/run_harvest_experiment.py \
        --seeds 2 --max-generations 20 --patience 6 \
        --n-cancer 20 --n-lymph 8 --n-evaluations 2000

    # Reproducir la Tabla 2 del paper (contratos agrupables × 1, 2, 3)
    python examples/run_harvest_experiment.py \
        --contracts aggregated minimum_guaranteed by_surface by_work_hours \
        --n-contracts 1 2 3

    # Reconstruir el dashboard desde un results.json existente
    python examples/run_harvest_experiment.py --from-json results/results.json

Salida (en --out-dir, por defecto ./results):
    results.json     configuración, versiones, log por generación y frentes de cada corrida
    dashboard.html   dashboard autocontenido (funciona sin internet)
"""

import argparse
import json
import os
from pathlib import Path

from cancer_ga.benchmarks.harvest import (
    CONTRACT_TYPES,
    generate_baseline,
    generate_paper_instance,
    generate_tabali_instance,
)
from cancer_ga.experiment import run_experiment
from cancer_ga.utils.dashboard import build_dashboard_html


DEFAULT_XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Instancia y contratos ---
    p.add_argument(
        "--instance", choices=["tabali", "paper"], default="tabali",
        help="tabali = datos reales (28 cuarteles); paper = instancia sintética de 63 lotes",
    )
    p.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="ruta a datos_Tabali.xlsx")
    p.add_argument("--horizon-days", type=int, default=45)
    p.add_argument("--contracts", nargs="+", choices=CONTRACT_TYPES,
                   default=list(CONTRACT_TYPES))
    p.add_argument(
        "--n-contracts", nargs="+", type=int, choices=[1, 2, 3], default=[1],
        help="agrupación de recursos en 1, 2 o 3 contratos (no aplica a 'punctual')",
    )

    # --- Presupuesto CancerGA ---
    p.add_argument("--seeds", type=int, default=5, help="número de semillas (0..N-1)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-generations", type=int, default=100)
    p.add_argument(
        "--patience", type=int, default=15,
        help="N_max de generaciones sin mejora del HV para detenerse (0 = desactivar)",
    )
    p.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="mejora mínima de HV normalizado que cuenta como mejora",
    )
    p.add_argument("--n-cancer", type=int, default=40)
    p.add_argument("--n-lymph", type=int, default=15)
    p.add_argument(
        "--norm-samples", type=int, default=200,
        help="soluciones aleatorias para fijar el nadir del HV",
    )

    # --- Baseline SSA ---
    p.add_argument(
        "--baseline", choices=["cancer", "ssa", "both"], default="both",
        help="cancer = solo CancerGA; ssa = solo SSA (scaffolding mínimo); both = comparación",
    )
    p.add_argument(
        "--n-evaluations", type=int, default=20000,
        help="presupuesto total de evaluaciones para el SSA (repartido entre K pesos)",
    )
    p.add_argument(
        "--ssa-weights", type=int, default=11,
        help="número K de pesos (w1, w2) para la escalarización del SSA",
    )
    p.add_argument(
        "--ssa-neighbor-scale", type=float, default=0.15,
        help="desviación típica del vecindario gaussiano del SSA",
    )

    # --- Infraestructura ---
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="procesos en paralelo para CancerGA (1 = secuencial)",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--from-json", default=None,
        help="reconstruir el dashboard desde un results.json existente",
    )
    return p.parse_args()


def _validate_args(args):
    """Validaciones tempranas de combinaciones de argumentos."""

    missing = set(args.contracts) - set(CONTRACT_TYPES)
    if missing:
        raise SystemExit(
            f"Contratos no implementados: {sorted(missing)}. "
            f"Disponibles: {list(CONTRACT_TYPES)}"
        )

    if "punctual" in args.contracts and any(n > 1 for n in args.n_contracts):
        print(
            f"Aviso: 'punctual' no depende de --n-contracts "
            f"(recibido {args.n_contracts}); se ignorará para ese contrato."
        )

    groupable = [c for c in args.contracts if c != "punctual"]
    if groupable and set(args.n_contracts) != {1, 2, 3}:
        print(
            f"Aviso: el paper compara n_contracts ∈ {{1,2,3}}; "
            f"recibido {args.n_contracts}. Para reproducir la Tabla 2 usa "
            f"--n-contracts 1 2 3."
        )

    if args.baseline in ("ssa", "both"):
        if args.ssa_weights < 2:
            raise SystemExit("--ssa-weights debe ser >= 2")
        if args.n_evaluations < args.ssa_weights:
            raise SystemExit(
                f"--n-evaluations ({args.n_evaluations}) debe ser >= "
                f"--ssa-weights ({args.ssa_weights})"
            )


def _build_instance(args):
    """Reconstruye la instancia con los mismos defaults que `run_experiment`."""
    if args.instance == "tabali":
        return generate_tabali_instance(args.xlsx, horizon_days=args.horizon_days)
    return generate_paper_instance(horizon_days=args.horizon_days)


def _attach_ssa_baseline(args, results, seeds):
    """
    Corre SSA por contrato y lo adjunta como `baseline_ssa`.

    - Reutiliza el `common_normalizer` del primer contrato (mismo que CancerGA).
    - Reconstruye `instance`/`plan` con los mismos defaults.
    - Se salta `punctual` con `n_contracts != 1` (no aplica).
    """
    from cancer_ga.baseline import run_ssa
    from cancer_ga.baseline.harvest_adapter import make_harvest_callbacks

    instance = _build_instance(args)
    plan, totals, _, _, _ = generate_baseline(instance)

    if not results["contracts"]:
        raise SystemExit(
            "No hay contratos en `results` para tomar el common_normalizer. "
            "Corre primero con --baseline cancer o both."
        )
    common_normalizer = results["contracts"][0]["common_normalizer"]

    index = {(c["type"], c["n_contracts"]): c for c in results["contracts"]}

    for contract_type in args.contracts:
        for n in args.n_contracts:
            if contract_type == "punctual" and n != 1:
                continue
            key = (contract_type, n)
            if key not in index:
                continue
            contract = index[key]
            print(f"\n=== SSA baseline — {contract['label']} ===")
            x0, ev, sn = make_harvest_callbacks(
                instance, plan, totals, contract_type,
                n_contracts=n,
                neighbor_scale=args.ssa_neighbor_scale,
            )
            ssa_result = run_ssa(
                x0=x0,
                evaluate_point=ev,
                sample_neighbor=sn,
                norm_contract=contract["normalizer"],
                common_normalizer=contract["common_normalizer"],
                seeds=seeds,
                n_evaluations=args.n_evaluations,
                K=args.ssa_weights,
                contract_meta={"type": contract_type, "n_contracts": n},
            )
            contract["baseline_ssa"] = ssa_result


def _fmt(v):
    return "nan" if v is None else f"{v:.3f}"


def _print_summary(results, skip_cancer=False):
    print("\nResumen (media ± desv. entre semillas; f2 NO se compara entre contratos):")
    header = (
        f"{'Contrato':<28}{'Método':<10}"
        f"{'Aceite':>14}{'PVI':>8}{'HV':>9}{'Spacing':>10}{'Tiempo(s)':>11}"
    )
    print(header)
    print("-" * len(header))

    for c in results["contracts"]:
        if not skip_cancer:
            s = c["summary"]
            t = s.get("elapsed_s", {}).get("mean")
            print(
                f"{c['label']:<28}{'CancerGA':<10}"
                f"{s['oil']['mean']:>14,.0f}"
                f"{s['pvi']['mean']:>8.1f}"
                f"{s['hv_oil_pvi']['mean']:>9.3f}"
                f"{_fmt(s.get('spacing', {}).get('mean')):>10}"
                f"{_fmt(t):>11}"
            )
        if "baseline_ssa" in c:
            bs = c["baseline_ssa"]["summary"]
            print(
                f"{c['label']:<28}{'SSA':<10}"
                f"{bs['oil']['mean']:>14,.0f}"
                f"{bs['pvi']['mean']:>8.1f}"
                f"{_fmt(bs['hv_oil_pvi']['mean']):>9}"
                f"{_fmt(bs['spacing']['mean']):>10}"
                f"{_fmt(bs['elapsed_s']['mean']):>11}"
            )


def main():
    args = parse_args()
    _validate_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        results = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print(f"Resultados cargados desde {args.from_json}")
    else:
        instance_cfg = {
            "kind": args.instance,
            "horizon_days": args.horizon_days,
            "seed": 42,
        }
        if args.instance == "tabali":
            instance_cfg["xlsx_path"] = str(Path(args.xlsx).resolve())

        seeds = list(range(args.seed_start, args.seed_start + args.seeds))

        # --baseline ssa: scaffolding mínimo de CancerGA (1 semilla, 1 generación)
        # solo para construir `results` y el `common_normalizer`. Los números
        # de CancerGA en este modo NO son representativos.
        if args.baseline == "ssa":
            print(
                "Modo --baseline ssa: CancerGA se ejecuta como scaffolding "
                "(seeds=[0], max_generations=1) para construir la estructura "
                "y el common_normalizer. Los números de CancerGA no son "
                "válidos para el paper en este modo."
            )
            cancer_seeds = [args.seed_start]
            cancer_max_gen = 1
            cancer_patience = None
        else:
            cancer_seeds = seeds
            cancer_max_gen = args.max_generations
            cancer_patience = args.patience if args.patience > 0 else None

        print(
            f"Instancia: {args.instance} | contratos: {args.contracts} | "
            f"n_contratos: {args.n_contracts} | semillas: {seeds} | "
            f"baseline: {args.baseline} | workers: {args.workers}\n"
        )

        results = run_experiment(
            instance_cfg,
            contract_types=args.contracts,
            n_contracts_list=args.n_contracts,
            seeds=cancer_seeds,
            ga_params={"n_cancer": args.n_cancer, "n_lymph": args.n_lymph},
            max_generations=cancer_max_gen,
            patience=cancer_patience,
            min_delta=args.min_delta,
            norm_samples=args.norm_samples,
            workers=args.workers,
        )

        if args.baseline in ("ssa", "both"):
            _attach_ssa_baseline(args, results, seeds)

        json_path = out_dir / "results.json"
        json_path.write_text(
            json.dumps(results, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResultados guardados en: {json_path}")

    html_path = build_dashboard_html(results, out_dir / "dashboard.html")
    print(f"Dashboard generado en:   {html_path}")

    _print_summary(results, skip_cancer=(args.baseline == "ssa"))


if __name__ == "__main__":
    main()