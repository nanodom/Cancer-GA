"""
Experimento completo: CancerGA sobre el modelo CP de re-planificación de
cosecha, con múltiples semillas, todos los tipos de contrato, criterio de
parada por hipervolumen, logging por generación y dashboard HTML.

Ejemplos
--------
    # Experimento por defecto (datos reales de Tabalí, 5 contratos x 5 semillas)
    python examples/run_harvest_experiment.py

    # Prueba rápida
    python examples/run_harvest_experiment.py --seeds 3 --max-generations 20 --patience 6 --n-cancer 20 --n-lymph 8

    # Sólo algunos contratos, con 1, 2 y 3 contratos agrupados (Sección 5.3 del paper)
    python examples/run_harvest_experiment.py --contracts aggregated by_work_hours --n-contracts 1 2 3

    # Reconstruir el dashboard desde resultados ya guardados (sin recalcular)
    python examples/run_harvest_experiment.py --from-json results/results.json

Salida (en --out-dir, por defecto ./results):
    results.json     configuración, versiones, log por generación y frentes de cada corrida
    dashboard.html   dashboard autocontenido (funciona sin internet)
"""

import argparse
import json
import os
from pathlib import Path

from cancer_ga.benchmarks.harvest import CONTRACT_TYPES
from cancer_ga.experiment import run_experiment
from cancer_ga.utils.dashboard import build_dashboard_html

DEFAULT_XLSX = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--instance",
        choices=["tabali", "paper"],
        default="tabali",
        help="tabali = datos reales (28 cuarteles); paper = instancia sintética de 63 lotes",
    )
    p.add_argument("--xlsx", default=str(DEFAULT_XLSX), help="ruta a datos_Tabali.xlsx")
    p.add_argument("--horizon-days", type=int, default=45)
    p.add_argument("--contracts", nargs="+", choices=CONTRACT_TYPES, default=list(CONTRACT_TYPES))
    p.add_argument(
        "--n-contracts",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1],
        help="agrupación de recursos en 1, 2 o 3 contratos (no aplica a 'punctual')",
    )
    p.add_argument("--seeds", type=int, default=5, help="número de semillas (0..N-1)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-generations", type=int, default=100)
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="N_max de generaciones sin mejora del hipervolumen para detenerse (0 = desactivar)",
    )
    p.add_argument(
        "--min-delta", type=float, default=1e-4, help="mejora mínima de HV normalizado que cuenta como mejora"
    )
    p.add_argument("--n-cancer", type=int, default=40)
    p.add_argument("--n-lymph", type=int, default=15)
    p.add_argument("--norm-samples", type=int, default=200, help="soluciones aleatorias para fijar el nadir del HV")
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="procesos en paralelo (1 = secuencial)"
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--from-json", default=None, help="reconstruir el dashboard desde un results.json existente")
    return p.parse_args()


def _validate_args(args):
    """Validaciones de combinaciones de argumentos. Falla temprano, antes de
    lanzar cualquier corrida costosa."""

    # 1) Contratos no implementados (defensivo: argparse ya filtra con choices,
    #    pero esto protege si CONTRACT_TYPES cambia en el futuro).
    missing = set(args.contracts) - set(CONTRACT_TYPES)
    if missing:
        raise SystemExit(
            f"Contratos no implementados: {sorted(missing)}. "
            f"Disponibles: {list(CONTRACT_TYPES)}"
        )

    # 2) 'punctual' no depende de --n-contracts. Aviso explícito para que no
    #    haya resultados silenciosamente ambiguos.
    if "punctual" in args.contracts and any(n > 1 for n in args.n_contracts):
        print(
            f"Aviso: 'punctual' no depende de --n-contracts "
            f"(recibido {args.n_contracts}); se ignorará para ese contrato."
        )

    # 3) El paper (Sección 5.3) compara n_contracts ∈ {1, 2, 3} para los
    #    contratos agrupables. Si no se pasa ese conjunto, advertir.
    groupable = [c for c in args.contracts if c != "punctual"]
    if groupable and set(args.n_contracts) != {1, 2, 3}:
        print(
            f"Aviso: el paper compara n_contracts ∈ {{1,2,3}}; "
            f"recibido {args.n_contracts}. Para reproducir la Tabla 2 usa "
            f"--n-contracts 1 2 3."
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
        instance_cfg = {"kind": args.instance, "horizon_days": args.horizon_days, "seed": 42}
        if args.instance == "tabali":
            instance_cfg["xlsx_path"] = str(Path(args.xlsx).resolve())

        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        print(
            f"Instancia: {args.instance} | contratos: {args.contracts} | n_contratos: {args.n_contracts} | "
            f"semillas: {seeds} | workers: {args.workers}\n"
        )

        results = run_experiment(
            instance_cfg,
            contract_types=args.contracts,
            n_contracts_list=args.n_contracts,
            seeds=seeds,
            ga_params={"n_cancer": args.n_cancer, "n_lymph": args.n_lymph},
            max_generations=args.max_generations,
            patience=args.patience if args.patience > 0 else None,
            min_delta=args.min_delta,
            norm_samples=args.norm_samples,
            workers=args.workers,
        )
        json_path = out_dir / "results.json"
        json_path.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
        print(f"\nResultados guardados en: {json_path}")

    html_path = build_dashboard_html(results, out_dir / "dashboard.html")
    print(f"Dashboard generado en:   {html_path}")

    print("\nResumen (media ± desv. entre semillas; f2 NO se compara entre contratos):")
    print(f"{'Contrato':<30}{'Aceite':>18}{'PVI':>16}{'HV(aceite,PVI)':>18}")
    for c in results["contracts"]:
        s = c["summary"]
        print(
            f"{c['label']:<30}{s['oil']['mean']:>12,.0f} ±{s['oil']['std']:>5,.0f}"
            f"{s['pvi']['mean']:>10.1f} ±{s['pvi']['std']:>4.1f}"
            f"{s['hv_oil_pvi']['mean']:>12.3f} ±{s['hv_oil_pvi']['std']:>5.3f}"
        )


if __name__ == "__main__":
    main()