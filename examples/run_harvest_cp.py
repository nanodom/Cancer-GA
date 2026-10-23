"""
Ejemplo: aplicar CancerGA al modelo bi-objetivo CP de re-planificación de
cosecha (paper de Urrejola-Arrellano & Montenegro-Dos Santo), a la escala de
la Sección 5.1 (63 lotes, 45 días, 11 recursos), comparando contrato puntual
vs. agregado tal como en la Tabla 2 del paper.

Uso:
    python examples/run_harvest_cp.py
"""
import time
import numpy as np

from cancer_ga import CancerGA
from cancer_ga.benchmarks.harvest import (
    generate_paper_instance,
    generate_baseline,
    make_harvest_objective,
    harvest_bounds,
    harvest_is_integer,
)


def run_contract(instance, baseline_plan, baseline_totals, contract_type, seed=42,
                  n_cancer=40, n_lymph=15, n_generations=25, disabled_days=None):
    obj = make_harvest_objective(
        instance, baseline_plan, baseline_totals,
        contract_type=contract_type, disabled_days=disabled_days,
    )
    ga = CancerGA(
        objective_function=obj,
        n_dim=2 * instance.n_lots,
        bounds=harvest_bounds(instance),
        n_obj=2,
        is_integer=harvest_is_integer(instance),
        n_cancer=n_cancer,
        n_lymph=n_lymph,
        seed=seed,
    )
    t0 = time.time()
    best_x, best_objs, feasible = ga.run(n_generations=n_generations, verbose=True)
    elapsed = time.time() - t0
    oil = -best_objs[0]
    return {
        "x": best_x, "oil": oil, "f2": best_objs[1],
        "feasible": feasible, "elapsed": elapsed,
    }


def main():
    print("Generando instancia a escala del paper (63 lotes, 45 dias, 11 recursos)...")
    instance = generate_paper_instance()
    plan, totals, baseline_oil, team_gene, priority = generate_baseline(instance)
    print(f"Plan base (BP): produccion de aceite = {baseline_oil:.2f}  "
          f"(equivalente a f1 sin reprogramar)\n")

    results = {}
    for contract_type in ("punctual", "aggregated"):
        print(f"=== Contrato: {contract_type} ===")
        results[contract_type] = run_contract(instance, plan, totals, contract_type)
        r = results[contract_type]
        print(f"  Mejor solucion -> oleico producido: {r['oil']:.2f}  "
              f"(baseline: {baseline_oil:.2f}, delta: {r['oil'] - baseline_oil:+.2f})")
        print(f"  Discrepancia f2 ({contract_type}): {r['f2']:.2f}")
        print(f"  Factible: {r['feasible']}  |  Tiempo: {r['elapsed']:.1f}s\n")

    print("Resumen (analogo a la Tabla 2 del paper):")
    print(f"{'Contrato':<12}{'f1 (oleico)':>14}{'f2 (discrepancia)':>20}")
    for contract_type, r in results.items():
        print(f"{contract_type:<12}{r['oil']:>14.2f}{r['f2']:>20.2f}")


if __name__ == "__main__":
    main()
