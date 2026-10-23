"""
Ejemplo: aplicar CancerGA al modelo bi-objetivo CP de re-planificación de
cosecha, calibrado con los datos reales de Fundo Tabalí (Coquimbo): 28
cuarteles reales, toneladas reales por cuartel y tasa de extracción de
aceite real (~18.1%).

Uso:
    python examples/run_tabali_cp.py
"""
import time
from pathlib import Path

from cancer_ga import CancerGA
from cancer_ga.benchmarks.harvest import (
    generate_tabali_instance,
    generate_baseline,
    make_harvest_objective,
    harvest_bounds,
    harvest_is_integer,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "datos_Tabali.xlsx"


def main():
    print(f"Leyendo datos reales desde: {DATA_PATH}")
    instance = generate_tabali_instance(DATA_PATH)
    print(f"Instancia calibrada: {instance.n_lots} cuarteles reales, "
          f"{instance.n_resources} recursos, horizonte de {instance.horizon_days} días")
    print(f"Tasa de extracción de aceite calibrada: {instance.extraction_ratio:.4f} L/kg "
          f"({instance.extraction_ratio * 100:.1f}%)")
    print(f"Toneladas totales de aceituna (reales): {instance.tons.sum():.1f} t\n")

    plan, totals, baseline_oil, team_gene, priority = generate_baseline(instance)
    print(f"Plan base (BP): producción de aceite = {baseline_oil:.1f} L "
          f"(equivalente a f1 sin reprogramar)\n")

    obj = make_harvest_objective(instance, plan, totals, contract_type="aggregated")
    ga = CancerGA(
        objective_function=obj,
        n_dim=2 * instance.n_lots,
        bounds=harvest_bounds(instance),
        n_obj=2,
        is_integer=harvest_is_integer(instance),
        n_cancer=40,
        n_lymph=15,
        seed=42,
    )

    t0 = time.time()
    best_x, best_objs, feasible = ga.run(n_generations=30, verbose=True)
    elapsed = time.time() - t0

    oil = -best_objs[0]
    print(f"\nMejor solución encontrada en {elapsed:.1f}s:")
    print(f"  Producción de aceite: {oil:.1f} L (baseline: {baseline_oil:.1f} L, "
          f"delta: {oil - baseline_oil:+.1f} L)")
    print(f"  Discrepancia f2 (agregada): {best_objs[1]:.1f}")
    print(f"  Factible: {feasible}")


if __name__ == "__main__":
    main()
