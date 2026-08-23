import numpy as np
from cancer_ga import CancerGA
from cancer_ga.benchmarks.single_objective import rastrigin, constrained_mixed_example
from cancer_ga.utils.constraints import linear_leq_constraints, linear_eq_constraints

if __name__ == "__main__":
    print("Ejecutando CancerGA en Rastrigin...")
    ga = CancerGA(
        objective_function=rastrigin,
        n_dim=10,
        bounds=(-5.12, 5.12),
        n_cancer=80,
        n_lymph=25,
        seed=42
    )
    result = ga.run(n_generations=100, verbose=True)
    if result:
        best_x, best_f, feasible = result
        print(f"\nMejor x: {best_x}")
        print(f"Mejor f(x): {best_f:.6f}")