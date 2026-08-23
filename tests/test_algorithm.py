import numpy as np
from cancer_ga import CancerGA
from cancer_ga.benchmarks.single_objective import sphere, rastrigin
from cancer_ga.benchmarks.ctp import CTP1

def test_single_objective_execution():
    ga = CancerGA(
        objective_function=sphere,
        n_dim=4,
        bounds=(-5.12, 5.12),
        n_cancer=20,
        n_lymph=10,
        seed=42
    )
    result = ga.run(n_generations=10, verbose=False)
    
    assert result is not None
    best_x, best_f, feasible = result
    assert len(best_x) == 4
    assert isinstance(best_f, (float, np.floating))
    assert feasible is True

def test_integer_variables_repair():
    # x0 continua, x1 entera
    is_int = [False, True]
    ga = CancerGA(
        objective_function=rastrigin,
        n_dim=2,
        bounds=(-5.0, 5.0),
        is_integer=is_int,
        n_cancer=10,
        n_lymph=5,
        seed=42
    )
    best_x, _, _ = ga.run(n_generations=5, verbose=False)
    
    # La variable entera debe ser exactamente un número entero
    assert best_x[1] == np.round(best_x[1])

def test_multiobjective_ctp1_execution():
    def ctp1_wrapper(x):
        objs, g = CTP1(x)
        return objs

    ga = CancerGA(
        objective_function=ctp1_wrapper,
        n_dim=2,
        bounds=(0.0, 1.0),
        n_obj=2,
        n_cancer=15,
        n_lymph=5,
        seed=42
    )
    result = ga.run(n_generations=5, verbose=False)
    
    assert result is not None
    best_x, best_objs, _ = result
    assert len(best_objs) == 2