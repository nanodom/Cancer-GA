import numpy as np
from cancer_ga import CancerGA
from cancer_ga.benchmarks.single_objective import sphere
from cancer_ga.operators.mutation import adaptive_mutation
from cancer_ga.operators.crossover import crossover
from cancer_ga.operators.starvation import starvation

def test_adaptive_mutation_bounds():
    ga = CancerGA(objective_function=sphere, n_dim=3, bounds=(-5.0, 5.0), seed=42)
    pop = ga._init_population(5)
    
    adaptive_mutation(pop, ga)
    
    for ind in pop:
        # Verificar que la mutación respete los límites definidos
        assert np.all(ind.x >= -5.0) and np.all(ind.x <= 5.0)
        assert ind.sigma >= ga.sigma_min

def test_crossover_generation():
    ga = CancerGA(objective_function=sphere, n_dim=2, bounds=(-5.0, 5.0), crossover_rate=1.0, seed=42)
    pop = ga._init_population(4)
    ga.evaluate(pop)
    
    new_pop = crossover(pop, ga)
    assert len(new_pop) >= len(pop)

def test_starvation_removal():
    ga = CancerGA(objective_function=sphere, n_dim=2, bounds=(-5.0, 5.0), seed=42)
    pop = ga._init_population(3)
    
    # Agotar energía intencionalmente a un individuo
    pop[0].energy = -1.0
    pop[1].energy = 1.0
    pop[2].energy = 1.0
    
    ga.evaluate(pop)
    survivors = starvation(pop, ga)
    
    assert len(survivors) == 2
    assert all(ind.energy > 0 for ind in survivors)