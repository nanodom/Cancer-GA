import numpy as np
from cancer_ga.individual import Individual

def test_individual_initialization():
    x = np.array([1.0, 2.0])
    ind = Individual(x=x, sigma=0.5, energy=1.0)
    
    assert np.array_equal(ind.x, x)
    assert ind.sigma == 0.5
    assert ind.energy == 1.0
    assert ind.cv == 0.0
    assert ind.niche_id == -1