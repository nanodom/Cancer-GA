import numpy as np
import pytest

from cancer_ga import CancerGA


def _bi(x):
    return np.array([np.sum(x**2), np.sum((x - 1) ** 2)])


@pytest.mark.parametrize("seed", range(8))
def test_biobjective_low_dimension_does_not_crash(seed):
    """Regresión: los hijos del cruce nacían con fitness=None y niche_proliferation
    los comparaba con float cuando un nicho mezclaba padres e hijos (TypeError).
    Ocurría en problemas bi-objetivo de baja dimensión, donde los nichos se agrupan."""
    ga = CancerGA(objective_function=_bi, n_dim=2, bounds=(-5.0, 5.0), n_obj=2, n_cancer=15, n_lymph=5, seed=seed)
    result = ga.run(n_generations=15, verbose=False)
    assert result is not None
