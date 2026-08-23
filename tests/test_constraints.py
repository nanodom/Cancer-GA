import numpy as np
from cancer_ga.utils.constraints import linear_leq_constraints, linear_eq_constraints

def test_linear_leq_constraints():
    # A*x <= b -> [1, 1]*[x0, x1] <= 2
    A = [[1.0, 1.0]]
    b = [2.0]
    funcs = linear_leq_constraints(A, b)
    
    assert len(funcs) == 1
    assert funcs[0](np.array([1.0, 0.0])) == -1.0  # Factible (<0)
    assert funcs[0](np.array([2.0, 1.0])) == 1.0   # Infactible (>0)

def test_linear_eq_constraints():
    # A*x = b -> [1, -1]*[x0, x1] = 0
    A = [[1.0, -1.0]]
    b = [0.0]
    funcs = linear_eq_constraints(A, b)
    
    assert funcs[0](np.array([2.0, 2.0])) == 0.0
    assert funcs[0](np.array([3.0, 1.0])) == 2.0