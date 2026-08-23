import numpy as np

def linear_leq_constraints(A, b):
    """Construye restricciones A·x <= b como lista de g_i(x) <= 0."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return [(lambda x, a=A[i], bi=b[i]: float(a @ x - bi)) for i in range(A.shape[0])]

def linear_eq_constraints(A, b):
    """Construye restricciones A·x = b como lista de |h_i(x)| <= tol."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return [(lambda x, a=A[i], bi=b[i]: float(a @ x - bi)) for i in range(A.shape[0])]