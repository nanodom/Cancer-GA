import numpy as np

def sphere(x):
    """Función Sphere continua multivariada."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

def rastrigin(x):
    """Función Rastrigin altamente multimodal."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def rosenbrock(x):
    """Función Valle de Rosenbrock."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

def ackley(x):
    """Función Ackley."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e)

def constrained_mixed_example(x):
    """Ejemplo continuo-mixto con restricciones para pruebas generales."""
    x = np.asarray(x, dtype=float)
    return float(x[0]**2 + x[1]**2 + 3 * x[2])