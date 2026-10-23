import numpy as np


def _scalar_guidance(x, weights, ga_instance):
    """Convierte el estado del individuo en un escalar 'mayor=mejor' para calcular el gradiente."""
    cv = ga_instance._constraint_violation(x)
    if cv > ga_instance.feas_eps:
        return -cv
    obj = ga_instance._evaluate_objectives(x)
    return -np.dot(weights, obj)


def _finite_diff_gradient(x, weights, ga_instance):
    eps = ga_instance.chemo_fd_eps
    grad = np.zeros(ga_instance.n_dim)
    f0 = _scalar_guidance(x, weights, ga_instance)
    for i in range(ga_instance.n_dim):
        if ga_instance.is_integer[i]:
            continue
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_scalar_guidance(xp, weights, ga_instance) - f0) / eps
    norm = np.linalg.norm(grad)
    return grad / norm if norm > 1e-12 else grad


def chemotaxis(pop, ga_instance):
    """Ejecuta el movimiento guiado por gradiente combinado con componente aleatoria.

    Si ``ga_instance.use_gradient`` es False (p. ej. problemas con decodificadores constantes a trozos,
    como la selección de flota), no se calcula el gradiente y el movimiento es puramente aleatorio.
    """
    use_gradient = getattr(ga_instance, "use_gradient", True)
    for ind in pop:
        random_dir = ga_instance.rng.normal(size=ga_instance.n_dim)
        rn = np.linalg.norm(random_dir)
        random_dir = random_dir / rn if rn > 1e-12 else random_dir
        if use_gradient:
            grad_dir = _finite_diff_gradient(ind.x, ind.weights, ga_instance)
        else:
            grad_dir = np.zeros(ga_instance.n_dim)
        step = ga_instance.chemo_step * ind.sigma / ga_instance.sigma_init

        # ECM remodelada por fibroblastos: más peso al movimiento guiado
        phi = ga_instance._phi_of(ind)
        beta = min(1.0, ga_instance.chemo_beta + ga_instance.fibro_beta_boost * phi)
        if not use_gradient:
            beta = 0.0
        move = (1 - beta) * random_dir + beta * grad_dir
        ind.x = ga_instance._repair(ind.x + step * move)
