import math
import numpy as np
from .proliferation import assign_niches

def _levy_flight(n_dim, beta, scale, rng):
    sigma_u = (
        math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=n_dim)
    v = rng.normal(0, 1, size=n_dim)
    step = u / (np.abs(v) ** (1 / beta))
    return scale * step

def metastasis(pop, ga_instance):
    """Mueve aleatoriamente algunos individuos a nichos distantes usando saltos Lévy."""
    assign_niches(pop, ga_instance.niche_radius)
    niche_sizes = {}
    for ind in pop:
        niche_sizes[ind.niche_id] = niche_sizes.get(ind.niche_id, 0) + 1

    for ind in pop:
        crowding = niche_sizes[ind.niche_id] / ga_instance.niche_capacity
        p_meta = ga_instance.p_meta0 * min(1.0, crowding)
        if ga_instance.rng.random() < p_meta:
            ind.x = ga_instance._repair(ind.x + _levy_flight(ga_instance.n_dim, ga_instance.levy_beta, ga_instance.levy_scale, ga_instance.rng))
            ind.sigma = ga_instance.sigma_init
            if ga_instance.n_obj is not None and ga_instance.n_obj > 1:
                ind.weights = ga_instance.rng.dirichlet(np.ones(ga_instance.n_obj))