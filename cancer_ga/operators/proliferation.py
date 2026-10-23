import numpy as np
from cancer_ga.individual import Individual

def assign_niches(pop, niche_radius):
    """Clustering greedy single-linkage por distancia euclidiana."""
    niche_id = 0
    for ind in pop:
        ind.niche_id = -1
    for i, ind in enumerate(pop):
        if ind.niche_id != -1:
            continue
        ind.niche_id = niche_id
        for other in pop[i + 1:]:
            if other.niche_id == -1 and np.linalg.norm(ind.x - other.x) < niche_radius:
                other.niche_id = niche_id
        niche_id += 1
    return niche_id

def compute_fibroblast_field(pop, ga_instance):
    """Calcula el nivel de activación phi de fibroblastos por nicho."""
    if not ga_instance.use_fibroblasts or not pop:
        return {}

    n_niches = assign_niches(pop, ga_instance.niche_radius)
    niche_quality = np.zeros(n_niches)
    niche_count = np.zeros(n_niches)
    
    for ind in pop:
        niche_quality[ind.niche_id] += ind.fitness
        niche_count[ind.niche_id] += 1
    niche_quality /= np.maximum(niche_count, 1)

    global_mean = niche_quality.mean()
    phi = 1.0 / (1.0 + np.exp(-ga_instance.fibro_k * (niche_quality - global_mean)))
    return {nid: float(phi[nid]) for nid in range(n_niches)}

def niche_proliferation(pop, max_pop, ga_instance):
    """Genera descendientes según crecimiento logístico por nicho."""
    # Los hijos del cruce nacen con fitness=None; evaluar antes de ordenar por fitness
    if any(ind.fitness is None for ind in pop):
        ga_instance.evaluate(pop)

    n_niches = assign_niches(pop, ga_instance.niche_radius)
    offspring = []

    for nid in range(n_niches):
        members = [ind for ind in pop if ind.niche_id == nid]
        n_local = len(members)
        expected_growth = ga_instance.growth_rate * n_local * (1 - n_local / ga_instance.niche_capacity)
        n_new = max(0, int(round(expected_growth)))
        members_sorted = sorted(members, key=lambda i: i.fitness, reverse=True)

        for k in range(n_new):
            parent = members_sorted[k % len(members_sorted)]
            child_x = ga_instance._repair(parent.x + ga_instance.rng.normal(scale=parent.sigma * 0.5, size=ga_instance.n_dim))
            offspring.append(Individual(child_x, parent.sigma, ga_instance.e_init * 0.7, weights=parent.weights))

    new_pop = pop + offspring
    if len(new_pop) > max_pop:
        ga_instance.evaluate(new_pop)
        new_pop.sort(key=lambda i: i.fitness, reverse=True)
        new_pop = new_pop[:max_pop]

    return new_pop