import numpy as np

def _normalize_fitness(pop):
    """Escala el fitness de una población al rango [0, 1]."""
    if not pop:
        return {}
    vals = np.array([ind.fitness for ind in pop])
    f_min, f_max = vals.min(), vals.max()
    span = f_max - f_min
    if span < 1e-12:
        return {id(ind): 0.5 for ind in pop}
    return {id(ind): (ind.fitness - f_min) / span for ind in pop}

def biological_duel(cancer_pop, lymph_pop, ga_instance):
    """Resuelve la interacción probabilística entre cáncer y linfocitos."""
    survivors_c, survivors_l = [], []
    matched_l = set()

    norm_c = _normalize_fitness(cancer_pop)
    norm_l = _normalize_fitness(lymph_pop)

    for c in cancer_pop:
        opponent = None
        best_d = ga_instance.duel_radius
        for j, l in enumerate(lymph_pop):
            if j in matched_l:
                continue
            d = np.linalg.norm(c.x - l.x)
            if d < best_d:
                best_d, opponent = d, j

        if opponent is None:
            survivors_c.append(c)
            continue

        matched_l.add(opponent)
        l = lymph_pop[opponent]
        affinity = np.exp(-(best_d ** 2) / (2 * (ga_instance.duel_radius / 2) ** 2))
        fc_n, fl_n = norm_c[id(c)], norm_l[id(l)]
        z = ga_instance.duel_k * (affinity * fl_n - fc_n)
        p_cancer_dies = 1.0 / (1.0 + np.exp(-z))
        
        # Escudo parcial por fibroblastos
        p_cancer_dies *= (1.0 - ga_instance.fibro_shield * ga_instance._phi_of(c))

        if ga_instance.rng.random() < p_cancer_dies:
            l.energy = min(l.energy + 0.3, 2.0)
            survivors_l.append(l)
        else:
            c.energy = min(c.energy + 0.2, 2.0)
            l.energy -= 0.4
            survivors_c.append(c)
            if l.energy > 0:
                survivors_l.append(l)

    for j, l in enumerate(lymph_pop):
        if j not in matched_l:
            survivors_l.append(l)

    return survivors_c, survivors_l