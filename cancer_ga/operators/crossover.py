from cancer_ga.individual import Individual

def crossover(pop, ga_instance):
    """Aplica cruce intra-poblacional mediante BLX-alpha."""
    if len(pop) < 2:
        return pop
    children = []
    pop_sorted = sorted(pop, key=lambda i: i.fitness, reverse=True)
    n_pairs = len(pop) // 2
    
    for _ in range(n_pairs):
        if ga_instance.rng.random() > ga_instance.crossover_rate:
            continue
        i1, i2 = ga_instance.rng.choice(len(pop_sorted), size=2, replace=False)
        p1, p2 = pop_sorted[i1], pop_sorted[i2]
        lam = ga_instance.rng.uniform(-ga_instance.blx_alpha, 1 + ga_instance.blx_alpha, size=ga_instance.n_dim)
        child_x = ga_instance._repair(lam * p1.x + (1 - lam) * p2.x)
        child_sigma = (p1.sigma + p2.sigma) / 2
        child_energy = (p1.energy + p2.energy) / 2
        
        if p1.weights is not None and p2.weights is not None:
            w = p1.weights + p2.weights
            child_weights = w / w.sum()
        else:
            child_weights = None
            
        children.append(Individual(child_x, child_sigma, child_energy, weights=child_weights))
        
    return pop + children