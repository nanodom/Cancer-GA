import numpy as np

def starvation(pop, ga_instance):
    """Calcula la pérdida/ganancia energética y elimina especímenes sin energía."""
    if not pop:
        return pop
        
    positions = np.array([ind.x for ind in pop])
    
    # Mapeo min-max del fitness
    vals = np.array([ind.fitness for ind in pop])
    f_min, f_max = vals.min(), vals.max()
    span = f_max - f_min
    if span < 1e-12:
        f_norm_map = {id(ind): 0.5 for ind in pop}
    else:
        f_norm_map = {id(ind): (ind.fitness - f_min) / span for ind in pop}

    survivors = []
    for ind in pop:
        dists = np.linalg.norm(positions - ind.x, axis=1)
        density = np.sum(dists < ga_instance.niche_radius) - 1
        f_norm = f_norm_map[id(ind)]
        fibro_bonus = ga_instance.fibro_energy_boost * ga_instance._phi_of(ind)
        
        ind.energy += (
            -ga_instance.e_maintenance
            + ga_instance.e_gain * f_norm
            + fibro_bonus
            - ga_instance.e_comp * density
        )
        if ind.energy > 0:
            survivors.append(ind)
            
    return survivors