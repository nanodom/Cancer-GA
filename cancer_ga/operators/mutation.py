import numpy as np

def adaptive_mutation(pop, ga_instance):
    """Aplica mutación auto-adaptativa tipo ES a la población."""
    n = ga_instance.n_dim
    tau = 1.0 / np.sqrt(2 * np.sqrt(n))
    tau_p = 1.0 / np.sqrt(2 * n)
    has_int = np.any(ga_instance.is_integer)
    
    for ind in pop:
        global_term = tau_p * ga_instance.rng.normal()
        local_term = tau * ga_instance.rng.normal()
        ind.sigma = max(ga_instance.sigma_min, ind.sigma * np.exp(global_term + local_term))

        new_x = ind.x.copy()
        cont_mask = ~ga_instance.is_integer
        
        # Variables continuas: paso gaussiano auto-adaptativo
        new_x[cont_mask] = new_x[cont_mask] + ind.sigma * ga_instance.rng.normal(size=cont_mask.sum())

        if has_int:
            # Variables enteras: paso entero proporcional a sigma
            int_idx = np.where(ga_instance.is_integer)[0]
            step_mag = max(1, int(round(ind.sigma)))
            int_steps = ga_instance.rng.integers(-step_mag, step_mag + 1, size=len(int_idx))
            new_x[int_idx] = new_x[int_idx] + int_steps

        ind.x = ga_instance._repair(new_x)