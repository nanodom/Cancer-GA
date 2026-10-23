import numpy as np


def adaptive_mutation(pop, ga_instance):
    """Aplica mutación auto-adaptativa tipo ES a la población.

    Si ``ga_instance.mutation_prob`` es None se perturban todos los genes (comportamiento original). Si es un
    valor en (0, 1], cada gen se muta con esa probabilidad (siempre al menos uno): imprescindible en problemas
    de alta dimensión con codificación de permutaciones/claves, donde mutar todos los genes equivale a reiniciar.
    """
    n = ga_instance.n_dim
    tau = 1.0 / np.sqrt(2 * np.sqrt(n))
    tau_p = 1.0 / np.sqrt(2 * n)
    has_int = np.any(ga_instance.is_integer)
    p_mut = getattr(ga_instance, "mutation_prob", None)

    for ind in pop:
        global_term = tau_p * ga_instance.rng.normal()
        local_term = tau * ga_instance.rng.normal()
        ind.sigma = max(ga_instance.sigma_min, ind.sigma * np.exp(global_term + local_term))

        if p_mut is None:
            gene_mask = np.ones(n, dtype=bool)
        else:
            gene_mask = ga_instance.rng.random(n) < p_mut
            if not gene_mask.any():
                gene_mask[ga_instance.rng.integers(n)] = True

        new_x = ind.x.copy()
        cont_mask = ~ga_instance.is_integer
        cmask = cont_mask & gene_mask

        # Variables continuas: paso gaussiano auto-adaptativo
        new_x[cmask] = new_x[cmask] + ind.sigma * ga_instance.rng.normal(size=cmask.sum())

        if has_int:
            # Variables enteras: paso entero proporcional a sigma
            int_idx = np.where(ga_instance.is_integer & gene_mask)[0]
            step_mag = max(1, int(round(ind.sigma)))
            int_steps = ga_instance.rng.integers(-step_mag, step_mag + 1, size=len(int_idx))
            new_x[int_idx] = new_x[int_idx] + int_steps

        ind.x = ga_instance._repair(new_x)
