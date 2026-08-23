import numpy as np

def fast_non_dominated_sort(pop, feas_eps=1e-6):
    """Ordenamiento por frentes usando dominancia RESTRINGIDA (regla de Deb):
      1) Un individuo factible siempre domina a uno infactible.
      2) Entre dos infactibles, domina el de menor violación total (cv).
      3) Entre dos factibles, aplica la dominancia de Pareto estándar.
    """
    n = len(pop)
    if n == 0:
        return []
        
    objs = np.array([ind.obj for ind in pop])          # (N, M)
    cvs = np.array([ind.cv for ind in pop])             # (N,)
    feasible = cvs <= feas_eps

    feas_i, feas_j = feasible[:, None], feasible[None, :]
    cv_i, cv_j = cvs[:, None], cvs[None, :]

    diff = objs[:, None, :] - objs[None, :, :]          # (N, N, M)
    leq_all = np.all(diff <= 1e-12, axis=2)
    lt_any = np.any(diff < -1e-12, axis=2)
    pareto_dom = leq_all & lt_any

    dom_matrix = np.zeros((n, n), dtype=bool)
    dom_matrix |= feas_i & feas_j & pareto_dom                   # Ambos factibles -> Pareto
    dom_matrix |= feas_i & (~feas_j)                             # Factible domina a infactible
    dom_matrix |= (~feas_i) & (~feas_j) & (cv_i < cv_j - 1e-12)  # Ambos infactibles -> menor cv
    np.fill_diagonal(dom_matrix, False)

    domination_count = dom_matrix.sum(axis=0)
    dominated_by = [np.nonzero(dom_matrix[i])[0] for i in range(n)]

    fronts_idx = [[]]
    for i in range(n):
        if domination_count[i] == 0:
            pop[i].rank = 0
            fronts_idx[0].append(i)

    k = 0
    domination_count = domination_count.copy()
    while fronts_idx[k]:
        next_front = []
        for i in fronts_idx[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    pop[j].rank = k + 1
                    next_front.append(j)
        k += 1
        fronts_idx.append(next_front)
    fronts_idx.pop()

    return [[pop[i] for i in front] for front in fronts_idx]


def crowding_distance(front, n_obj):
    """Calcula la distancia de aglomeración dentro de un mismo frente de Pareto."""
    crowd = {id(ind): 0.0 for ind in front}
    if len(front) <= 2:
        for ind in front:
            crowd[id(ind)] = np.inf
        return crowd

    for k in range(n_obj):
        front_sorted = sorted(front, key=lambda ind: ind.obj[k])
        crowd[id(front_sorted[0])] = np.inf
        crowd[id(front_sorted[-1])] = np.inf
        span = front_sorted[-1].obj[k] - front_sorted[0].obj[k]
        if span < 1e-12:
            continue
        for idx in range(1, len(front_sorted) - 1):
            prev_o = front_sorted[idx - 1].obj[k]
            next_o = front_sorted[idx + 1].obj[k]
            crowd[id(front_sorted[idx])] += (next_o - prev_o) / span
            
    return crowd


def assign_rank_and_crowding(pop, n_obj, feas_eps=1e-6):
    """Asigna rank, crowd y fitness compuesto a los individuos de la población."""
    if not pop:
        return

    if n_obj == 1:
        # Atajo O(N log N) para caso mono-objetivo
        n = len(pop)

        def sort_key(ind):
            if ind.cv <= feas_eps:
                return (0, ind.obj[0])
            return (1, ind.cv)

        order = sorted(pop, key=sort_key)
        denom = max(n - 1, 1)
        for idx, ind in enumerate(order):
            ind.rank = idx
            ind.crowd = 0.0
            ind.fitness = 1.0 - idx / denom
        return

    fronts = fast_non_dominated_sort(pop, feas_eps=feas_eps)
    max_rank = max(len(fronts) - 1, 1)
    for front in fronts:
        crowd_map = crowding_distance(front, n_obj)
        finite_vals = [c for c in crowd_map.values() if np.isfinite(c)]
        c_max = max(finite_vals) if finite_vals else 1.0
        
        for ind in front:
            ind.crowd = crowd_map[id(ind)]
            c_norm = 1.0 if not np.isfinite(ind.crowd) else (
                ind.crowd / c_max if c_max > 0 else 0.5
            )
            rank_term = 1.0 - (ind.rank / max_rank) if max_rank > 0 else 1.0
            ind.fitness = rank_term + 0.05 * c_norm