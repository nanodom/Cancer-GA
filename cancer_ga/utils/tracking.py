"""
Ejecución instrumentada de CancerGA para problemas bi-objetivo.

`CancerGA.run()` sólo devuelve el mejor individuo final. Para analizar la
convergencia real y comparar corridas hace falta además:

  - un ARCHIVO ÉLITE (todas las soluciones no dominadas halladas hasta el
    momento, no sólo las de la población actual),
  - el HIPERVOLUMEN de ese archivo en cada generación,
  - un CRITERIO DE PARADA basado en estancamiento del hipervolumen,
  - un registro por generación (f1, f2, hipervolumen, tamaños, tiempo).

Se mide el hipervolumen del archivo (no de la población) a propósito: el
archivo sólo puede mejorar o mantenerse, así que su HV es monótono no
decreciente y "N generaciones sin mejora" tiene un significado preciso.

NOTA DE MANTENIMIENTO: `run_tracked` reproduce el ciclo de `CancerGA.run()`
(mismos operadores, mismo orden, sin consumir números aleatorios extra), de
modo que con la misma semilla recorre exactamente la misma trayectoria.
`tests/test_tracking.py` verifica esa equivalencia; si cambias el ciclo en
`algorithm.py`, ese test avisará.
"""

import time

import numpy as np

from cancer_ga.operators.mutation import adaptive_mutation
from cancer_ga.operators.chemotaxis import chemotaxis
from cancer_ga.operators.duel import biological_duel
from cancer_ga.operators.crossover import crossover
from cancer_ga.operators.proliferation import niche_proliferation, compute_fibroblast_field
from cancer_ga.operators.metastasis import metastasis
from cancer_ga.operators.starvation import starvation
from cancer_ga.utils.metrics import nondominated_mask


def _update_archive(archive, pop, feas_eps):
    """Fusiona los individuos factibles de `pop` en el archivo y deja sólo los
    no dominados. `archive` es una lista de dicts {"x": ndarray, "obj": ndarray}."""
    candidates = [{"x": ind.x.copy(), "obj": ind.obj.copy()} for ind in pop if ind.cv <= feas_eps]
    if not candidates:
        return archive
    merged = archive + candidates
    mask = nondominated_mask(np.array([m["obj"] for m in merged]))
    return [m for m, keep in zip(merged, mask) if keep]


def _log_entry(gen, archive, normalizer, cancer_pop, lymph_pop, t0):
    objs = np.array([m["obj"] for m in archive]) if archive else np.empty((0, 2))
    return {
        "gen": gen,
        "hv": float(normalizer.hypervolume(objs)) if len(objs) else 0.0,
        "best_f1": float(objs[:, 0].min()) if len(objs) else None,  # mejor f1 del archivo (f1 = -aceite)
        "min_f2": float(objs[:, 1].min()) if len(objs) else None,  # mejor f2 del archivo
        "archive_size": int(len(archive)),
        "cancer_size": len(cancer_pop),
        "lymph_size": len(lymph_pop),
        "elapsed_s": float(time.time() - t0),
    }


def run_tracked(ga, normalizer, max_generations=100, patience=15, min_delta=1e-4, verbose=False, log_every=10):
    """Corre CancerGA (bi-objetivo) registrando f1, f2 e hipervolumen por generación.

    Parámetros
    ----------
    ga            : instancia de CancerGA con n_obj == 2.
    normalizer    : ObjectiveNormalizer (cotas fijas; define el HV comparable).
    max_generations : tope duro de generaciones.
    patience      : N_max de generaciones consecutivas SIN mejora del HV tras
                    las cuales se detiene. None desactiva el criterio.
    min_delta     : mejora mínima de HV (en escala normalizada [0, 1]) que
                    cuenta como "mejora". Evita seguir corriendo por ganancias
                    numéricamente irrelevantes.

    Devuelve un dict con:
      log            lista de registros por generación (gen 0 = población inicial)
      archive        soluciones no dominadas finales [{"x", "obj"}]
      best           (x, obj, factible) igual que `CancerGA.run()`
      generations_run, stop_reason ("patience" | "max_generations"), best_hv, elapsed_s
    """
    if ga.n_obj != 2:
        raise ValueError("run_tracked soporta problemas bi-objetivo (n_obj == 2)")

    t0 = time.time()
    cancer_pop = ga._init_population(ga.n_cancer)
    lymph_pop = ga._init_population(ga.n_lymph)
    ga.evaluate(cancer_pop)
    ga.evaluate(lymph_pop)

    archive = _update_archive([], cancer_pop, ga.feas_eps)
    log = [_log_entry(0, archive, normalizer, cancer_pop, lymph_pop, t0)]
    best_hv = log[0]["hv"]
    stagnation = 0
    stop_reason = "max_generations"

    for gen in range(1, max_generations + 1):
        if ga.use_fibroblasts:
            ga.fibro_phi = compute_fibroblast_field(cancer_pop, ga)

        adaptive_mutation(cancer_pop, ga)
        adaptive_mutation(lymph_pop, ga)

        chemotaxis(cancer_pop, ga)
        chemotaxis(lymph_pop, ga)

        ga.evaluate(cancer_pop)
        ga.evaluate(lymph_pop)

        cancer_pop, lymph_pop = biological_duel(cancer_pop, lymph_pop, ga)
        cancer_pop = crossover(cancer_pop, ga)
        cancer_pop = niche_proliferation(cancer_pop, ga.n_cancer, ga)
        metastasis(cancer_pop, ga)

        ga.evaluate(cancer_pop)
        ga.evaluate(lymph_pop)

        cancer_pop = starvation(cancer_pop, ga)
        lymph_pop = starvation(lymph_pop, ga)

        if len(cancer_pop) < 5:
            cancer_pop.extend(ga._init_population(ga.n_cancer - len(cancer_pop)))
        if len(lymph_pop) < 2:
            lymph_pop.extend(ga._init_population(ga.n_lymph - len(lymph_pop)))

        ga.evaluate(cancer_pop)
        ga.evaluate(lymph_pop)

        archive = _update_archive(archive, cancer_pop, ga.feas_eps)
        entry = _log_entry(gen, archive, normalizer, cancer_pop, lymph_pop, t0)
        log.append(entry)

        if entry["hv"] > best_hv + min_delta:
            best_hv = entry["hv"]
            stagnation = 0
        else:
            stagnation += 1

        if verbose and (gen == 1 or gen % log_every == 0):
            print(
                f"  gen {gen:3d} | HV {entry['hv']:.4f} | mejor aceite {-entry['best_f1']:.1f} | "
                f"min f2 {entry['min_f2']:.1f} | archivo {entry['archive_size']:3d} | "
                f"sin mejora {stagnation}"
            )

        if patience is not None and stagnation >= patience:
            stop_reason = "patience"
            break

    # Mejor individuo "a la CancerGA.run()" (para verificar equivalencia con run())
    feasible_c = [ind for ind in cancer_pop if ind.cv <= ga.feas_eps]
    if feasible_c:
        b = max(feasible_c, key=lambda i: i.fitness)
        best = (b.x.copy(), b.obj.copy(), True)
    else:
        b = min(cancer_pop, key=lambda i: i.cv)
        best = (b.x.copy(), b.obj.copy(), False)

    return {
        "log": log,
        "archive": archive,
        "best": best,
        "generations_run": log[-1]["gen"],
        "stop_reason": stop_reason,
        "best_hv": float(log[-1]["hv"]),
        "elapsed_s": float(time.time() - t0),
    }
