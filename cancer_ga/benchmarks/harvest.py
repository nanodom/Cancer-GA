"""
Benchmark del modelo bi-objetivo CP de re-planificación de cosecha de
aceitunas (paper de Urrejola-Arrellano & Montenegro-Dos Santo), listo para
usarse con CancerGA de la misma forma que CTP/CF/MW.

--------------------------------------------------------------------------
Formulación original (paper, Sección 3.1)
--------------------------------------------------------------------------
Variables: x_jtk (tons cosechadas del lote j, día t, equipo k), h_jtk/y_jtk
(binarias de continuidad/inicio), o_jt (inventario remanente).
Objetivos: max f1 = sum x_jtk * OC_jt  (producción de aceite)
           min f2 = discrepancia con el plan base r_jtk (contrato puntual o
                    agregado).
Restricciones: continuidad de cosecha (un equipo, una vez iniciado un lote,
lo sigue cosechando hasta terminarlo o hasta el fin del horizonte), un lote
por equipo por día, balance de inventario, capacidad diaria de la planta.

--------------------------------------------------------------------------
Codificación usada aquí para CancerGA
--------------------------------------------------------------------------
CancerGA opera sobre vectores reales/mixtos con reparo por caja
(clip + redondeo de enteros), no resuelve MILPs directamente. Por eso el
modelo combinatorio se codifica con un esquema de "llaves aleatorias" +
decodificador determinista (Schedule Generation Scheme), técnica estándar en
metaheurísticas para scheduling:

  x = [team_0, ..., team_{J-1}, priority_0, ..., priority_{J-1}]

  - team_j (entero, is_integer=True): recurso preferido para el lote j.
    Si el recurso elegido no es elegible para ese lote (p.ej. una máquina en
    un lote de ladera), se repara determinísticamente al recurso elegible
    más cercano.
  - priority_j (continuo en [0,1]): orden de cosecha del lote j dentro de la
    cola de su recurso asignado.

El decodificador (`simulate_plan`) construye, para cada recurso, una cola de
lotes ordenada por prioridad y los procesa secuencialmente día a día
—garantizando por construcción la continuidad de cosecha (3), un lote por
equipo por día (5) y el inicio único por lote (4)—, y reparte la capacidad
diaria de la planta (9) escalando proporcionalmente las propuestas del día
si se excede. Esto deja el problema como una optimización de caja
(bounds + enteros) sin restricciones adicionales que CancerGA deba manejar
explícitamente: toda la factibilidad estructural queda garantizada por el
decodificador.

f1 se devuelve negado (-producción de aceite) porque CancerGA/NSGA-II en
este repo asumen minimización de todos los objetivos (ver
`utils/nsga2.fast_non_dominated_sort`, donde "menor es mejor").
"""

import numpy as np

from .harvest_instance import HarvestInstance, generate_paper_instance, generate_tabali_instance, read_tabali_lots

__all__ = [
    "HarvestInstance",
    "generate_paper_instance",
    "generate_tabali_instance",
    "read_tabali_lots",
    "simulate_plan",
    "generate_baseline",
    "make_harvest_objective",
    "make_contract_function",
    "make_harvest_normalizer",
    "make_oil_pvi_normalizer",
    "plan_variability_index",
    "CONTRACT_TYPES",
    "CONTRACT_GROUPS",
    "HOURS_PER_DAY",
    "harvest_cp_punctual",
    "harvest_cp_aggregated",
    "harvest_bounds",
    "harvest_is_integer",
]


def simulate_plan(instance, team_gene, priority_gene, disabled_days=None):
    """Decodifica (team_gene, priority_gene) en un plan de cosecha factible.

    Devuelve:
      plan: dict {(j, t, k): tons} sólo con entradas no nulas (representación
            dispersa; a lo más horizon_days * n_resources entradas).
      resource_totals: array (n_resources,) con el total cosechado por cada
            recurso en todo el horizonte (usado para el contrato agregado).
      total_oil: producción total de aceite (f1 sin negar).
    """
    n_res = instance.n_resources
    n_lots = instance.n_lots

    # --- Reparo de elegibilidad: recurso propuesto -> recurso elegible más cercano ---
    team_idx = np.clip(np.round(np.asarray(team_gene)), 0, n_res - 1).astype(int)
    fixed_team = np.empty(n_lots, dtype=int)
    for j in range(n_lots):
        elig = instance.eligible_resources(j)
        if len(elig) == 0:
            continue  # no debería ocurrir con las instancias generadas aquí
        fixed_team[j] = team_idx[j] if team_idx[j] in elig else elig[np.argmin(np.abs(elig - team_idx[j]))]

    # --- Colas de cosecha por recurso, ordenadas por prioridad ---
    queues = [[] for _ in range(n_res)]
    for j in range(n_lots):
        queues[fixed_team[j]].append(j)
    for k in range(n_res):
        queues[k].sort(key=lambda j: priority_gene[j])

    remaining = instance.tons.copy()
    ptr = [0] * n_res
    plan = {}
    resource_totals = np.zeros(n_res)
    total_oil = 0.0

    for t in range(instance.horizon_days):
        proposals = np.zeros(n_res)
        current_lot = [None] * n_res

        for k in range(n_res):
            if disabled_days is not None and t in disabled_days.get(k, ()):
                continue
            while ptr[k] < len(queues[k]) and remaining[queues[k][ptr[k]]] <= 1e-9:
                ptr[k] += 1
            if ptr[k] >= len(queues[k]):
                continue
            j = queues[k][ptr[k]]
            current_lot[k] = j
            proposals[k] = min(instance.capacity[k], remaining[j])

        total_proposed = proposals.sum()
        cap_today = instance.mill_capacity[t]
        factor = 1.0
        if total_proposed > cap_today and total_proposed > 1e-9:
            factor = cap_today / total_proposed  # reparto proporcional (restricción 9)

        for k in range(n_res):
            j = current_lot[k]
            if j is None or proposals[k] <= 0:
                continue
            amt = proposals[k] * factor
            if amt <= 1e-9:
                continue
            remaining[j] -= amt
            plan[(j, t, k)] = amt
            resource_totals[k] += amt
            total_oil += amt * instance.OC[j, t]

    return plan, resource_totals, total_oil


def generate_baseline(instance):
    """Genera el plan base (BP) que define r_jtk (Step 0 del esquema RH del
    paper): un plan factible que prioriza cosechar cada lote cerca del punto
    de mayor concentración de aceite de su variedad, balanceando la carga
    entre los recursos elegibles."""
    n_lots, n_res = instance.n_lots, instance.n_resources

    # Prioridad: día donde la curva de aceite crece más rápido (aprox. al
    # punto de inflexión / cosecha "óptima" antes de sobre-madurar).
    priority = np.array([np.argmax(np.gradient(instance.OC[j])) for j in range(n_lots)], dtype=float)

    # Asignación balanceada: lotes más grandes primero, cada uno al recurso
    # elegible con menor carga acumulada hasta el momento.
    order = np.argsort(-instance.tons)
    load = np.zeros(n_res)
    team_gene = np.zeros(n_lots)
    for j in order:
        elig = instance.eligible_resources(j)
        best_k = elig[np.argmin(load[elig])]
        team_gene[j] = best_k
        load[best_k] += instance.tons[j]

    plan, totals, oil = simulate_plan(instance, team_gene, priority)
    return plan, totals, oil, team_gene, priority


def harvest_bounds(instance):
    """Límites [team_j en [0, n_resources-1], priority_j en [0, 1]] para
    usar directamente como `bounds` de CancerGA."""
    n = instance.n_lots
    bounds = np.zeros((2 * n, 2))
    bounds[:n, 0] = 0
    bounds[:n, 1] = instance.n_resources - 1
    bounds[n:, 0] = 0.0
    bounds[n:, 1] = 1.0
    return bounds


def harvest_is_integer(instance):
    """Máscara is_integer: True para los genes de asignación de recurso,
    False para los de prioridad."""
    n = instance.n_lots
    return np.array([True] * n + [False] * n)


# --------------------------------------------------------------------------
# Contratos (paper, Sección 5.3) y Plan Variability Index
# --------------------------------------------------------------------------
# El paper describe los contratos verbalmente (no da fórmulas). Las
# definiciones de abajo son una OPERACIONALIZACIÓN fiel a esas descripciones;
# conviene validarlas con el profesor antes de reportar resultados.
#
# El número de contratos agrupa los recursos comprometidos (Sección 5.3):
#   1 contrato : todos los recursos
#   2 contratos: {Machine, ShakerA, ShakerB}  |  {Manual}
#   3 contratos: {Machine}  |  {ShakerA, ShakerB}  |  {Manual}
CONTRACT_TYPES = ("punctual", "aggregated", "minimum_guaranteed", "by_surface", "by_work_hours")

CONTRACT_GROUPS = {
    1: [{"Machine", "ShakerA", "ShakerB", "Manual"}],
    2: [{"Machine", "ShakerA", "ShakerB"}, {"Manual"}],
    3: [{"Machine"}, {"ShakerA", "ShakerB"}, {"Manual"}],
}

# Horas de trabajo por día de un equipo a capacidad nominal (contrato by_work_hours)
HOURS_PER_DAY = 8.0


def plan_variability_index(instance, plan, baseline_plan):
    """Plan Variability Index (PVI): desviación absoluta diaria promedio entre
    el plan nuevo y el plan base, en toneladas de aceituna por día:

        PVI = (1 / |T|) * sum_{<j,t,k>} |x_jtk - r_jtk|

    A diferencia de f2, el PVI NO depende del tipo de contrato, por eso es la
    métrica para comparar contratos entre sí (el paper advierte que los f2 de
    contratos distintos no son comparables).

    SUPUESTO: el paper nombra el PVI y reporta que se mide en "toneladas de
    aceituna/día", pero la fórmula exacta no aparece en el texto disponible.
    Esta es la lectura natural de esas unidades; reemplazar esta función si
    el paper (o el profesor) usa otra definición.
    """
    keys = set(plan) | set(baseline_plan)
    total = sum(abs(plan.get(k, 0.0) - baseline_plan.get(k, 0.0)) for k in keys)
    return float(total / instance.horizon_days)


def _resource_groups(instance, n_contracts):
    if n_contracts not in CONTRACT_GROUPS:
        raise ValueError(f"n_contracts debe ser uno de {sorted(CONTRACT_GROUPS)}")
    groups = CONTRACT_GROUPS[n_contracts]
    res_group = np.full(instance.n_resources, -1, dtype=int)
    for k, res in enumerate(instance.resources):
        for g, kinds in enumerate(groups):
            if res["kind"] in kinds:
                res_group[k] = g
    if np.any(res_group < 0):
        raise ValueError("Hay recursos que no pertenecen a ningún contrato")
    return res_group, len(groups)


def make_contract_function(instance, baseline_plan, contract_type="punctual", n_contracts=1):
    """Devuelve f2(plan) -> float para el tipo de contrato indicado.

    Sea c un contrato (grupo de recursos), r el plan base y x el plan nuevo:

    punctual           f2 = sum_{<j,t,k>} |x_jtk - r_jtk|
                       Lote, día y recurso fijos. (No depende de n_contracts.)

    aggregated         f2 = sum_c sum_{(j,t)} | sum_{k in c} (x_jtk - r_jtk) |
                       Fija lote y día pero permite intercambiar recursos
                       dentro del contrato ("differences in the utilization
                       of the resources", Sección 3).

    minimum_guaranteed f2 = sum_c sum_t max(0, R_ct - X_ct)
                       con R_ct, X_ct = toneladas diarias del grupo c en el
                       plan base / nuevo. Sólo penaliza quedar por DEBAJO del
                       mínimo comprometido; exceder es gratis ("harvest the
                       quantities fixed in the contracts or even exceed them").

    by_surface         f2 = sum_{j} surface_j * | b_j - n_j |
                       Cada lote se compromete con el grupo que más lo cosechó
                       en el plan base; b_j y n_j son la fracción del lote
                       cosechada por ese grupo en el plan base / nuevo.
                       Unidades: hectáreas. ("field divided into sectors ...
                       harvested with a specific work team".)

    by_work_hours      f2 = sum_c | sum_{k in c} (X_k - R_k) * H / cap_k |
                       con H = HOURS_PER_DAY horas y cap_k la capacidad diaria
                       del recurso k. Controla las horas-equipo comprometidas:
                       mover toneladas de una máquina rápida a un equipo
                       manual cambia las horas aunque las toneladas no.
                       Unidades: horas.

    Estas fórmulas son una operacionalización de las descripciones verbales
    del paper (ver nota en la cabecera de la sección).
    """
    if contract_type not in CONTRACT_TYPES:
        raise ValueError(f"contract_type debe ser uno de {CONTRACT_TYPES}")

    res_group, n_g = _resource_groups(instance, n_contracts)
    n_lots, n_days = instance.n_lots, instance.horizon_days

    if contract_type == "punctual":

        def f2_punctual(plan):
            keys = plan.keys() | baseline_plan.keys()
            return float(sum(abs(plan.get(k, 0.0) - baseline_plan.get(k, 0.0)) for k in keys))

        return f2_punctual

    if contract_type == "aggregated":
        neg_base = {}
        for (j, t, k), amt in baseline_plan.items():
            key = (j, t, int(res_group[k]))
            neg_base[key] = neg_base.get(key, 0.0) - amt

        def f2_aggregated(plan):
            diff = dict(neg_base)
            for (j, t, k), amt in plan.items():
                key = (j, t, int(res_group[k]))
                diff[key] = diff.get(key, 0.0) + amt
            return float(sum(abs(v) for v in diff.values()))

        return f2_aggregated

    if contract_type == "minimum_guaranteed":
        base_gt = np.zeros((n_g, n_days))
        for (j, t, k), amt in baseline_plan.items():
            base_gt[res_group[k], t] += amt

        def f2_min_guaranteed(plan):
            new_gt = np.zeros((n_g, n_days))
            for (j, t, k), amt in plan.items():
                new_gt[res_group[k], t] += amt
            return float(np.maximum(0.0, base_gt - new_gt).sum())

        return f2_min_guaranteed

    if contract_type == "by_surface":
        base_jg = np.zeros((n_lots, n_g))
        for (j, t, k), amt in baseline_plan.items():
            base_jg[j, res_group[k]] += amt
        committed = np.argmax(base_jg, axis=1)
        has_commitment = base_jg.sum(axis=1) > 1e-9
        lot_idx = np.arange(n_lots)
        tons = instance.tons
        b_frac = base_jg[lot_idx, committed] / tons

        def f2_by_surface(plan):
            new_jg = np.zeros((n_lots, n_g))
            for (j, t, k), amt in plan.items():
                new_jg[j, res_group[k]] += amt
            n_frac = new_jg[lot_idx, committed] / tons
            gap = np.abs(b_frac - n_frac) * instance.surface
            return float(gap[has_commitment].sum())

        return f2_by_surface

    # by_work_hours
    hours_per_ton = HOURS_PER_DAY / instance.capacity  # (n_res,)
    base_hours = np.zeros(n_g)
    for (j, t, k), amt in baseline_plan.items():
        base_hours[res_group[k]] += amt * hours_per_ton[k]

    def f2_by_work_hours(plan):
        new_hours = np.zeros(n_g)
        for (j, t, k), amt in plan.items():
            new_hours[res_group[k]] += amt * hours_per_ton[k]
        return float(np.abs(new_hours - base_hours).sum())

    return f2_by_work_hours


def make_harvest_objective(
    instance, baseline_plan, baseline_totals=None, contract_type="punctual", disabled_days=None, n_contracts=1
):
    """Fábrica de función objetivo compatible con CancerGA:
    obj(x) -> (np.array([f1, f2]), [])  (f1 = -aceite, f2 según el contrato).

    `baseline_totals` se conserva por compatibilidad con versiones anteriores
    pero ya no se usa (los contratos se calculan desde `baseline_plan`).
    """
    n = instance.n_lots
    f2_fn = make_contract_function(instance, baseline_plan, contract_type, n_contracts)

    def objective(x):
        x = np.asarray(x, dtype=float)
        plan, _, total_oil = simulate_plan(instance, x[:n], x[n:], disabled_days=disabled_days)
        return np.array([-total_oil, f2_fn(plan)]), []

    return objective


def make_harvest_normalizer(instance, objective, n_samples=200, seed=12345, margin=0.1):
    """Normalizador de objetivos con cotas FIJAS para calcular el hipervolumen.

    - Ideal f1 = -(aceite máximo teórico): cada lote cosechado completo en su
      día de mayor concentración de aceite. Ideal f2 = 0 (plan idéntico al base).
    - Nadir = peor valor observado en `n_samples` soluciones aleatorias más un
      margen relativo.

    Usa su propio generador (`seed`), independiente de la semilla de la corrida
    de CancerGA: así todas las semillas de un mismo problema comparten el mismo
    punto de referencia y sus hipervolúmenes son comparables entre sí.
    NO son comparables entre tipos de contrato (f2 cambia de escala).
    """
    from cancer_ga.utils.metrics import ObjectiveNormalizer

    rng = np.random.default_rng(seed)
    bounds = harvest_bounds(instance)
    samples = []
    for _ in range(n_samples):
        x = rng.uniform(bounds[:, 0], bounds[:, 1])
        x[harvest_is_integer(instance)] = np.round(x[harvest_is_integer(instance)])
        samples.append(objective(x)[0])

    max_oil = float(np.sum(instance.tons * instance.OC.max(axis=1)))
    return ObjectiveNormalizer.from_samples(np.array(samples), ideal=[-max_oil, 0.0], margin=margin)


def make_oil_pvi_normalizer(instance, baseline_plan, n_samples=200, seed=12345, margin=0.1):
    """Normalizador en el espacio (-aceite, PVI), COMÚN a todos los contratos.

    f2 cambia de definición y de unidades según el contrato, así que el
    hipervolumen en (f1, f2) no sirve para compararlos. El PVI, en cambio, se
    define igual para cualquier contrato. Este normalizador usa cotas que sólo
    dependen de la instancia (no del contrato), de modo que el hipervolumen en
    el espacio (aceite, PVI) SÍ es comparable entre tipos de contrato.
    """
    from cancer_ga.utils.metrics import ObjectiveNormalizer

    rng = np.random.default_rng(seed)
    bounds = harvest_bounds(instance)
    int_mask = harvest_is_integer(instance)
    samples = []
    for _ in range(n_samples):
        x = rng.uniform(bounds[:, 0], bounds[:, 1])
        x[int_mask] = np.round(x[int_mask])
        plan, _, oil = simulate_plan(instance, x[: instance.n_lots], x[instance.n_lots :])
        samples.append([-oil, plan_variability_index(instance, plan, baseline_plan)])

    max_oil = float(np.sum(instance.tons * instance.OC.max(axis=1)))
    return ObjectiveNormalizer.from_samples(np.array(samples), ideal=[-max_oil, 0.0], margin=margin)


# --------------------------------------------------------------------------
# Instancia lista para usar, a la escala del paper (63 lotes, 45 días, 11
# recursos individuales según la Tabla 1), con su plan base ya calculado.
# Se construye perezosamente la primera vez que se importa/usa.
# --------------------------------------------------------------------------
_PAPER_INSTANCE = None
_PAPER_BASELINE = None


def _get_paper_instance():
    global _PAPER_INSTANCE, _PAPER_BASELINE
    if _PAPER_INSTANCE is None:
        _PAPER_INSTANCE = generate_paper_instance()
        plan, totals, oil, team_gene, priority = generate_baseline(_PAPER_INSTANCE)
        _PAPER_BASELINE = {"plan": plan, "totals": totals, "oil": oil, "team_gene": team_gene, "priority": priority}
    return _PAPER_INSTANCE, _PAPER_BASELINE


def harvest_cp_punctual(x):
    """f2 = contrato puntual (discrepancia lote-día-equipo, ecuación (1) del paper)."""
    instance, bl = _get_paper_instance()
    obj = make_harvest_objective(instance, bl["plan"], bl["totals"], contract_type="punctual")
    return obj(x)


def harvest_cp_aggregated(x):
    """f2 = contrato agregado (discrepancia del total cosechado por recurso)."""
    instance, bl = _get_paper_instance()
    obj = make_harvest_objective(instance, bl["plan"], bl["totals"], contract_type="aggregated")
    return obj(x)
