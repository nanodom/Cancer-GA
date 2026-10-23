"""
Generador de instancias para el problema de re-planificación de cosecha de
aceitunas (modelo CP) descrito en:

  "A non-myopic Rolling Horizon scheme for rescheduling in agricultural
  harvest" (Urrejola-Arrellano & Montenegro-Dos Santo, 2026).

Reconstruye, a escala del paper, los datos que no vienen tabulados
explícitamente (curvas de concentración de aceite por variedad, tonelaje por
lote, accesibilidad de cada lote) de forma sintética pero razonable, y deja
fijos los datos que sí están tabulados en el paper (Tabla 1: equipos de
cosecha y sus capacidades).
"""

import numpy as np

# Tabla 1 del paper: modo de cosecha, cantidad disponible, capacidad (t/día)
HARVEST_MODES = [
    ("Machine", 1, [180.0]),
    ("ShakerA", 1, [26.0]),
    ("ShakerB", 2, [12.0, 6.0]),
    ("Manual", 7, [10.7] * 7),
]

# Modos que NO pueden operar en lotes de difícil acceso (pendientes/laderas)
RESTRICTED_ON_SLOPE = {"Machine"}

# Densidad de cosecha de respaldo (t/ha) para lotes sin superficie conocida.
# Es el promedio global de los 28 cuarteles reales de Tabalí (1393.5 t / 320.7 ha).
# Sólo se usa en el contrato "by_surface" cuando un lote no trae `has`.
TONS_PER_HA = 4.3


class HarvestInstance:
    """Datos de una instancia del modelo CP (re-planificación de cosecha)."""

    def __init__(self, lots, resources, mill_capacity, horizon_days, oil_curves, varieties):
        self.lots = lots  # list[dict]: id, variety, tons, slope
        self.resources = resources  # list[dict]: id, kind, capacity
        self.mill_capacity = np.asarray(mill_capacity, dtype=float)  # (horizon_days,)
        self.horizon_days = horizon_days
        self.oil_curves = oil_curves  # dict variety -> callable(day)->pct
        self.varieties = varieties

        self.n_lots = len(lots)
        self.n_resources = len(resources)

        # Precomputar elegibilidad lote-recurso (True si el recurso puede
        # cosechar ese lote) y matriz de concentración de aceite OC[j, t].
        self.eligible = np.ones((self.n_lots, self.n_resources), dtype=bool)
        for j, lot in enumerate(lots):
            if lot["slope"]:
                for k, res in enumerate(resources):
                    if res["kind"] in RESTRICTED_ON_SLOPE:
                        self.eligible[j, k] = False

        self.OC = np.zeros((self.n_lots, horizon_days))
        for j, lot in enumerate(lots):
            curve = oil_curves[lot["variety"]]
            self.OC[j, :] = [curve(t) for t in range(horizon_days)]

        self.capacity = np.array([r["capacity"] for r in resources])  # (n_resources,)
        self.tons = np.array([lot["tons"] for lot in lots])  # (n_lots,)
        # Superficie (ha) de cada lote: dato real si existe, si no tons / TONS_PER_HA
        self.surface = np.array([lot["has"] if lot.get("has") else lot["tons"] / TONS_PER_HA for lot in lots])

    def eligible_resources(self, j):
        return np.nonzero(self.eligible[j])[0]


def _oil_curve_factory(peak_day, oc_min, oc_max, width, rng):
    """Curva de acumulación de aceite tipo campana/logística por variedad:
    sube desde oc_min hasta oc_max, alcanzando el máximo en peak_day, y luego
    se mantiene (las aceitunas no pierden aceite una vez maduras, sólo se
    arriesgan a pérdida por clima, modelada aparte via OLjt)."""

    def curve(day):
        z = (day - peak_day) / width
        s = 1.0 / (1.0 + np.exp(-z))
        return oc_min + (oc_max - oc_min) * s

    return curve


def generate_paper_instance(
    n_lots=63,
    n_varieties=7,
    horizon_days=45,
    mill_capacity=250.0,
    slope_fraction=0.25,
    tons_range=(5.0, 25.0),
    seed=42,
):
    """Genera una instancia a la escala usada en el paper (Sección 5.1):
    horizonte de 45 días, 63 lotes repartidos en 7 variedades de aceitunas,
    y los 4 modos de cosecha de la Tabla 1 (11 recursos individuales:
    1 máquina, 1 shaker A, 2 shaker B, 7 equipos manuales)."""
    rng = np.random.default_rng(seed)

    # --- Recursos (Tabla 1) ---
    resources = []
    for kind, qty, caps in HARVEST_MODES:
        for i, cap in enumerate(caps):
            resources.append({"id": f"{kind}_{i + 1}", "kind": kind, "capacity": cap})

    # --- Curvas de concentración de aceite por variedad ---
    oil_curves = {}
    varieties = [f"V{i + 1}" for i in range(n_varieties)]
    for v in varieties:
        peak_day = rng.uniform(0.25, 0.75) * horizon_days
        oc_min = rng.uniform(8.0, 12.0)
        oc_max = oc_min + rng.uniform(8.0, 14.0)
        width = rng.uniform(4.0, 9.0)
        oil_curves[v] = _oil_curve_factory(peak_day, oc_min, oc_max, width, rng)

    # --- Lotes ---
    lots = []
    for j in range(n_lots):
        variety = varieties[j % n_varieties]
        tons = rng.uniform(*tons_range)
        slope = rng.random() < slope_fraction
        lots.append({"id": j, "variety": variety, "tons": float(tons), "slope": bool(slope)})

    mill_cap_series = np.full(horizon_days, mill_capacity, dtype=float)

    return HarvestInstance(lots, resources, mill_cap_series, horizon_days, oil_curves, varieties)


# --------------------------------------------------------------------------
# Instancia calibrada con datos reales (Fundo Tabalí, Coquimbo)
# --------------------------------------------------------------------------
# Estructura de "Hoja2" del archivo real (fila de encabezado en índice 3,
# 0-indexed, tras las filas en blanco): una fila por cuartel con su etapa,
# sección, año de plantación, variedad, número de cuartel, hectáreas, kilos
# de aceituna cosechados, litros de AOEV producidos, agua utilizada, costos
# e ingresos de la temporada registrada.
_TABALI_COLUMNS = [
    "_c0",
    "_c1",
    "_c2",
    "_c3",
    "etapa",
    "seccion",
    "anio",
    "variedad",
    "cuartel",
    "has",
    "kilos",
    "litros_aoev",
    "agua",
    "costos",
    "ingresos",
    "_n15",
    "_n16",
]


def read_tabali_lots(xlsx_path):
    """Lee 'Hoja2' del archivo real de Fundo Tabalí y devuelve una lista de
    dicts, un dict por cuartel, con sus datos tal como están en el archivo
    (kilos, litros AOEV, agua, costos, ingresos — todos por temporada, sin
    desglose diario)."""
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name="Hoja2", header=None)
    df = df.iloc[5:].reset_index(drop=True)  # salta filas en blanco + fila de encabezado
    df.columns = _TABALI_COLUMNS[: df.shape[1]]
    df = df.dropna(subset=["variedad", "kilos"])

    lots = []
    for _, row in df.iterrows():
        lots.append(
            {
                "cuartel": int(row["cuartel"]),
                "variety": str(row["variedad"]),
                "kilos": float(row["kilos"]),
                "litros_aoev": float(row["litros_aoev"]),
                "has": float(row["has"]),
                "agua": float(row["agua"]) if not pd.isna(row["agua"]) else None,
                "costos": float(row["costos"]) if not pd.isna(row["costos"]) else None,
                "ingresos": float(row["ingresos"]) if not pd.isna(row["ingresos"]) else None,
            }
        )
    return lots


def generate_tabali_instance(
    xlsx_path,
    horizon_days=45,
    mill_capacity=250.0,
    slope_fraction=0.25,
    oc_min_ratio=0.6,
    ripening_width_range=(4.0, 9.0),
    seed=42,
):
    """Instancia calibrada con datos reales de Fundo Tabalí (Coquimbo).

    Datos reales usados (de `xlsx_path`, hoja 'Hoja2'):
      - 28 cuarteles (lotes), cada uno con su variedad y kilos de aceituna
        realmente cosechados en la temporada registrada -> `tons` del lote.
      - Tasa de extracción de aceite (litros AOEV / kilos), que en el
        archivo resulta CONSTANTE ≈ 0.181 para las 6 variedades -> se usa
        como techo real de la curva de concentración de aceite (OC_max),
        en litros por tonelada, igual para todas las variedades (no hay
        evidencia en los datos de que varíe por variedad).

    Supuestos que el archivo NO permite calibrar (se mantienen del paper o
    se generan sintéticamente, documentados aquí):
      - Horizonte de planificación en días (`horizon_days`): el archivo es
        un acumulado de temporada, sin fechas; se usa el mismo horizonte de
        45 días del paper por comparabilidad.
      - Forma de la curva de maduración dentro de la temporada (día de
        pico `peak_day` y ancho `width` por variedad): sintética, con
        `peak_day` escalonado de forma determinista entre variedades para
        diferenciar el orden de cosecha (no hay dato diario real).
      - `oc_min_ratio`: fracción del rendimiento final que se asume al
        inicio de temporada (aceituna inmadura); 0.6 por defecto.
      - Accesibilidad del lote (`slope`, qué tan apto es para máquina):
        no está en los datos -> asignación aleatoria con `slope_fraction`,
        igual que en `generate_paper_instance`.
      - Equipos de cosecha y capacidad de la planta: no están en los datos
        de Tabalí -> se reutiliza la Tabla 1 del paper (HARVEST_MODES).

    Los campos económicos reales (`costos`, `ingresos`, `agua`, `has`) se
    conservan en `lot["costos"]`, `lot["ingresos"]`, etc. de cada lote,
    disponibles para quien quiera extender el objetivo f1 a rentabilidad
    en vez de producción física de aceite.
    """
    rng = np.random.default_rng(seed)
    raw_lots = read_tabali_lots(xlsx_path)

    varieties = sorted({lot["variety"] for lot in raw_lots})

    # Tasa de extracción real global (ponderada por kilos, por si difiriera
    # levemente entre filas por redondeo)
    total_kilos = sum(lot["kilos"] for lot in raw_lots)
    total_litros = sum(lot["litros_aoev"] for lot in raw_lots)
    extraction_ratio = total_litros / total_kilos  # L aceite / kg aceituna
    oc_max = extraction_ratio * 1000.0  # L aceite / tonelada aceituna
    oc_min = oc_min_ratio * oc_max

    # --- Recursos (Tabla 1 del paper; no viene en los datos de Tabalí) ---
    resources = []
    for kind, qty, caps in HARVEST_MODES:
        for i, cap in enumerate(caps):
            resources.append({"id": f"{kind}_{i + 1}", "kind": kind, "capacity": cap})

    # --- Curvas de concentración de aceite por variedad ---
    # Mismo techo real (oc_max) para todas las variedades (así lo indican
    # los datos); se escalona el día de pico determinísticamente para que
    # el modelo de cosecha siga teniendo variedades más/menos tempranas.
    oil_curves = {}
    for i, v in enumerate(varieties):
        peak_day = horizon_days * (0.3 + 0.5 * i / max(len(varieties) - 1, 1))
        width = rng.uniform(*ripening_width_range)
        oil_curves[v] = _oil_curve_factory(peak_day, oc_min, oc_max, width, rng)

    # --- Lotes (reales: variedad y toneladas; sintético: accesibilidad) ---
    lots = []
    for j, raw in enumerate(raw_lots):
        slope = rng.random() < slope_fraction
        lots.append(
            {
                "id": j,
                "cuartel": raw["cuartel"],
                "variety": raw["variety"],
                "tons": raw["kilos"] / 1000.0,
                "slope": bool(slope),
                "has": raw["has"],
                "agua": raw["agua"],
                "costos": raw["costos"],
                "ingresos": raw["ingresos"],
            }
        )

    mill_cap_series = np.full(horizon_days, mill_capacity, dtype=float)

    instance = HarvestInstance(lots, resources, mill_cap_series, horizon_days, oil_curves, varieties)
    instance.extraction_ratio = extraction_ratio  # metadato: L/kg real usado para calibrar
    return instance
