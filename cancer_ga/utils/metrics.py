"""
Métricas de calidad para optimización multiobjetivo (minimización de todos
los objetivos, igual que el resto del repo).

- `nondominated_mask`  : filtro de no-dominancia de Pareto.
- `hypervolume_2d`     : hipervolumen exacto para 2 objetivos (barrido O(n log n)).
- `ObjectiveNormalizer`: lleva los objetivos a [0, 1] con cotas FIJAS
  (ideal y nadir), de modo que el hipervolumen sea comparable entre
  generaciones y entre semillas dentro de un mismo problema.

Importante: el hipervolumen depende de la escala de cada objetivo, así que
sólo es comparable entre corridas que usen el MISMO normalizador. Para
problemas cuyo f2 cambia de definición (p.ej. distintos tipos de contrato)
el HV NO es comparable entre ellos.
"""

import numpy as np


def nondominated_mask(points, tol=1e-12):
    """Máscara booleana de los puntos no dominados (minimización).

    Un punto p es dominado si existe q con q <= p en todos los objetivos y
    q < p en al menos uno. Los duplicados exactos se conservan una sola vez.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = pts - pts[i]  # (n, m): q - p_i
        # j domina a i  <=>  pts[j] <= pts[i] en todo y < en alguno
        dominates_i = np.all(diff <= tol, axis=1) & np.any(diff < -tol, axis=1)
        if np.any(dominates_i):
            keep[i] = False
            continue
        # i domina a j  <=>  pts[i] <= pts[j] en todo y < en alguno
        dominated_by_i = np.all(diff >= -tol, axis=1) & np.any(diff > tol, axis=1)
        keep[dominated_by_i] = False

    # Eliminar duplicados exactos (se queda con el primero)
    seen = set()
    for i in range(n):
        if not keep[i]:
            continue
        key = tuple(np.round(pts[i], 12))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return keep


def hypervolume_2d(points, ref):
    """Hipervolumen de un conjunto de puntos 2D (minimización) respecto al
    punto de referencia `ref` (debe ser peor que los puntos en ambos ejes).

    Sólo contribuyen los puntos que dominan estrictamente a `ref`.
    Devuelve 0.0 si no hay ninguno.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    ref = np.asarray(ref, dtype=float)
    if len(pts) == 0:
        return 0.0

    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    pts = pts[nondominated_mask(pts)]
    pts = pts[np.argsort(pts[:, 0])]  # f1 ascendente => f2 descendente en un frente

    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in pts:
        hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(hv)


class ObjectiveNormalizer:
    """Normaliza objetivos a [0, 1] con cotas fijas: 0 = ideal, 1 = nadir.

    Con esta normalización el punto de referencia del hipervolumen es
    siempre (1, 1) y el HV queda acotado en [0, 1].
    """

    def __init__(self, ideal, nadir):
        self.ideal = np.asarray(ideal, dtype=float)
        self.nadir = np.asarray(nadir, dtype=float)
        span = self.nadir - self.ideal
        if np.any(span <= 0):
            raise ValueError("nadir debe ser estrictamente peor que ideal en todos los objetivos")
        self.span = span

    @classmethod
    def from_samples(cls, sample_objs, ideal, margin=0.1):
        """Construye el normalizador con el nadir estimado desde una muestra
        de soluciones (p.ej. aleatorias): peor valor observado más un margen
        relativo al rango [ideal, peor]."""
        samples = np.asarray(sample_objs, dtype=float)
        ideal = np.asarray(ideal, dtype=float)
        worst = samples.max(axis=0)
        nadir = worst + margin * np.maximum(worst - ideal, 1e-12)
        return cls(ideal, nadir)

    def __call__(self, objs):
        return (np.asarray(objs, dtype=float) - self.ideal) / self.span

    def hypervolume(self, objs):
        """Hipervolumen normalizado (ref = (1, 1)) de un conjunto de objetivos."""
        objs = np.asarray(objs, dtype=float).reshape(-1, len(self.ideal))
        if len(objs) == 0:
            return 0.0
        return hypervolume_2d(self(objs), np.ones(2))

    def to_dict(self):
        return {"ideal": self.ideal.tolist(), "nadir": self.nadir.tolist()}
