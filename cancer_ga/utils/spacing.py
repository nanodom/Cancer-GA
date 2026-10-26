"""Spacing-to-front (Schott, 1995) sobre el frente (oil, PVI) normalizado."""

from __future__ import annotations
import numpy as np

MIN_POINTS = 3  # mínimo de puntos para que el espaciado de Schott sea informativo


def _ideal_nadir(normalizer):
    if hasattr(normalizer, "ideal"):
        return np.asarray(normalizer.ideal, dtype=float), np.asarray(normalizer.nadir, dtype=float)
    return np.asarray(normalizer["ideal"], dtype=float), np.asarray(normalizer["nadir"], dtype=float)


def spacing_to_front(points: list[dict], normalizer) -> dict:
    n = len(points)
    if n == 0:
        return {"spacing": None, "n_points": 0, "reason": "empty_front"}
    if n < MIN_POINTS:
        # Con 2 puntos las dos distancias al vecino más cercano son iguales, así que el
        # espaciado vale 0 POR CONSTRUCCIÓN (no por uniformidad). Con n<3 no es informativo y
        # sesgaría la comparación a favor del método con frentes más pobres.
        return {"spacing": None, "n_points": n, "reason": "too_few_points"}

    ideal, nadir = _ideal_nadir(normalizer)
    span = nadir - ideal
    if np.any(span <= 0):
        return {"spacing": None, "n_points": n, "reason": "invalid_normalizer"}

    pts = np.array([[(p["oil"] - ideal[0]) / span[0], (p["pvi"] - ideal[1]) / span[1]] for p in points])

    d = np.empty(n)
    for i in range(n):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = np.inf
        d[i] = dists.min()

    d_bar = d.mean()
    if d_bar < 1e-12:
        return {"spacing": 0.0, "n_points": n, "reason": "degenerate_collapse"}

    sp = float(np.sqrt(np.sum((d - d_bar) ** 2) / (n - 1)))
    return {"spacing": sp, "n_points": n, "reason": None}
