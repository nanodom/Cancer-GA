from cancer_ga.utils.metrics import ObjectiveNormalizer
from cancer_ga.utils.spacing import spacing_to_front

NORM = ObjectiveNormalizer(ideal=[-10, 0], nadir=[0, 10])


def _pts(*pairs):
    return [{"oil": -o, "pvi": p} for o, p in pairs]


def test_spacing_needs_at_least_three_points():
    """Con 2 puntos el espaciado es 0 por construcción: no debe reportarse como si fuera uniformidad."""
    for pts in ([], _pts((5, 1)), _pts((5, 1), (3, 4))):
        assert spacing_to_front(pts, NORM)["spacing"] is None


def test_spacing_zero_for_uniform_front_and_positive_otherwise():
    uniform = _pts((8, 2), (6, 4), (4, 6), (2, 8))
    assert spacing_to_front(uniform, NORM)["spacing"] == 0.0 or spacing_to_front(uniform, NORM)["spacing"] < 1e-9
    uneven = _pts((9, 1), (8.5, 1.5), (3, 8), (1, 9))
    assert spacing_to_front(uneven, NORM)["spacing"] > 0.05
