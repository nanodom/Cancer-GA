import numpy as np
import pytest

from cancer_ga.utils.metrics import nondominated_mask, hypervolume_2d, ObjectiveNormalizer


def test_nondominated_mask_basic():
    pts = [[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]]
    mask = nondominated_mask(pts)
    # (3,3) lo domina (2,2); (5,5) lo domina todo
    assert mask.tolist() == [True, True, True, False, False]


def test_nondominated_mask_removes_duplicates():
    pts = [[1, 1], [1, 1], [2, 0.5]]
    mask = nondominated_mask(pts)
    assert mask.sum() == 2


def test_hypervolume_single_point():
    # Rectángulo entre (1,1) y la referencia (3,4): 2 * 3 = 6
    assert hypervolume_2d([[1, 1]], ref=[3, 4]) == pytest.approx(6.0)


def test_hypervolume_staircase_hand_computed():
    # Frente {(1,3), (2,2), (3,1)} con ref (4,4):
    #   (4-1)*(4-3) + (4-2)*(3-2) + (4-3)*(2-1) = 3 + 2 + 1 = 6
    pts = [[1, 3], [2, 2], [3, 1]]
    assert hypervolume_2d(pts, ref=[4, 4]) == pytest.approx(6.0)


def test_hypervolume_ignores_dominated_points():
    base = hypervolume_2d([[1, 3], [2, 2], [3, 1]], ref=[4, 4])
    with_dominated = hypervolume_2d([[1, 3], [2, 2], [3, 1], [3.5, 3.5]], ref=[4, 4])
    assert with_dominated == pytest.approx(base)


def test_hypervolume_points_beyond_reference_contribute_nothing():
    assert hypervolume_2d([[5, 5]], ref=[4, 4]) == 0.0
    assert hypervolume_2d([], ref=[4, 4]) == 0.0


def test_hypervolume_monotone_when_adding_nondominated_point():
    a = hypervolume_2d([[1, 3], [3, 1]], ref=[4, 4])
    b = hypervolume_2d([[1, 3], [2, 2], [3, 1]], ref=[4, 4])
    assert b > a


def test_normalizer_maps_ideal_to_zero_and_nadir_to_one():
    norm = ObjectiveNormalizer(ideal=[-10, 0], nadir=[0, 100])
    assert np.allclose(norm([-10, 0]), [0, 0])
    assert np.allclose(norm([0, 100]), [1, 1])
    # HV normalizado del punto ideal es el cuadrado unitario completo
    assert norm.hypervolume([[-10, 0]]) == pytest.approx(1.0)


def test_normalizer_from_samples_nadir_is_worse_than_all_samples():
    samples = np.array([[-5.0, 10.0], [-8.0, 40.0], [-2.0, 25.0]])
    norm = ObjectiveNormalizer.from_samples(samples, ideal=[-10.0, 0.0], margin=0.1)
    assert np.all(norm.nadir > samples.max(axis=0))


def test_normalizer_rejects_degenerate_bounds():
    with pytest.raises(ValueError):
        ObjectiveNormalizer(ideal=[0, 0], nadir=[0, 1])
