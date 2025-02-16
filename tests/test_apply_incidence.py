"""Test functions which apply incidence matrices."""

import numpy as np
import pytest
from interplib.mimetic.mimetic2d import Element2D


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10(order: int):
    """Check that left multilpication be E10 works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample(((order + 1) ** 2, 3 * order))
    out_1 = element.incidence_10() @ in_mat
    out_2 = element.apply_e10(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21(order: int):
    """Check that left multilpication be E21 works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((2 * order * (order + 1), 4 * order + 4))
    out_1 = element.incidence_21() @ in_mat
    out_2 = element.apply_e21(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_t(order: int):
    """Check that left multilpication be E10^T works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((2 * order * (order + 1), 4 * order + 4))
    out_1 = element.incidence_10().T @ in_mat
    out_2 = element.apply_e10_t(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_t(order: int):
    """Check that left multilpication be E21^T works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((order**2, order + 3))
    out_1 = element.incidence_21().T @ in_mat
    out_2 = element.apply_e21_t(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_r(order: int):
    """Check that right multilpication be E10 works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((4 * order + 4, 2 * order * (order + 1)))
    out_1 = in_mat @ element.incidence_10()
    out_2 = element.apply_e10_r(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_r(order: int):
    """Check that right multilpication be E21 works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((order + 3, order**2))
    out_1 = in_mat @ element.incidence_21()
    out_2 = element.apply_e21_r(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_rt(order: int):
    """Check that right multilpication be E10^T works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((3 * order, (order + 1) ** 2))
    out_1 = in_mat @ element.incidence_10().T
    out_2 = element.apply_e10_rt(in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_rt(order: int):
    """Check that right multilpication be E21^T works."""
    np.random.seed(0)
    element = Element2D(order, (0, 0), (0, 0), (0, 0), (0, 0))
    in_mat = np.random.random_sample((4 * order + 4, 2 * order * (order + 1)))
    out_1 = in_mat @ element.incidence_21().T
    out_2 = element.apply_e21_rt(in_mat)
    assert pytest.approx(out_1) == out_2
