"""Test functions which apply incidence matrices."""

import numpy as np
import pytest
from mfv2d.mimetic2d import (
    apply_e10,
    apply_e10_r,
    apply_e10_rt,
    apply_e10_t,
    apply_e21,
    apply_e21_r,
    apply_e21_rt,
    apply_e21_t,
    incidence_10,
    incidence_21,
)


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10(order: int):
    """Check that left multilpication be E10 works."""
    np.random.seed(0)
    in_mat = np.random.random_sample(((order + 1) ** 2, 3 * order))
    out_1 = incidence_10(order) @ in_mat
    out_2 = apply_e10(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21(order: int):
    """Check that left multilpication be E21 works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((2 * order * (order + 1), 4 * order + 4))
    out_1 = incidence_21(order) @ in_mat
    out_2 = apply_e21(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_t(order: int):
    """Check that left multilpication be E10^T works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((2 * order * (order + 1), 4 * order + 4))
    out_1 = incidence_10(order).T @ in_mat
    out_2 = apply_e10_t(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_t(order: int):
    """Check that left multilpication be E21^T works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((order**2, order + 3))
    out_1 = incidence_21(order).T @ in_mat
    out_2 = apply_e21_t(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_r(order: int):
    """Check that right multilpication be E10 works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((4 * order + 4, 2 * order * (order + 1)))
    out_1 = in_mat @ incidence_10(order)
    out_2 = apply_e10_r(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_r(order: int):
    """Check that right multilpication be E21 works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((order + 3, order**2))
    out_1 = in_mat @ incidence_21(order)
    out_2 = apply_e21_r(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e10_rt(order: int):
    """Check that right multilpication be E10^T works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((3 * order, (order + 1) ** 2))
    out_1 = in_mat @ incidence_10(order).T
    out_2 = apply_e10_rt(order, in_mat)
    assert pytest.approx(out_1) == out_2


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5))
def test_left_e21_rt(order: int):
    """Check that right multilpication be E21^T works."""
    np.random.seed(0)
    in_mat = np.random.random_sample((4 * order + 4, 2 * order * (order + 1)))
    out_1 = in_mat @ incidence_21(order).T
    out_2 = apply_e21_rt(order, in_mat)
    assert pytest.approx(out_1) == out_2
