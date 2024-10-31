"""Test Spline functions."""

import numpy as np
import pytest
from interplib import SplineBC, nodal_interpolating_splinei


@pytest.mark.parametrize("n,order", ((2, 1), (4, 3), (10, 5), (100, 7)))
def test_spline_at_nodes(n: int, order: int):
    """Check nodal Spline1D returns same values at all nodes."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    spline = nodal_interpolating_splinei(order, x)
    assert pytest.approx(x) == spline(np.arange(n))


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_linear(n: int):
    """Check linear nodal Spline1D is exact for linear functions."""
    np.random.seed(0)
    i = np.arange(n)
    a, b = np.random.random_sample((2,))
    y = a * i + b
    spline = nodal_interpolating_splinei(1, y)
    assert spline.derivative(i) == pytest.approx(a)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_quadratic(n: int):
    """Check cubic nodal Spline1D is exact for quadratic functions."""
    np.random.seed(0)
    i = np.arange(n)
    a, b, c = np.random.random_sample((3,))
    y = a * i**2 + b * i + c
    spline = nodal_interpolating_splinei(
        3,
        y,
        [SplineBC([0, 1, 0], 2 * a * i[0] + b)],
        [SplineBC([0, 1, 0], 2 * a * i[-1] + b)],
    )
    assert spline.derivative(i) == pytest.approx(2 * a * i + b)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_cubic(n: int):
    """Check cubic nodal Spline1D is exact for cubic functions."""
    np.random.seed(0)
    i = np.arange(n)
    a, b, c, d = np.random.random_sample((4,))
    y = a * i**3 + b * i**2 + c * i + d
    spline = nodal_interpolating_splinei(
        3,
        y,
        [SplineBC([0, 1, 0], 3 * a * i[0] ** 2 + 2 * b * i[0] + c)],
        [SplineBC([0, 1, 0], 3 * a * i[-1] ** 2 + 2 * b * i[-1] + c)],
    )
    assert spline.derivative(i) == pytest.approx(3 * a * i**2 + 2 * b * i + c)
