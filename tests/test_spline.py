"""Test Spline functions."""

import numpy as np
import pytest
from interplib import (
    Polynomial1D,
    SplineBC,
    element_interpolating_splinei,
    nodal_interpolating_splinei,
)
from scipy.integrate import quad


@pytest.mark.parametrize("n,order", ((2, 1), (4, 3), (10, 5), (100, 7)))
def test_spline_at_nodes(n: int, order: int):
    """Check nodal Spline1Di returns same values at all nodes."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    spline = nodal_interpolating_splinei(order, x)
    assert pytest.approx(x) == spline(np.arange(n))


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_linear(n: int):
    """Check linear nodal Spline1Di is exact for linear functions."""
    np.random.seed(0)
    i = np.arange(n)
    a, b = np.random.random_sample((2,))
    y = a * i + b
    spline = nodal_interpolating_splinei(1, y)
    assert spline.derivative(i) == pytest.approx(a)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_quadratic(n: int):
    """Check cubic nodal Spline1Di is exact for quadratic functions."""
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


@pytest.mark.parametrize("n, ntest", ((2, 10), (4, 20), (10, 1000)))
def test_spline_element(n: int, ntest: int):
    """Check quartic element Spline1Di is exact for cubic functions."""
    np.random.seed(0)
    i = np.arange(n)
    p = Polynomial1D(np.random.random_sample((4,)))
    antiderivative = p.antiderivative
    derivative = p.derivative
    spline = element_interpolating_splinei(
        4,
        antiderivative(i[1:]) - antiderivative(i[:-1]),
        [
            SplineBC([1, 0, 0, 0], float(p(i[0]))),
            SplineBC([0, 1, 0, 0], float(derivative(i[0]))),
        ],
        [
            SplineBC([0, 1, 0, 0], float(derivative(i[-1]))),
            SplineBC([1, 0, 0, 0], float(p(i[-1]))),
        ],
    )
    xtest = np.linspace(i[0], i[-1], ntest)
    assert spline(xtest) == pytest.approx(p(xtest))


@pytest.mark.parametrize("n,order", ((2, 0), (4, 2), (10, 6), (100, 10)))
def test_spline_averages(n: int, order: int):
    """Check nodal Spline1Di returns same averages on all elements."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    spline = element_interpolating_splinei(order, x)
    for i in range(n):
        q, _ = quad(spline, i, i + 1)
        assert (q == pytest.approx(x[i])) or (q == pytest.approx(x[i]))
