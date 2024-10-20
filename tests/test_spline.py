"""Test Spline functions."""

import numpy as np
import pytest
from interplib.hermite import HermiteSpline, SplineBC


def test_bcs():
    """Check SplineBCs give right tuple."""
    np.random.seed(0)
    x = np.random.random_sample((3,))
    assert np.all(SplineBC(*x).as_tuple() == x)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_at_nodes(n: int):
    """Check HermiteSpline returns same values at all nodes."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    spline = HermiteSpline(x)
    assert pytest.approx(x) == spline(np.arange(n))


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_linear(n: int):
    """Check HermiteSpline is exact for linear funcitons."""
    np.random.seed(0)
    i = np.arange(n)
    a, b = np.random.random_sample((2,))
    y = a * i + b
    spline = HermiteSpline(y)
    assert spline.derivatives == pytest.approx(a)

@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_quadratic(n: int):
    """Check HermiteSpline is exact for quadratic funcitons."""
    np.random.seed(0)
    i = np.arange(n)
    a, b, c = np.random.random_sample((3,))
    y = a * i ** 2 + b * i + c
    spline = HermiteSpline(y, SplineBC(1, 0, b), SplineBC(1, 0, 2 * a * i[-1] + b))
    assert spline.derivatives == pytest.approx(2 * a * i + b)

@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_cubic(n: int):
    """Check HermiteSpline is exact for cubic funcitons."""
    np.random.seed(0)
    i = np.arange(n)
    a, b, c, d = np.random.random_sample((4,))
    y = a * i ** 3 + b * i ** 2 + c * i + d
    spline = HermiteSpline(y, SplineBC(0, 1, 2 * b + 6 * a * i[0]), SplineBC(0, 1, 2 * b + 6 * a * i[-1]))
    assert spline.derivatives == pytest.approx(3 * a * i ** 2 + 2 * b * i + c)

