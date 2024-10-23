"""Test very basic functionality."""
import numpy as np
import pytest
from interplib import _interp


def test_internal_import():
    assert _interp.test() == "Test successful!\n"

@pytest.mark.parametrize("n", np.logspace(1, 6, base=2, num=10, dtype=int))
def test_lagrange1d(n):
    """Check random functions at random nodes as being close enough."""
    np.random.seed(0)
    xvals = np.random.random_sample(n)
    yvals = np.random.random_sample(n)
    xvals = np.array(xvals, dtype=np.float64)
    yvals = np.array(yvals, dtype=np.float64)

    interpolated = _interp.lagrange1d(xvals, xvals) @ yvals
    assert interpolated == pytest.approx(yvals)



@pytest.mark.parametrize("n", np.arange(2, 6))
def test_lagrange1d_2(n):
    """Check functions it should get exactly at different nodes"""
    np.random.seed(0)
    xvals = np.random.random_sample(n + 1)
    xvals = np.array(xvals, dtype=np.float64)
    yvals = xvals ** n
    xeval = np.array(np.random.random_sample(n + 1), dtype=np.float64)


    interpolated = _interp.dlagrange1d(xeval, xvals) @ yvals
    assert interpolated == pytest.approx(xeval ** (n - 1) * n)






