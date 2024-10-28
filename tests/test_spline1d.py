"""Test C-type spline."""

import numpy as np
import pytest
from interplib._interp import Spline1D


def test_properties():
    """Check properties reutrn same as they're given."""
    test_nodes = np.array([0, 0.5, 0.7, 0.9, 1.0], np.float64)
    test_coefficients = np.array(
        [
            [0, 1, 2],
            [2, 4, 8],
            [-1, -2, -3],
            [-2, -2, -2],
        ],
        np.float64,
    )
    spl = Spline1D(test_nodes, test_coefficients)
    assert np.all(spl.nodes == test_nodes)
    assert np.all(spl.coefficients == test_coefficients)


def test_interpolation():
    """Check that interpolation with coefficients gives right results."""
    n = 10
    m = 3
    for power in range(m):
        test_nodes = np.arange(n)
        test_coefficients = np.zeros((n - 1, m + 1))
        test_coefficients[:, power] = 1.0
        spl = Spline1D(test_nodes, test_coefficients)
        eval_nodes = np.linspace(0, n - 1, 10 * n + 1)
        y = spl(eval_nodes)
        frac, whole = np.modf(eval_nodes)
        expected = frac**power
        # expected[expected == 0] = 1.0
        expected[-1] = 1
        assert pytest.approx(y) == expected
