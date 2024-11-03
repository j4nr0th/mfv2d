"""Test C-type spline."""

import numpy as np
import pytest
from interplib._interp import Polynomial1D, Spline1D


def test_properties():
    """Check properties return same as they're given."""
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


@pytest.mark.parametrize("n,m,evaln", ((3, 4, 10), (5, 44, 1000)))
def test_interpolation(n: int, m: int, evaln: int):
    """Check that interpolation with coefficients gives right results."""
    for power in range(m):
        test_nodes = np.cos(np.linspace(np.pi, 2 * np.pi, n))
        test_coefficients = np.zeros((n - 1, m + 1))
        test_coefficients[:, power] = 1.0
        spl = Spline1D(test_nodes, test_coefficients)
        eval_nodes = np.linspace(np.min(test_nodes), np.max(test_nodes), evaln)
        y = spl(eval_nodes)
        iright = np.argmax(eval_nodes[:, None] < test_nodes[None, :], axis=1)
        frac = eval_nodes - test_nodes[iright - 1]
        expected = frac**power
        assert (
            pytest.approx(y[:-1]) == expected[:-1]
        )  # Last element can't be computed fast with numpy, so IDK


@pytest.mark.parametrize("n,m,ntest", ((3, 4, 100), (10, 20, 1000)))
def test_antiderivative(n: int, m: int, ntest: int):
    """Check antiderivative of a spline matches with the polynomial it is from."""
    np.random.seed(912)
    p = Polynomial1D(np.random.random_sample(n))
    # Nodes must start at zero, since Polynomial1D antiderivative has a zero there,
    # while a spline has it at its first node.
    nodes = np.pad(np.sort(np.random.random_sample(m - 1)), (1, 0))
    # Make spline by just shifting the polynomial to the nodes.
    spl = Spline1D(nodes, tuple(p.offset_by(nodes[i]).coefficients for i in range(m - 1)))
    xtest = np.linspace(np.min(nodes), np.max(nodes), ntest)
    assert pytest.approx(spl.antiderivative(xtest)) == p.antiderivative(xtest)
