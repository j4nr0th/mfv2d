"""Test C-type spline."""

import numpy as np
import pytest
from interplib._interp import Polynomial1D, Spline1D
from interplib.splines import (
    SplineBC,
    element_interpolating_spline,
    nodal_interpolating_spline,
)
from scipy.integrate import quad


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


@pytest.mark.parametrize("n,order", ((2, 1), (4, 3), (10, 5), (100, 7)))
def test_spline_at_nodes(n: int, order: int):
    """Check nodal Spline1D returns same values at all nodes."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    nodes = np.sort(np.random.random_sample((n,)))
    spline = nodal_interpolating_spline(order, x, nodes)
    assert pytest.approx(x) == spline(nodes)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_linear(n: int):
    """Check linear nodal Spline1D is exact for linear functions."""
    np.random.seed(0)
    nodes = np.sort(np.random.random_sample((n,)))
    a, b = np.random.random_sample((2,))
    y = a * nodes + b
    spline = nodal_interpolating_spline(1, y, nodes)
    assert spline.derivative(nodes) == pytest.approx(a)


@pytest.mark.parametrize("n", (2, 4, 10, 100))
def test_spline_quadratic(n: int):
    """Check cubic nodal Spline1D is exact for quadratic functions."""
    np.random.seed(0)
    nodes = np.sort(np.random.random_sample((n,)))
    a, b, c = np.random.random_sample((3,))
    y = a * nodes**2 + b * nodes + c
    spline = nodal_interpolating_spline(
        3,
        y,
        nodes,
        [SplineBC([0, 1, 0], 2 * a * nodes[0] + b)],
        [SplineBC([0, 1, 0], 2 * a * nodes[-1] + b)],
    )
    assert spline.derivative(nodes) == pytest.approx(2 * a * nodes + b)


@pytest.mark.parametrize("n,ntest", ((2, 10), (4, 20), (10, 100), (100, 1000)))
def test_spline_quadratic_antiderivative(n: int, ntest: int):
    """Check cubic nodal Spline1D antiderivative is exact for quadratic functions."""
    np.random.seed(0)
    nodes = np.pad(np.sort(np.random.random_sample((n - 1,))), (1, 0))
    a, b, c = np.random.random_sample((3,))
    y = a * nodes**2 + b * nodes + c
    spline = nodal_interpolating_spline(
        3,
        y,
        nodes,
        [SplineBC([0, 1, 0], 2 * a * nodes[0] + b)],
        [SplineBC([0, 1, 0], 2 * a * nodes[-1] + b)],
    )
    itest = np.linspace(nodes[0], nodes[-1], ntest)
    assert spline.antiderivative(itest) == pytest.approx(
        a / 3 * itest**3 + b / 2 * itest**2 + c * itest
    )


@pytest.mark.parametrize("n, ntest", ((2, 10), (4, 20), (10, 1000)))
def test_spline_element(n: int, ntest: int):
    """Check quartic element Spline1D is exact for cubic functions."""
    np.random.seed(0)
    nodes = np.sort(np.random.random_sample((n + 1,)))
    p = Polynomial1D(np.random.random_sample((4,)))
    antiderivative = p.antiderivative
    derivative = p.derivative
    spline = element_interpolating_spline(
        4,
        antiderivative(nodes[1:]) - antiderivative(nodes[:-1]),
        nodes,
        [
            SplineBC([1, 0, 0, 0], float(p(nodes[0]))),
            SplineBC([0, 1, 0, 0], float(derivative(nodes[0]))),
        ],
        [
            SplineBC([0, 1, 0, 0], float(derivative(nodes[-1]))),
            SplineBC([1, 0, 0, 0], float(p(nodes[-1]))),
        ],
    )
    xtest = np.linspace(nodes[0], nodes[-1], ntest)
    assert spline(xtest) == pytest.approx(p(xtest))


@pytest.mark.parametrize("n,order", ((2, 0), (4, 2), (10, 6)))
def test_spline_averages(n: int, order: int):
    """Check nodal Spline1D returns same averages on all elements."""
    np.random.seed(0)
    x = np.random.random_sample((n,))
    nodes = np.sort(np.random.random_sample((n + 1,)))
    spline = element_interpolating_spline(order, x, nodes)
    for i in range(n):
        q, _ = quad(spline, nodes[i], nodes[i + 1])
        assert (q == pytest.approx(x[i])) or (q == pytest.approx(x[i]))
