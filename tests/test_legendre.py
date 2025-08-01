"""Check that the Legendre functions actually work."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import IntegrationRule1D, compute_legendre


def test_orthonormal() -> None:
    """Check that the polynomals are orthogonal."""
    n = 5
    positions = np.linspace(-1, +1, 101)
    vals = compute_legendre(n, positions)
    assert vals is compute_legendre(n, positions, vals)
    rule = IntegrationRule1D(n + 1)

    vals = compute_legendre(n, rule.nodes)

    for i1 in range(n + 1):
        p1 = vals[i1, ...]
        for i2 in range(n + 1):
            p2 = vals[i2, ...]

            integral = np.sum(p1 * p2 * rule.weights)

            if i1 != i2:
                assert pytest.approx(integral) == 0
            else:
                assert pytest.approx(integral) == 2 / (2 * i1 + 1)


def compute_legendre_coeffs(
    order: int,
    rule: IntegrationRule1D,
    func: Callable[[npt.NDArray[np.floating]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Compute 1D Legendre coefficients."""
    leg1 = compute_legendre(order, rule.nodes)
    wleg1 = leg1 * rule.weights[None, ...]
    func_vals = np.asarray(func(rule.nodes))

    rleg = np.sum(
        func_vals[None, ...] * wleg1[:, ...],
        axis=-1,
    )
    n1 = np.arange(order + 1)
    norms1 = 2 / (2 * n1 + 1)
    rleg /= norms1

    return rleg


def legendre_integral_basis(
    order: int, positions: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """Compute integral Legendre basis."""
    base_order = order
    leg = compute_legendre(base_order, positions)
    k = 2 * np.arange(1, base_order) + 1
    base = leg[2:, ...] - leg[:-2, ...]
    return np.concatenate(
        (
            np.ones_like(positions)[None, ...],
            (positions)[None, ...],
            base / k[:, None],
        ),
        axis=0,
        dtype=np.float64,
    )[: order + 1, ...]


def matrix_legendre_coeffs_to_h1(
    coeffs: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Create a matrix that converts legendre coefficients into H^1 coefficients."""
    order = coeffs.size - 1
    assert order >= 0
    # Scale back up
    end = np.sum(coeffs)
    beginning = np.sum(coeffs * (-1) ** np.arange(order + 1))
    norms = 2 / (2 * np.arange(order + 1) + 1)
    c = coeffs * norms

    out = np.zeros_like(coeffs, np.float64)
    # First two coefficients are obtained from beginning and end values
    out[0] = (end + beginning) / 2
    if order > 0:
        out[1] = (end - beginning) / 2

    for n in range(2, order + 1):
        carry = 0
        m = n // 2
        for j in range(1, m + 1):
            carry += (2 * n - 4 * j + 1) * c[n - 2 * j]

        if n & 1:
            # Odd
            k = (end - beginning) - carry
        else:
            # Even
            k = (end + beginning) - carry

        # This here is just the L^2 norm of Legendre polynomial one order lower than
        # the integral basis with itself (hence ``n-1``).
        scale = (2 * (n - 1) + 1) / 2
        out[n] = scale * k

    return out


def test_integrated_1():
    """Check that conversion to integral basis works."""
    posx = np.linspace(-1, +1, 101)

    def test_function(x):
        return (
            1
            + x
            + 3 / 2 * (1 + x) * (1 - x)
            + 5 / 2 * (x + 1) * x * (x - 1)
            + 3.21 * x
            + x**3
            - 2 * x**4
            + x**5
        )

    order = 5
    rule = IntegrationRule1D(order + 1)
    coeffs = compute_legendre_coeffs(order, rule, test_function)

    realy = test_function(posx)
    new_coeffs = matrix_legendre_coeffs_to_h1(coeffs)
    val = legendre_integral_basis(order, posx)

    recon_2 = np.sum(val * new_coeffs[:, None], axis=0)
    assert pytest.approx(recon_2) == realy


def test_integrated_2():
    """Check that conversion to integral basis works."""
    posx = np.linspace(-1, +1, 101)

    def test_function(x):
        return 3 * np.sin(3 * x) - 2 * np.cos(x**2) - 2 / (x**2 + 1)

    order = 5
    rule = IntegrationRule1D(order + 1)
    coeffs = compute_legendre_coeffs(order, rule, test_function)

    lag = compute_legendre(order, posx)

    recon_1 = np.sum(lag * coeffs[:, None], axis=0)

    new_coeffs = matrix_legendre_coeffs_to_h1(coeffs)
    val = legendre_integral_basis(order, posx)

    recon_2 = np.sum(val * new_coeffs[:, None], axis=0)
    assert pytest.approx(recon_2) == recon_1


def test_basis_relation():
    """Check that gradients of the integral basis really reconstruct originals."""
    order = 5
    posx = np.linspace(-1, +1, 2001)
    val = legendre_integral_basis(order, posx)
    lag = compute_legendre(order, posx)
    for i in range(1, val.shape[0]):
        num_grad = (val[i, 1:] - val[i, :-1]) / (posx[1:] - posx[:-1])
        real_grad = (lag[i - 1, 1:] + lag[i - 1, :-1]) / 2

        assert np.abs(num_grad - real_grad).max() < 5e-6
