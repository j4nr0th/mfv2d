"""Check that the Legendre functions actually work."""

import numpy as np
import pytest
from mfv2d._mfv2d import IntegrationRule1D, compute_legendre
from mfv2d.mimetic2d import jacobian
from mfv2d.refinement import compute_legendre_coefficients


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


@pytest.mark.parametrize(
    ("nh", "nv", "k"), ((1, 1, 3), (2, 2, 3), (4, 3, 2), (3, 4, 2), (5, 1, 4))
)
def test_orthonormal_2d(nh: int, nv: int, k: int) -> None:
    """Check that the 2D polynomals are orthogonal."""
    rng = np.random.default_rng(seed=0)

    # Get the integration set up
    corners = np.array(
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)), np.float64
    ) + rng.uniform(-0.1, +0.1, (4, 2))
    rule_h = IntegrationRule1D(nh + k)
    rule_v = IntegrationRule1D(nv + k)
    xi = rule_h.nodes[None, :]
    eta = rule_v.nodes[:, None]
    (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
    det = j00 * j11 - j01 * j10
    weights = rule_h.weights[None, :] * rule_v.weights[:, None]

    # Prepare the vertical and horizontal basis
    legh = compute_legendre(nh, xi)
    legv = compute_legendre(nv, eta)

    # Create the 2D basis
    basis = (
        np.array(
            [
                legh[ih, ...] * legv[iv, ...]
                for iv in range(nv + 1)
                for ih in range(nh + 1)
            ]
        )
        / np.sqrt(det)[None, ...]
    )

    for ib in range(basis.shape[0]):
        b1 = basis[ib]
        for jb in range(basis.shape[0]):
            b2 = basis[jb]

            val = np.sum(weights * det * b1 * b2)

            if ib != jb:
                assert pytest.approx(val) == 0

            else:
                kh = ib % (nh + 1)
                kv = ib // (nh + 1)
                assert pytest.approx(val) == 4 / (2 * kh + 1) / (2 * kv + 1)


@pytest.mark.parametrize(
    ("order_h", "order_v", "k"), ((i, j, 1) for i in range(5) for j in range(5))
)
def test_reconstruction(order_h: int, order_v: int, k: int) -> None:
    """Check that polynomials of the specified order can be reconstructed."""
    rng = np.random.default_rng(seed=0)

    # Get the integration set up
    corners = np.array(
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)), np.float64
    ) + rng.uniform(-0.1, +0.1, (4, 2))
    rule_h = IntegrationRule1D(order_h + k)
    rule_v = IntegrationRule1D(order_v + k)
    xi = rule_h.nodes[None, :]
    eta = rule_v.nodes[:, None]
    (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
    det = j00 * j11 - j01 * j10
    weights = rule_h.weights[None, :] * rule_v.weights[:, None]

    # Generate the random coefficients
    coeffs = rng.uniform(-1, +1, (order_v + 1, order_h + 1))

    # Prepare the vertical and horizontal basis
    legh = compute_legendre(order_h, xi)
    legv = compute_legendre(order_v, eta)

    # Generate the reconstruction
    reconstructed = np.zeros_like(weights)
    for ov in range(order_v + 1):
        bv = legv[ov]
        for oh in range(order_h + 1):
            bh = legh[oh]
            c = coeffs[ov, oh]
            reconstructed += c * bh * bv / np.sqrt(det)

    # Compute the coefficients with the function
    recomputed = compute_legendre_coefficients(
        order_h,
        order_v,
        np.astype(xi, np.float64, copy=False),
        np.astype(eta, np.float64, copy=False),
        reconstructed * weights * det,
        det,
    )
    assert pytest.approx(coeffs) == recomputed


if __name__ == "__main__":
    for i in range(3, 5):
        for j in range(3, 5):
            test_reconstruction(i, j, k=1)

    for nh, nv, k in ((1, 1, 3), (2, 2, 3), (4, 3, 2), (3, 4, 2), (5, 1, 4)):
        test_orthonormal_2d(nh, nv, k)
