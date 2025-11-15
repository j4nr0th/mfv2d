"""Check that Gauss-Legendre-Lobatto nodes."""

import numpy as np
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    GLLCache,
    IntegrationRule1D,
    compute_gll,
)
from mfv2d.examples import unit_square_mesh
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import integrate_over_elements, jacobian, mesh_create, reconstruct
from mfv2d.refinement import compute_legendre_error_estimates
from mfv2d.solve_system import compute_element_dual, compute_element_primal_from_dual
from mfv2d.system import ElementFormSpecification
from scipy.integrate import dblquad


@pytest.mark.parametrize("n", (1, 3, 5, 6, 8, 10, 20))
def test_weight_sum(n: int):
    """Check that the weights sum to 2."""
    _, w = compute_gll(n)
    assert np.sum(w) == pytest.approx(2.0)


@pytest.mark.parametrize("n", (0, 1, 2, 3, 5, 6, 8, 10, 20))
def test_weight_integration(n: int):
    """Check that integrals hold."""
    x, w = compute_gll(n)
    for p in range(max(n + 1, 2 * n - 2)):
        # Exact up to and including 2 * n - 3
        num = np.sum(x**p * w)
        assert (1 / (p + 1) - ((-1) ** (p + 1)) / (p + 1)) == pytest.approx(num)


@pytest.mark.parametrize("n", (0, 1, 2, 3, 5, 6, 8, 10, 20))
def test_independence(n: int):
    """Check that calling it multiple times is consistent."""
    x, w = compute_gll(n)
    for p in range(10 * (n + 1)):
        # Exact up to and including 2 * n - 3
        x1, w1 = compute_gll(n)
        assert np.allclose(x, x1)
        assert np.allclose(w, w1)


@pytest.mark.parametrize(("nh", "nv", "m"), ((3, 3, 10), (4, 5, 3), (6, 6, 4)))
def test_mesh_2d_integrals(nh: int, nv: int, m: int) -> None:
    """Check mesh integration works as good as could be expected."""
    deformed_mesh = unit_square_mesh(
        nh,
        nv,
        1,
        deformation=lambda xi, eta: (
            xi + 0.1 * np.sin(np.pi * xi) * np.sin(np.pi * eta),
            eta - 0.1 * np.sin(np.pi * xi) * np.sin(np.pi * eta),
        ),
    )

    def test_function(x, y):
        """Function being tested."""
        return x**m + y**m + x ** (m - 1) * y + y ** (m - 1) * x

    integrals = integrate_over_elements(deformed_mesh, test_function, orders=m)
    assert pytest.approx(integrals.sum()) == dblquad(test_function, -1, +1, -1, +1)[0]


@pytest.mark.parametrize(("m",), ((10,), (15,), (20,)))
def test_gll_cache(m: int) -> None:
    """Check that GLL caching works as expeced."""
    cache_1 = GLLCache()
    cache_2 = GLLCache()

    E_1 = 1e-15
    E_2 = 1e-6

    rng = np.random.default_rng(0)
    orders = list(range(10, m + 10))

    for order in rng.permutation(orders):
        res_1 = compute_gll(order, tol=E_1, cache=cache_1)
        res_2 = compute_gll(order, tol=E_2, cache=cache_2)
        assert np.any(res_1[0] != res_2[0]) or np.any(res_1[1] != res_2[1])

    for order in rng.permutation(orders):
        compute_gll(order, tol=E_2, cache=cache_1)

    for order in rng.permutation(orders):
        res_1 = compute_gll(order, tol=E_1, cache=cache_1)
        res_2 = compute_gll(order, tol=E_2, cache=cache_2)
        assert np.all(res_1[0] == res_2[0])
        assert np.all(res_1[1] == res_2[1])


@pytest.mark.parametrize("m", range(1, 5))
def test_error_measure(m: int) -> None:
    """Check that the error measurement function works correctly."""
    rng = np.random.default_rng(seed=0)
    corners = np.array(
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)), np.float64
    ) + rng.uniform(-0.1, +0.1, (4, 2))
    form = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0)
    form_specs = ElementFormSpecification(form)
    element_space = ElementFemSpace2D(
        Basis2D(
            Basis1D(m, IntegrationRule1D(3 + m)),
            Basis1D(m, IntegrationRule1D(3 + m)),
        ),
        corners,
    )

    def test_function(x, y):
        return x**m + y**m + x ** (m - 1) * y + y ** (m - 1) * x

    def test_function_squared(x, y):
        return test_function(x, y) ** 2

    dofs = compute_element_primal_from_dual(
        form_specs,
        compute_element_dual(form_specs, [test_function], element_space),
        element_space,
    )

    mesh = mesh_create(m, corners, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))
    real_integral_2 = integrate_over_elements(mesh, test_function_squared, 2 * m)

    xi = element_space.basis_xi.rule.nodes[None, :]
    eta = element_space.basis_eta.rule.nodes[:, None]

    w = (
        element_space.basis_xi.rule.weights[None, :]
        * element_space.basis_eta.rule.weights[:, None]
    )
    (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
    det = j00 * j11 - j10 * j01

    reconstructed = reconstruct(element_space, form.order, dofs, xi, eta)

    zero_valued_form = np.zeros_like(reconstructed)
    l2_norm, _ = compute_legendre_error_estimates(
        m,
        m,
        np.astype(xi, np.float64, copy=False),
        np.astype(eta, np.float64, copy=False),
        w,
        det,
        zero_valued_form,
        reconstructed,
    )
    assert pytest.approx(l2_norm) == real_integral_2[0]


if __name__ == "__main__":
    test_gll_cache(10)
