"""Check that projections work correctly."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    IntegrationRule1D,
    compute_element_projector,
)
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import (
    bilinear_interpolate,
    element_dual_dofs,
    element_primal_dofs,
    reconstruct,
)
from mfv2d.system import ElementFormSpecification


def test_reconstruction_nodal() -> None:
    """Check nodal reconstruction is exact at a high enough order."""
    N = 6

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    corners = np.array([(-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1)], np.float64)

    int_rule = IntegrationRule1D(N + 2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    element_space = ElementFemSpace2D(basis_2d, corners)

    dual = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, element_space, test_function
    )
    test_v = np.linspace(-1, +1, 21)
    recon = reconstruct(
        element_space,
        UnknownFormOrder.FORM_ORDER_0,
        dual,
        test_v[None, :],
        test_v[:, None],
    )

    real = test_function(
        bilinear_interpolate(corners[:, 0], test_v[None, :], test_v[:, None]),
        bilinear_interpolate(corners[:, 1], test_v[None, :], test_v[:, None]),
    )
    assert pytest.approx(recon) == real


def test_reconstruction_edge() -> None:
    """Check edge reconstruction is exact at a high enough order."""
    N = 6

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return np.stack((x**3 + 2 * y - x * y, y**3 - 2 * x + 2 * x * y), axis=-1)

    corners = np.array([(-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1)], np.float64)

    int_rule = IntegrationRule1D(N + 2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    element_space = ElementFemSpace2D(basis_2d, corners)

    dual = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, element_space, test_function
    )
    test_v = np.linspace(-1, +1, 21)
    recon = reconstruct(
        element_space,
        UnknownFormOrder.FORM_ORDER_1,
        dual,
        test_v[None, :],
        test_v[:, None],
    )

    real = test_function(
        bilinear_interpolate(corners[:, 0], test_v[None, :], test_v[:, None]),
        bilinear_interpolate(corners[:, 1], test_v[None, :], test_v[:, None]),
    )
    assert pytest.approx(recon) == real


def test_reconstruction_surf() -> None:
    """Check nodal reconstruction is exact at a high enough order."""
    N = 6

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    corners = np.array([(-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1)], np.float64)

    int_rule = IntegrationRule1D(N + 2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    element_space = ElementFemSpace2D(basis_2d, corners)

    dual = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, element_space, test_function
    )
    test_v = np.linspace(-1, +1, 21)
    recon = reconstruct(
        element_space,
        UnknownFormOrder.FORM_ORDER_2,
        dual,
        test_v[None, :],
        test_v[:, None],
    )

    real = test_function(
        bilinear_interpolate(corners[:, 0], test_v[None, :], test_v[:, None]),
        bilinear_interpolate(corners[:, 1], test_v[None, :], test_v[:, None]),
    )
    assert pytest.approx(recon) == real


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_self_check(n: int) -> None:
    """Check projection to same space is identity."""
    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array([(-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1)], np.float64)
    out_space = ElementFemSpace2D(basis_2d, corners)

    for order in UnknownFormOrder:
        specs = ElementFormSpecification(KFormUnknown("test", order))
        (mat,) = compute_element_projector(specs, out_space.basis_2d, out_space)

        assert pytest.approx(mat) == np.eye(order.full_unknown_count(n, n))


def test_projection_contained_node() -> None:
    """Check that projection to higher order works.

    An L2 projection of a function that is fully contained in the lower-order
    space should be exactly the same when projected to the higher space as when
    the L2 projection is done directly on the higher space.
    """
    int_rule = IntegrationRule1D(10)
    basis_1d_low = Basis1D(3, int_rule)
    basis_2d_low = Basis2D(basis_1d_low, basis_1d_low)

    basis_1d_high = Basis1D(7, int_rule)
    basis_2d_high = Basis2D(basis_1d_high, basis_1d_high)

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function, which is 2nd order."""
        return x * y - 2 * x + y - 2

    corners = np.array([(-2, -1.5), (+0.9, -1), (+1, +1), (-1.5, +1.1)], np.float64)
    element_high = ElementFemSpace2D(basis_2d_high, corners)
    element_low = ElementFemSpace2D(basis_2d_low, corners)

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, element_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, element_high, test_function
    )
    specs = ElementFormSpecification(KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0))
    projector = compute_element_projector(specs, basis_2d_low, element_high)

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(specs, basis_2d_high, element_low)
    projected = (reverse_projector @ high_dofs).flatten()
    assert pytest.approx(projected) == low_dofs


def test_projection_contained_edge() -> None:
    """Check that projection to higher order works.

    An L2 projection of a function that is fully contained in the lower-order
    space should be exactly the same when projected to the higher space as when
    the L2 projection is done directly on the higher space.
    """
    int_rule = IntegrationRule1D(10)
    basis_1d_low = Basis1D(4, int_rule)
    basis_2d_low = Basis2D(basis_1d_low, basis_1d_low)

    basis_1d_high = Basis1D(7, int_rule)
    basis_2d_high = Basis2D(basis_1d_high, basis_1d_high)

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function, which is 2nd order."""
        return np.stack((x * y - 2 * x + y - 2, 3 * y - 2 * x * y + 1), axis=-1)

    corners = np.array([(-2, -1.5), (+0.9, -1), (+1, +1), (-1.5, +1.1)], np.float64)
    element_high = ElementFemSpace2D(basis_2d_high, corners)
    element_low = ElementFemSpace2D(basis_2d_low, corners)

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, element_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, element_high, test_function
    )
    specs = ElementFormSpecification(KFormUnknown("u", UnknownFormOrder.FORM_ORDER_1))
    projector = compute_element_projector(specs, basis_2d_low, element_high)

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(specs, basis_2d_high, element_low)
    projected = (reverse_projector @ high_dofs).flatten()
    assert pytest.approx(projected) == low_dofs


def test_projection_contained_surf() -> None:
    """Check that projection to higher order works.

    An L2 projection of a function that is fully contained in the lower-order
    space should be exactly the same when projected to the higher space as when
    the L2 projection is done directly on the higher space.
    """
    int_rule = IntegrationRule1D(10)
    basis_1d_low = Basis1D(4, int_rule)
    basis_2d_low = Basis2D(basis_1d_low, basis_1d_low)

    basis_1d_high = Basis1D(7, int_rule)
    basis_2d_high = Basis2D(basis_1d_high, basis_1d_high)

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function, which is 2nd order."""
        return x * y - 2 * x + y - 2

    corners = np.array([(-2, -1.5), (+0.9, -1), (+1, +1), (-1.5, +1.1)], np.float64)
    element_high = ElementFemSpace2D(basis_2d_high, corners)
    element_low = ElementFemSpace2D(basis_2d_low, corners)

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, element_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, element_high, test_function
    )
    specs = ElementFormSpecification(KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2))
    projector = compute_element_projector(specs, basis_2d_low, element_high)

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(specs, basis_2d_high, element_low)
    projected = (reverse_projector @ high_dofs).flatten()
    assert pytest.approx(projected) == low_dofs


_TEST_ORDERS = (
    (1, 1),
    (3, 3),
    (6, 6),
    (1, 3),
    (3, 6),
    (5, 6),
)


@pytest.mark.parametrize(("n1", "n2"), _TEST_ORDERS)
def test_projector_contained_nodes(n1: int, n2: int) -> None:
    """Check that the projector transpose behaves as expected."""
    assert n1 <= n2
    rule = IntegrationRule1D(max(n1, n2) + 4)

    basis_low = Basis1D(n1, rule)
    basis_high = Basis1D(n2, rule)

    corners = np.array([(-2, -1.5), (+0.9, -1), (+1, +1), (-1.5, +1.1)], np.float64)

    space_low = ElementFemSpace2D(Basis2D(basis_low, basis_low), corners)
    space_high = ElementFemSpace2D(Basis2D(basis_high, basis_high), corners)

    def test_low_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Function that should be resolvable."""
        return x**n1 + y**n1 + x * y ** (n1 - 1) + x ** (n1 - 1) * y

    primal_low = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, space_low, test_low_function
    )
    primal_high = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, space_high, test_low_function
    )

    dual_high = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_0, space_high, test_low_function
    )

    specs = ElementFormSpecification(KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0))
    (projector_pl_ph,) = compute_element_projector(
        specs, space_low.basis_2d, space_high, dual=False
    )

    specs = ElementFormSpecification(KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0))
    (projector_pl_dh,) = compute_element_projector(
        specs, space_low.basis_2d, space_high, dual=True
    )

    assert pytest.approx(primal_high) == projector_pl_ph @ primal_low
    assert pytest.approx(dual_high) == projector_pl_dh @ primal_low
