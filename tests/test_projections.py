"""Check that projections work correctly."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    IntegrationRule1D,
    compute_element_projector,
)
from mfv2d.element import (
    element_primal_dofs,
    poly_x,
    poly_y,
    reconstruct,
)
from mfv2d.kform import UnknownFormOrder


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

    dual = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, corners, basis_2d, test_function
    )
    test_v = np.linspace(-1, +1, 21)
    recon = reconstruct(corners, 0, dual, test_v[None, :], test_v[:, None], basis_2d)

    real = test_function(
        poly_x(corners[:, 0], test_v[None, :], test_v[:, None]),
        poly_y(corners[:, 1], test_v[None, :], test_v[:, None]),
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

    dual = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, corners, basis_2d, test_function
    )
    test_v = np.linspace(-1, +1, 21)
    recon = reconstruct(corners, 2, dual, test_v[None, :], test_v[:, None], basis_2d)

    real = test_function(
        poly_x(corners[:, 0], test_v[None, :], test_v[:, None]),
        poly_y(corners[:, 1], test_v[None, :], test_v[:, None]),
    )
    assert pytest.approx(recon) == real


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_self_check(n: int) -> None:
    """Check projection to same space is identity."""
    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array([(-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1)], np.float64)

    for m, order in zip(((n + 1) ** 2, 2 * n * (n + 1), n**2), UnknownFormOrder):
        (mat,) = compute_element_projector([order], corners, basis_2d, basis_2d)

        assert pytest.approx(mat) == np.eye(m)


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

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, corners, basis_2d_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, corners, basis_2d_high, test_function
    )

    projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_0], corners, basis_2d_low, basis_2d_high
    )

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_0], corners, basis_2d_high, basis_2d_low
    )
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

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, corners, basis_2d_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, corners, basis_2d_high, test_function
    )

    projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_1], corners, basis_2d_low, basis_2d_high
    )

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_1], corners, basis_2d_high, basis_2d_low
    )
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

    low_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, corners, basis_2d_low, test_function
    )

    high_dofs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, corners, basis_2d_high, test_function
    )

    projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_2], corners, basis_2d_low, basis_2d_high
    )

    projected = (projector @ low_dofs).flatten()
    assert pytest.approx(projected) == high_dofs

    # Should work the other way around as well.
    reverse_projector = compute_element_projector(
        [UnknownFormOrder.FORM_ORDER_2], corners, basis_2d_high, basis_2d_low
    )
    projected = (reverse_projector @ high_dofs).flatten()
    assert pytest.approx(projected) == low_dofs
