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
from mfv2d.element import ElementLeaf2D, element_dual_dofs, element_primal_dofs
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.solve_system import BasisCache, rhs_2d_element_projection


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_nodal(n: int) -> None:
    """Check nodal projection."""

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    e = ElementLeaf2D(None, n, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    dual = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_0, corners, basis_2d, test_function
    )

    w = KFormUnknown(2, "u", 0).weight
    cache = BasisCache(n, n + 2)
    prev = rhs_2d_element_projection(w @ test_function, corners, n, n, cache, cache)
    assert pytest.approx(prev) == dual


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_surf(n: int) -> None:
    """Check surface projection."""

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    e = ElementLeaf2D(None, n, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    dual = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_2, corners, basis_2d, test_function
    )

    w = KFormUnknown(2, "u", 2).weight
    cache = BasisCache(n, n + 2)
    prev = rhs_2d_element_projection(w @ test_function, corners, n, n, cache, cache)
    assert pytest.approx(prev) == dual


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_edge(n: int) -> None:
    """Check edge projection."""

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return np.stack((x**3 + 2 * y - x * y, x**2 - y**2 + x + 2), axis=-1)

    e = ElementLeaf2D(None, n, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    dual = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_1, corners, basis_2d, test_function
    )

    w = KFormUnknown(2, "u", 1).weight
    cache = BasisCache(n, n + 2)
    prev = rhs_2d_element_projection(w @ test_function, corners, n, n, cache, cache)
    assert pytest.approx(prev) == dual


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_projection_self_check(n: int) -> None:
    """Check projection to same space is identity."""
    int_rule = IntegrationRule1D(n + 2)
    basis_1d = Basis1D(n, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    e = ElementLeaf2D(None, n, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

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
