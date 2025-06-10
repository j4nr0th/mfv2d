"""Check that projections work correctly."""

from itertools import accumulate

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    IntegrationRule1D,
    compute_element_projector,
    dlagrange1d,
    lagrange1d,
)
from mfv2d.element import (
    ElementLeaf2D,
    element_dual_dofs,
    element_primal_dofs,
    jacobian,
    poly_x,
    poly_y,
    reconstruct,
)
from mfv2d.kform import KElementProjection, KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import BasisCache


def rhs_2d_element_projection(
    right: KElementProjection,
    corners: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    cache_1: BasisCache,
    cache_2: BasisCache,
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element2D
        The element on which the projection is evaluated on.
    cache : BasisCache
        Cache for the correct element order.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    # TODO: don't recompute basis, just reuse the cached values.
    assert cache_1.basis_order == order_1
    assert cache_2.basis_order == order_1
    fn = right.func

    n_dof: int
    if right.weight.order == 0:
        n_dof = (order_1 + 1) * (order_2 + 1)
    elif right.weight.order == 1:
        n_dof = (order_1 + 1) * order_2 + order_1 * (order_2 + 1)
    elif right.weight.order == 2:
        n_dof = order_1 * order_2
    else:
        raise ValueError(f"Invalid weight order {right.weight.order}.")

    if fn is None:
        return np.zeros(n_dof)

    out_vec = np.empty(n_dof)

    basis_vals: list[npt.NDArray[np.floating]] = list()

    nodes_1 = cache_1.int_nodes_1d
    weights_1 = cache_1.int_weights_1d
    nodes_2 = cache_2.int_nodes_1d
    weights_2 = cache_2.int_weights_1d

    (j00, j01), (j10, j11) = jacobian(corners, nodes_1[None, :], nodes_2[:, None])
    det = j00 * j11 - j10 * j01

    real_x = poly_x(corners[:, 0], nodes_1[None, :], nodes_2[:, None])
    real_y = poly_y(corners[:, 1], nodes_1[None, :], nodes_2[:, None])
    f_vals = fn(real_x, real_y)
    weights_2d = weights_1[None, :] * weights_2[:, None]

    # Deal with vectors first. These need special care.
    if right.weight.order == 1:
        values1 = lagrange1d(cache_1.nodes_1d, nodes_1)
        d_vals1 = dlagrange1d(cache_1.nodes_1d, nodes_1)
        values2 = lagrange1d(cache_2.nodes_1d, nodes_2)
        d_vals2 = dlagrange1d(cache_2.nodes_1d, nodes_2)
        d_values1 = tuple(accumulate(-d_vals1[..., i] for i in range(order_1)))
        d_values2 = tuple(accumulate(-d_vals2[..., i] for i in range(order_2)))

        new_f0 = j00 * f_vals[..., 0] + j01 * f_vals[..., 1]
        new_f1 = j10 * f_vals[..., 0] + j11 * f_vals[..., 1]

        for i1 in range(order_2 + 1):
            v1 = values2[..., i1]
            for j1 in range(order_1):
                u1 = d_values1[j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[i1 * order_1 + j1] = np.sum(basis1 * weights_2d * new_f1)

        for i1 in range(order_2):
            v1 = d_values2[i1]
            for j1 in range(order_1 + 1):
                u1 = values1[..., j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[order_1 * (order_2 + 1) + i1 * (order_1 + 1) + j1] = np.sum(
                    basis1 * weights_2d * new_f0
                )
        return out_vec

    if right.weight.order == 2:
        d_vals1 = dlagrange1d(cache_1.nodes_1d, nodes_1)
        d_vals2 = dlagrange1d(cache_2.nodes_1d, nodes_2)
        d_values1 = tuple(accumulate(-d_vals1[..., i] for i in range(order_1)))
        d_values2 = tuple(accumulate(-d_vals2[..., i] for i in range(order_2)))
        for i1 in range(order_2):
            v1 = d_values2[i1]
            for j1 in range(order_1):
                u1 = d_values1[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof

    elif right.weight.order == 0:
        values1 = lagrange1d(cache_1.nodes_1d, nodes_1)
        values2 = lagrange1d(cache_2.nodes_1d, nodes_2)
        for i1 in range(order_2 + 1):
            v1 = values2[..., i1]
            for j1 in range(order_1 + 1):
                u1 = values1[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof
        weights_2d *= det

    else:
        raise ValueError(f"Invalid weight order {right.weight.order}.")

    # Compute rhs integrals
    for i, bv in enumerate(basis_vals):
        out_vec[i] = np.sum(bv * f_vals * weights_2d)

    return out_vec


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


def test_reconstruction_nodal() -> None:
    """Check nodal reconstruction is exact at a high enough order."""
    N = 6

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    e = ElementLeaf2D(None, N, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

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


def test_reconstruction_surf() -> None:
    """Check nodal reconstruction is exact at a high enough order."""
    N = 6

    def test_function(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Test function."""
        return x**3 + 2 * y - x * y

    e = ElementLeaf2D(None, N, (-2, -1.1), (+0.7, -1.5), (+1, +1), (-1.2, +1))
    corners = np.array(
        [e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64
    )

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
