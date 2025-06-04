"""Check that element matrices are properly computed."""

import numpy as np
import pytest
from mfv2d._mfv2d import compute_element_matrix_test

# from mfv2d.element import _element_node_mass_mixed
from mfv2d.mimetic2d import (
    # Basis1D,
    # Basis2D,
    BasisCache,
    ElementLeaf2D,
    # IntegrationRule1D,
)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_nodal_undef(n: int) -> None:
    """Check that nodal matrix is correct on an undeformed."""
    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 2)
    mat_nodal, _, _ = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_node(cache)

    assert np.allclose(mat_nodal, mat_real)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_nodal_def(n: int) -> None:
    """Check that nodal matrix is correct on a deformed."""
    corners = np.array(
        (
            (-2, -1.1),
            (+1, -1.3),
            (+2, +2),
            (-1.3, +0.5),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 2)
    mat_nodal, _, _ = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_node(cache)

    assert np.allclose(mat_nodal, mat_real)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_surf_undef(n: int) -> None:
    """Check that surface matrix is correct on an undeformed."""
    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 2)
    _, _, mat_surf = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_surface(cache)

    assert np.allclose(mat_surf, mat_real)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_surf_def(n: int) -> None:
    """Check that surface matrix is correct on a deformed."""
    corners = np.array(
        (
            (-2, -1.1),
            (+1, -1.3),
            (+2, +2),
            (-1.3, +0.5),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 2)
    _, _, mat_surf = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_surface(cache)

    assert np.allclose(mat_surf, mat_real)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_edge_undef(n: int) -> None:
    """Check that edge matrix is correct on an undeformed."""
    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 2)
    _, mat_edge, _ = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_edge(cache)

    assert np.allclose(mat_edge, mat_real)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def check_edge_def(n: int) -> None:
    """Check that edge matrix is correct on a deformed."""
    corners = np.array(
        (
            (-2, -1.1),
            (+1, -1.3),
            (+2, +2),
            (-1.3, +0.5),
        ),
        np.float64,
    )
    cache = BasisCache(n, n + 5)
    _, mat_edge, _ = compute_element_matrix_test(
        corners,
        n,
        n,
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
        np.ascontiguousarray(cache.nodal_1d.T, np.float64),
        np.ascontiguousarray(cache.edge_1d.T, np.float64),
        np.ascontiguousarray(cache.int_weights_1d),
        np.ascontiguousarray(cache.int_nodes_1d),
    )
    e = ElementLeaf2D(None, n, *corners)
    mat_real = e.mass_matrix_edge(cache)

    assert np.allclose(mat_edge, mat_real)


# @pytest.mark.parametrize(("n1", "n2"), ((1, 1), (2, 2), (1, 2), (2, 1), (4, 2), (3, 2)))
# def check_nodal_mixed(n1: int, n2: int) -> None:
#     """Check that mixed nodal matrix is correct on an undeformed."""
#     corners = np.array(
#         (
#             (-1, -1),
#             (+1, -1),
#             (+1, +1),
#             (-1, +1),
#         ),
#         np.float64,
#     )
#     rule = IntegrationRule1D(n1 + n2)
#     b1_in = Basis1D(n1, rule)
#     b2_in = Basis2D(b1_in, b1_in)
#     b1_out = Basis1D(n2, rule)

#     mat_nodal, _, _ = compute_element_matrix_test(
#         corners,
#         n,
#         n,
#         np.ascontiguousarray(cache.nodal_1d.T, np.float64),
#         np.ascontiguousarray(cache.edge_1d.T, np.float64),
#         np.ascontiguousarray(cache.int_weights_1d),
#         np.ascontiguousarray(cache.int_nodes_1d),
#         np.ascontiguousarray(cache.nodal_1d.T, np.float64),
#         np.ascontiguousarray(cache.edge_1d.T, np.float64),
#         np.ascontiguousarray(cache.int_weights_1d),
#         np.ascontiguousarray(cache.int_nodes_1d),
#     )

#     mat_real = _element_node_mass_mixed(

#     )

#     assert np.allclose(mat_nodal, mat_real)
