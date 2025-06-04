"""Check that element matrices are properly computed."""

import numpy as np
import pytest
from mfv2d._mfv2d import compute_element_matrix_test
from mfv2d.mimetic2d import (
    BasisCache,
    ElementLeaf2D,
)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def test_check_nodal_undef(n: int) -> None:
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
def test_check_nodal_def(n: int) -> None:
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
def test_check_surf_undef(n: int) -> None:
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
def test_check_surf_def(n: int) -> None:
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
def test_check_edge_undef(n: int) -> None:
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
def test_check_edge_def(n: int) -> None:
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
