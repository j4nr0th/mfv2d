"""Check that the mesh works as intended."""

import numpy as np
from mfv2d.mimetic2d import mesh_create


def test_manual():
    """Check that manually created mesh is correct."""
    points = (
        (-2, -1),
        (-1, -2),
        (+1, -1),
        (0, 0),
        (+1, +2),
        (0, +1),
        (-2, 0),
    )
    orders = ((2, 4), (3, 5), (5, 1))

    mesh = mesh_create(
        orders,
        points,
        (
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (3, 5),
            (5, 6),
            (4, 6),
            (1, 7),
            (6, 7),
        ),
        ((1, 2, 3, 4), (-3, 5, 6, -7), (-4, 7, 9, -8)),
    )

    bnd = mesh.boundary_indices
    assert set(bnd) == {0, 1, 4, 5, 8, 7}

    assert mesh.element_count == 3
    real_corners = (
        (points[0], points[1], points[2], points[3]),
        (points[3], points[2], points[4], points[5]),
        (points[0], points[3], points[5], points[6]),
    )
    for i in range(3):
        corners = mesh.get_leaf_corners(i)
        assert np.all(corners == real_corners[i])
        order_1, order_2 = mesh.get_leaf_orders(i)
        assert order_1 == orders[i][0] and order_2 == orders[i][1]
        assert mesh.get_element_children(i) is None
        assert mesh.get_element_parent(i) is None


def test_subdivision():
    """Check that subdividing the mesh works as expected."""
    points = (
        (-2, -1),
        (-1, -2),
        (+1, -1),
        (0, 0),
        (+1, +2),
        (0, +1),
        (-2, 0),
    )
    orders = ((2, 4), (3, 5), (5, 1))

    mesh = mesh_create(
        orders,
        points,
        (
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (3, 5),
            (5, 6),
            (4, 6),
            (1, 7),
            (6, 7),
        ),
        ((1, 2, 3, 4), (-3, 5, 6, -7), (-4, 7, 9, -8)),
    )
    split_orders = ((1, 1), (2, 1), (1, 2), (2, 2))
    mesh.split_element(1, *split_orders)
    assert mesh.element_count == 7
    assert np.all(mesh.get_leaf_indices() == (0, 2, 3, 4, 5, 6))
    children = mesh.get_element_children(1)
    assert children is not None

    new_corners = (
        (
            (0, 0),
            (0.5, -0.5),
            (0.5, 0.5),
            (0, 0.5),
        ),
        (
            (0.5, -0.5),
            (+1, -1),
            (+1, 0.5),
            (0.5, 0.5),
        ),
        (
            (0.5, 0.5),
            (+1, 0.5),
            (+1, +2),
            (0.5, 1.5),
        ),
        (
            (0, 0.5),
            (0.5, 0.5),
            (0.5, 1.5),
            (0, 1),
        ),
    )

    for j, i in enumerate((3, 4, 5, 6)):
        assert mesh.get_element_parent(i) == 1
        assert children[j] == i
        assert np.all(mesh.get_leaf_corners(i) == new_corners[j])
        assert np.all(mesh.get_leaf_orders(i) == split_orders[j])


def test_double_subdivision():
    """Check that subdividing twice the mesh works as expected."""
    points = (
        (-2, -1),
        (-1, -2),
        (+1, -1),
        (0, 0),
        (+1, +2),
        (0, +1),
        (-2, 0),
    )
    orders = ((2, 4), (3, 5), (5, 1))

    mesh = mesh_create(
        orders,
        points,
        (
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (3, 5),
            (5, 6),
            (4, 6),
            (1, 7),
            (6, 7),
        ),
        ((1, 2, 3, 4), (-3, 5, 6, -7), (-4, 7, 9, -8)),
    )
    split_orders = ((1, 1), (2, 1), (1, 2), (2, 2))
    mesh.split_element(1, *split_orders)
    # Check it can not double split
    try:
        mesh.split_element(1, *split_orders)
        assert False, "Should not double split!"
    except RuntimeError:
        pass

    mesh.split_element(4, *split_orders)
    assert mesh.element_count == 11
    assert np.all(mesh.get_leaf_indices() == (0, 2, 3, 5, 6, 7, 8, 9, 10))
    assert mesh.get_element_parent(4) == 1
    children = mesh.get_element_children(4)
    assert children is not None
    for j, ie in enumerate((7, 8, 9, 10)):
        assert mesh.get_element_parent(ie) == 4
        assert mesh.get_leaf_orders(ie) == split_orders[j]
        assert children[j] == ie

    children2 = mesh.get_element_children(1)
    assert children2 == (3, 4, 5, 6)
