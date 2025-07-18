"""Test that DoFs are properly expressed."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import compute_gll
from mfv2d.kform import UnknownFormOrder
from mfv2d.mimetic2d import (
    ElementLeaf2D,
    ElementSide,
    get_side_dofs,
    get_side_order,
    mesh_create,
)


def test_evaluation_twice() -> None:
    """Check that interpolation from child to parent works for double division."""
    mesh = mesh_create(
        1,
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
        ((1, 2), (2, 3), (3, 4), (4, 1)),
        ((1, 2, 3, 4),),
    )

    mesh.split_element(0, (1, 1), (1, 1), (1, 1), (1, 1))
    mesh.split_element(1, (2, 2), (4, 4), (1, 1), (1, 1))
    mesh.split_element(2, (2, 2), (3, 3), (1, 1), (1, 1))

    constraints = get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (5, 6, 9, 10)
    bnd_leaves = [
        ElementLeaf2D(None, mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
        for ie in bnd_indices
    ]

    assert max_order == sum(leaf.order for leaf in bnd_leaves)
    bnd_pts = tuple(
        (
            compute_gll(leaf.order)[0] * (leaf.bottom_right[0] - leaf.bottom_left[0]) / 2
            + (leaf.bottom_left[0] + leaf.bottom_right[0]) / 2
        )
        for leaf in bnd_leaves
    )
    pos_real, _ = compute_gll(max_order)

    for order in range(max_order + 1):
        child_vals = tuple(test_function(pts, order) for pts in bnd_pts)
        real_vals = test_function(pos_real, order)

        for ic, con in enumerate(constraints):
            res = 0
            row = list()
            for ie, ec in enumerate(con.element_constraints):
                e_idx = bnd_indices.index(ec.i_e)
                e_val = child_vals[e_idx][ec.dofs]
                e_coe = ec.coeffs
                res += np.sum(e_coe * e_val)
                row.append(e_coe)
            assert pytest.approx(res) == real_vals[ic]
        #     print(f"Error for {ic=} {order=}: {np.abs(res - real_vals[ic]):.3e}")
        # print()


def test_evaluation_once() -> None:
    """Check that interpolation from child to parent works for single division."""
    mesh = mesh_create(
        1,
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
        ((1, 2), (2, 3), (3, 4), (4, 1)),
        ((1, 2, 3, 4),),
    )

    mesh.split_element(0, (4, 4), (7, 7), (1, 1), (1, 1))

    constraints = get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == 11

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (1, 2)
    bnd_leaves = [
        ElementLeaf2D(None, mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
        for ie in bnd_indices
    ]
    bnd_pts = tuple(
        (
            compute_gll(leaf.order)[0] * (leaf.bottom_right[0] - leaf.bottom_left[0]) / 2
            + (leaf.bottom_left[0] + leaf.bottom_right[0]) / 2
        )
        for leaf in bnd_leaves
    )

    for order in range(max_order + 1):
        val_child = tuple(test_function(pts, order) for pts in bnd_pts)
        pos_real, _ = compute_gll(max_order)
        val_real = test_function(pos_real, order)

        for ic, con in enumerate(constraints):
            # print(f"Constraint {ic=}:")
            # for ie, ec in enumerate(con.element_constraints):
            #     print(f"\t{ec=}")
            # print("\n")
            res = 0
            for elem_con in con.element_constraints:
                res += np.sum(
                    val_child[bnd_indices.index(elem_con.i_e)][elem_con.dofs]
                    * elem_con.coeffs
                )
            # print(f"Error for order {order=} is {np.abs(res - val_real[ic]):.3e}")
            assert np.isclose(res, val_real[ic])


def test_evaluation_twice_1() -> None:
    """Check that interpolation from child to parent works for double division."""
    mesh = mesh_create(
        1,
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
        ((1, 2), (2, 3), (3, 4), (4, 1)),
        ((1, 2, 3, 4),),
    )

    mesh.split_element(0, (1, 1), (1, 1), (1, 1), (1, 1))
    mesh.split_element(1, (2, 2), (4, 4), (1, 1), (1, 1))
    mesh.split_element(2, (2, 2), (3, 3), (1, 1), (1, 1))
    constraints = get_side_dofs(
        mesh, 1, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 1, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (5, 6)
    bnd_leaves = [
        ElementLeaf2D(None, mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
        for ie in bnd_indices
    ]
    assert max_order == sum(leaf.order for leaf in bnd_leaves)
    bnd_pts = tuple(
        (
            compute_gll(leaf.order)[0] / 2
            + (leaf.bottom_left[0] + leaf.bottom_right[0])
            + 1
        )
        for leaf in bnd_leaves
    )
    pos_real, _ = compute_gll(max_order)

    for order in range(0, max_order + 1):
        val_child = tuple(test_function(pts, order) for pts in bnd_pts)
        val_real = test_function(pos_real, order)

        for ic, con in enumerate(constraints):
            res = 0
            for elem_con in con.element_constraints:
                res += np.sum(
                    val_child[bnd_indices.index(elem_con.i_e)][elem_con.dofs]
                    * elem_con.coeffs
                )
            # print(f"Error for order {order=} is {np.abs(res - val_real[ic]):.3e}")
            # print("")
            assert pytest.approx(res) == val_real[ic]


def test_evaluation() -> None:
    """Check that interpolation from child to parent works."""
    mesh = mesh_create(
        1,
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
        ((1, 2), (2, 3), (3, 4), (4, 1)),
        ((1, 2, 3, 4),),
    )

    mesh.split_element(0, (1, 1), (1, 1), (1, 1), (1, 1))
    mesh.split_element(2, (1, 1), (1, 1), (1, 1), (1, 1))
    mesh.split_element(5, (1, 1), (1, 1), (1, 1), (1, 1))

    constraints = get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == 4

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (1, 9, 10, 6)
    bnd_leaves = [
        ElementLeaf2D(None, mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
        for ie in bnd_indices
    ]
    bnd_pts = tuple(
        (
            compute_gll(leaf.order)[0] * (leaf.bottom_right[0] - leaf.bottom_left[0]) / 2
            + (leaf.bottom_left[0] + leaf.bottom_right[0]) / 2
        )
        for leaf in bnd_leaves
    )

    for order in range(max_order + 1):
        val_child = tuple(test_function(pts, order) for pts in bnd_pts)
        pos_real, _ = compute_gll(max_order)
        val_real = test_function(pos_real, order)

        for ic, con in enumerate(constraints):
            # print(f"Constraint {ic=}:")
            # for ie, ec in enumerate(con.element_constraints):
            #     print(f"\t{ec=}")
            # print("\n")
            res = 0
            for elem_con in con.element_constraints:
                res += np.sum(
                    val_child[bnd_indices.index(elem_con.i_e)][elem_con.dofs]
                    * elem_con.coeffs
                )
            # print(f"Error for order {order=} is {np.abs(res - val_real[ic]):.3e}")
            assert np.isclose(res, val_real[ic])
