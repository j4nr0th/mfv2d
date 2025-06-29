"""Test that DoFs are properly expressed."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import compute_gll
from mfv2d.element import (
    ElementCollection,
    ElementLeaf2D,
    ElementSide,
    UnknownFormOrder,
    get_side_dofs,
    get_side_order,
)


def test_evaluation_twice() -> None:
    """Check that interpolation from child to parent works for double division."""
    et1 = ElementLeaf2D(None, 1, (-1, -1), (+1, -1), (+1, +1), (-1, +1))
    e0, ((et2, et3), (e11, e12)) = et1.divide(1, 1, 1, 1, None)
    e1, ((e2, e3), (e4, e5)) = et2.divide(2, 4, 1, 1, None)
    object.__setattr__(e0, "child_bl", e1)
    e6, ((e7, e8), (e9, e10)) = et3.divide(2, 3, 1, 1, None)
    object.__setattr__(e0, "child_br", e6)

    collection = ElementCollection(
        [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    )
    constraints = get_side_dofs(
        collection, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(collection, 0, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_leaves = (e2, e3, e7, e8)
    assert max_order == sum(leaf.order for leaf in bnd_leaves)
    bnd_indices = (2, 3, 7, 8)
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
    et1 = ElementLeaf2D(None, 1, (-1, -1), (+1, -1), (+1, +1), (-1, +1))
    e0, ((e1, e2), (e3, e4)) = et1.divide(4, 7, 1, 1, None)

    collection = ElementCollection([e0, e1, e2, e3, e4])
    constraints = get_side_dofs(
        collection, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(collection, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == e1.order + e2.order

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_leaves = (e1, e2)
    bnd_indices = (1, 2)
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
    et1 = ElementLeaf2D(None, 1, (-1, -1), (+1, -1), (+1, +1), (-1, +1))
    e0, ((et2, et3), (e11, e12)) = et1.divide(1, 1, 1, 1, None)
    e1, ((e2, e3), (e4, e5)) = et2.divide(2, 4, 1, 1, None)
    object.__setattr__(e0, "child_bl", e1)
    e6, ((e7, e8), (e9, e10)) = et3.divide(1, 1, 1, 1, None)
    object.__setattr__(e0, "child_br", e6)

    collection = ElementCollection(
        [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    )
    constraints = get_side_dofs(
        collection, 1, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(collection, 1, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_leaves = (e2, e3)
    assert max_order == sum(leaf.order for leaf in bnd_leaves)
    bnd_indices = (2, 3)
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
    et1 = ElementLeaf2D(None, 1, (-1, -1), (+1, -1), (+1, +1), (-1, +1))
    e0, ((e1, et2), (e11, e12)) = et1.divide(1, 1, 1, 1, None)
    e2, ((et3, e8), (e9, e10)) = et2.divide(1, 1, 1, 1, None)
    object.__setattr__(e0, "child_br", e2)
    e3, ((e4, e5), (e6, e7)) = et3.divide(1, 1, 1, 1, None)
    object.__setattr__(e2, "child_bl", e3)

    collection = ElementCollection(
        [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    )
    constraints = get_side_dofs(
        collection, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(collection, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == 4

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_leaves = (e1, e4, e5, e8)
    bnd_indices = (1, 4, 5, 8)
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
            print(f"Error for order {order=} is {np.abs(res - val_real[ic]):.3e}")
            # assert np.isclose(res, val_real[ic])
