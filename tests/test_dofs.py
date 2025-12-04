"""Test that DoFs are properly expressed."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import Mesh, compute_gll
from mfv2d.continuity import _get_side_dof_nodes, _get_side_dofs
from mfv2d.kform import UnknownFormOrder
from mfv2d.mimetic2d import ElementSide, get_side_order, mesh_create


@dataclass(eq=False)
class ElementLeaf2D:
    """Two dimensional square element.

    This type facilitates operations related to calculations which need
    to be carried out on the reference element itself, such as calculation
    of the mass and incidence matrices, as well as the reconstruction of
    the solution.

    Parameters
    ----------
    order_h : int
        Order of the basis functions used for the nodal basis in the first dimension.
    order_v : int
        Order of the basis functions used for the nodal basis in the second dimension.
    bottom_left : (float, float)
        Coordinates of the bottom left corner.
    bottom_right : (float, float)
        Coordinates of the bottom right corner.
    top_right : (float, float)
        Coordinates of the top right corner.
    top_left : (float, float)
        Coordinates of the top left corner.
    """

    order: int
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]
    top_left: tuple[float, float]


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

    constraints = _get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (5, 6, 9, 10)
    bnd_leaves = [
        ElementLeaf2D(mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
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
                e_idx = bnd_indices.index(mesh.find_leaf_by_index(ec.i_e))
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

    constraints = _get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == 11

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (1, 2)
    bnd_leaves = [
        ElementLeaf2D(mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
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
                    val_child[bnd_indices.index(mesh.find_leaf_by_index(elem_con.i_e))][
                        elem_con.dofs
                    ]
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
    constraints = _get_side_dofs(
        mesh, 1, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 1, ElementSide.SIDE_BOTTOM)

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (5, 6)
    bnd_leaves = [
        ElementLeaf2D(mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
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
                    val_child[bnd_indices.index(mesh.find_leaf_by_index(elem_con.i_e))][
                        elem_con.dofs
                    ]
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

    constraints = _get_side_dofs(
        mesh, 0, ElementSide.SIDE_BOTTOM, UnknownFormOrder.FORM_ORDER_0
    )
    max_order = get_side_order(mesh, 0, ElementSide.SIDE_BOTTOM)
    assert max_order == 4

    def test_function(x: npt.ArrayLike, order: int):
        """Function to test with."""
        return np.asarray(x, np.float64) ** order

    bnd_indices = (1, 9, 10, 6)
    bnd_leaves = [
        ElementLeaf2D(mesh.get_leaf_orders(ie)[0], *mesh.get_leaf_corners(ie))
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
                    val_child[bnd_indices.index(mesh.find_leaf_by_index(elem_con.i_e))][
                        elem_con.dofs
                    ]
                    * elem_con.coeffs
                )
            # print(f"Error for order {order=} is {np.abs(res - val_real[ic]):.3e}")
            assert np.isclose(res, val_real[ic])


@pytest.mark.parametrize(("max_order", "pdiv"), ((3, 0.7), (5, 0.9), (4, 0.8)))
def test_mesh_merged_boundary(max_order: int, pdiv: float) -> None:
    """Check that mesh methods correctly return information about merged boundaries."""
    rng = np.random.default_rng(35)
    mesh = mesh_create(
        rng.integers(1, max_order),
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
        ((1, 2), (2, 3), (3, 4), (4, 1)),
        ((1, 2, 3, 4),),
    )

    def division_function(_m: Mesh, ie: int):
        """Division function."""
        if ie > 0 and pdiv > rng.random():
            return None

        o1 = int(rng.integers(1, max_order))
        o2 = int(rng.integers(1, max_order))
        o3 = int(rng.integers(1, max_order))
        o4 = int(rng.integers(1, max_order))
        return ((o1, o1), (o2, o2), (o3, o3), (o4, o4))

    mesh = mesh.split_depth_first(5, division_function)
    print("Divided mesh had", mesh.leaf_count, "leaves.")

    for ie in range(mesh.element_count):
        for side in ElementSide:
            # For checking the nodes only, the form order is irrelevant.
            element_constraints = _get_side_dof_nodes(
                mesh, ie, side, UnknownFormOrder.FORM_ORDER_0
            )
            expected_nodes = np.concatenate([ec.coeffs for ec in element_constraints])
            computed_nodes = mesh.get_element_side_merged_nodes(ie, side)
            boundary_elements = mesh.get_boundary_leaves(ie, side)
            assert len(computed_nodes) == mesh.get_element_side_merged_order(ie, side) + 1
            assert pytest.approx(computed_nodes) == expected_nodes
            assert np.all(
                tuple(mesh.get_leaf_index(ie) for ie in boundary_elements)
                == tuple(ec.i_e for ec in element_constraints)
            )


if __name__ == "__main__":
    test_mesh_merged_boundary(3, 0.7)
