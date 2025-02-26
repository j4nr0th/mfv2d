"""Functionality related to creating a full system of equations."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from time import perf_counter
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms as kform
from interplib._interp import compute_gll, lagrange1d
from interplib._mimetic import (
    GeoID,
    Surface,
    compute_element_matrices,
    compute_element_matrices_2,
    #     continuity,
)
from interplib.kforms.eval import _ctranslate, translate_equation
from interplib.kforms.kform import KBoundaryProjection
from interplib.mimetic.mimetic2d import BasisCache, Element2D, Mesh2D, element_system


@dataclass(init=False, frozen=True)
class ConstraintEquation:
    """Equation which represents a constraint enforced by a Lagrange multiplier.

    Parameters
    ----------
    indices : array_like
        Indices of of the degrees of freedom.
    values : array_like
        Values of the coefficients of the degrees of freedom in the equation.
    rhs : float
        Value of the constraint.
    """

    indices: npt.NDArray[np.uint32]
    values: npt.NDArray[np.float64]
    rhs: np.float64

    def __init__(self, indices: npt.ArrayLike, values: npt.ArrayLike, rhs: float) -> None:
        i = np.array(indices, dtype=np.uint32)
        v = np.array(values, dtype=np.float64)
        r = np.float64(rhs)
        if v.ndim != 1 or i.shape != v.shape:
            raise ValueError("Indices and values must be 1D arrays of equal length")
        object.__setattr__(self, "indices", i)
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "rhs", r)

    def __eq__(self, other, /) -> bool:
        """Check if two equations are identical."""
        if not isinstance(other, ConstraintEquation):
            return NotImplemented
        return bool(
            self.rhs == other.rhs
            and (self.indices.size == other.indices.size)
            and np.all(self.indices == other.indices)
            and np.all(self.values == other.values)
        )


@dataclass(init=False)
class ElementTree:
    """Container for mesh elements."""

    # Tuple of all elements in the tree.
    elements: tuple[Element2D, ...]
    # Order of the elements
    orders: npt.NDArray[np.uint32]
    # If true, then element with the same index has 4 children.
    children: npt.NDArray[np.bool]
    # How many elements are in each level.
    sizes: tuple[int, ...]
    # Offset of the first DoF within an element
    element_offsets: npt.NDArray[np.uint32]
    # Offset of the first DoF of a variable within an element
    dof_offsets: tuple[npt.NDArray[np.uint32], ...]
    # Total number of degrees of freedom
    n_dof: int
    # Total number of degrees of freedom in the leaf nodes
    n_dof_leaves: int

    def __init__(
        self,
        elements: Sequence[Element2D],
        predicate: None | Callable[[Element2D, int], bool],
        max_levels: int,
        unknowns: Sequence[kform.KFormUnknown],
    ) -> None:
        # Divide the elements as long as predicate is true.
        children: list[npt.NDArray[np.bool]] = list()
        sizes: list[int] = list()
        all_elems: list[Element2D] = list()
        for i in range(max_levels + 1):
            v: list[bool] = list()
            new_elem: list[Element2D] = list()

            sz = 0
            for e in elements:
                b = (
                    predicate(e, i)
                    if predicate is not None and i != max_levels
                    else False
                )
                v.append(b)
                all_elems.append(e)
                sz += 1
                if b:
                    # raise NotImplementedError("Not dividing the elements up (yet)!")
                    (ebl, ebr), (etl, etr) = e.divide(e.order)
                    new_elem.extend((ebl, ebr, etl, etr))

            children.append(np.array(v, np.bool))
            sizes.append(sz)
            if len(new_elem) == 0:
                break

            elements = new_elem

        # if i == max_levels:
        #     raise RuntimeError(
        #         f"Maximum number of refinement levels ({max_levels}) has been exceeded."
        #     )

        self.sizes = tuple(sizes)
        self.elements = tuple(all_elems)
        self.children = np.concatenate(children, dtype=np.bool)
        del all_elems, sizes, children

        # Number the elements. First are the leaves, then all the others
        n_total = len(self.elements)
        n_leaves = n_total - np.count_nonzero(self.children)
        indices = np.empty(n_total, dtype=np.uint32)
        indices[~self.children] = np.arange(n_leaves)
        indices[self.children] = np.arange(n_leaves, n_total)

        self.orders = np.array([e.order for e in self.elements], dtype=np.uint32)
        dof_sizes: list[npt.NDArray[np.uint32]] = list()
        # Compute DoF offsets within elements
        for form in unknowns:
            n = np.zeros_like(self.orders, np.uint32)
            if form.order == 0:
                n[self.children] = 4 * self.orders[self.children]
                n[~self.children] = (self.orders[~self.children] + 1) ** 2
            elif form.order == 1:
                n[self.children] = 4 * self.orders[self.children]
                n[~self.children] = (
                    (self.orders[~self.children] + 1) * self.orders[~self.children] * 2
                )
            elif form.order == 2:
                n[self.children] = 0
                n[~self.children] = (self.orders[~self.children]) ** 2
            else:
                assert False
            dof_sizes.append(n)
        self.dof_offsets = (np.zeros_like(self.orders, np.uint32), *accumulate(dof_sizes))
        off = np.pad(np.cumsum(self.dof_offsets[-1][indices]), (1, 0))
        self.element_offsets = off[indices]
        self.n_dof_leaves = int(off[n_leaves])
        self.n_dof = int(off[-1])

    def iter_leaves(self) -> Generator[Element2D]:
        """Iterate over leaves."""
        c: np.bool
        for e, c in zip(self.elements, self.children):
            if c:
                continue
            yield e

    def element_edge_dofs(self, index: int, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of edge DoFs on the boundary of an element."""
        n = self.orders[index]
        if self.children[index]:
            if bnd_idx == 0:
                return np.arange(n, dtype=np.uint32)

            elif bnd_idx == 1:
                return np.arange(n, 2 * n, dtype=np.uint32)

            elif bnd_idx == 2:
                return np.arange(2 * n, 3 * n, dtype=np.uint32)

            elif bnd_idx == 3:
                return np.arange(3 * n, 4 * n, dtype=np.uint32)

            raise ValueError("Only boundary ID of up to 3 is allowed.")

        if bnd_idx == 0:
            return np.arange(0, n, dtype=np.uint32)

        elif bnd_idx == 1:
            return np.astype(
                (n * (n + 1)) + n + np.arange(0, n, dtype=np.uint32) * (n + 1),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 2:
            return np.astype(
                np.flip(n * n + np.arange(0, n, dtype=np.uint32)),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 3:
            return np.astype(
                np.flip((n * (n + 1)) + np.arange(0, n, dtype=np.uint32) * (n + 1)),
                np.uint32,
                copy=False,
            )

        raise ValueError("Only boundary ID of up to 3 is allowed.")

    def element_node_dofs(self, index: int, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of nodal DoFs on the boundary of an element."""
        n = self.orders[index]
        if self.children[index]:
            if bnd_idx == 0:
                return np.arange(n + 1, dtype=np.uint32)

            elif bnd_idx == 1:
                return np.arange(n, 2 * (n + 1) - 1, dtype=np.uint32)

            elif bnd_idx == 2:
                return np.arange(2 * (n + 1) - 2, 3 * (n + 1) - 2, dtype=np.uint32)

            elif bnd_idx == 3:
                a = np.arange(3 * (n + 1) - 3, 4 * (n + 1) - 3, dtype=np.uint32)
                a[-1] = 0
                return a

            raise ValueError("Only boundary ID of up to 3 is allowed.")

        if bnd_idx == 0:
            return np.arange(0, n + 1, dtype=np.uint32)

        elif bnd_idx == 1:
            return np.astype(
                n + np.arange(0, n + 1, dtype=np.uint32) * (n + 1),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 2:
            return np.astype(
                np.flip((n + 1) * n + np.arange(0, n + 1, dtype=np.uint32)),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 3:
            return np.astype(
                np.flip(np.arange(0, n + 1, dtype=np.uint32) * (n + 1)),
                np.uint32,
                copy=False,
            )

        raise ValueError("Only boundary ID of up to 3 is allowed.")

    @property
    def leave_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of leaf elements."""
        return np.astype(np.nonzero(~self.children)[0], np.uint32)

    @property
    def n_levels(self) -> int:
        """Number of levels in the tree."""
        return len(self.sizes)


@cache
def continuity_matrices(
    n1: int, n2: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute continuity equation coefficients for 0 and 1 forms between elements.

    Parameters
    ----------
    n1 : int
        Higher order.
    n2 : int
        Lower order.

    Returns
    -------
    (n1 + 1, n2 + 1) array
        Array of coefficients for 0-form continuity.

    (n1, n2) array
        Array of coefficients for 1-form continuity.
    """
    assert n1 > n2
    nodes_n1, _ = compute_gll(n1)
    nodes_n2, _ = compute_gll(n2)

    # Axis 0: xi_j
    # Axis 1: psi_i
    nodal_basis = lagrange1d(nodes_n2, nodes_n1)

    # Axis 0: [xi_{j + 1}, xi_j]
    # Axis 1: psi_i
    coeffs_1_form = np.zeros((n1, n2), np.float64)
    for j in range(n1):
        coeffs_1_form[j, 0] = nodal_basis[j, 0] - nodal_basis[j + 1, 0]
        for i in range(1, n2):
            coeffs_1_form[j, i] = coeffs_1_form[j, i - 1] + (
                nodal_basis[j, i] - nodal_basis[j + 1, i]
            )

    diffs = nodal_basis[:-1, :] - nodal_basis[+1:, :]
    coeffs_1_fast = np.stack(
        [x for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))],
        axis=-1,
        dtype=np.float64,
    )

    assert np.allclose(coeffs_1_fast, coeffs_1_form)

    return np.astype(nodal_basis, np.float64, copy=False), coeffs_1_form


@cache
def continuity_child_matrices(
    nchild: int, nparent: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute continuity equation coefficients for 0 and 1 forms between elements.

    Parameters
    ----------
    nc : int
        Child's order.
    nparent : int
        Parent's order.

    Returns
    -------
    (nc + 1, np + 1) array
        Array of coefficients for 0-form continuity.

    (nc, np) array
        Array of coefficients for 1-form continuity.
    """
    assert nchild >= nparent
    nodes_child, _ = compute_gll(nchild)
    nodes_parent, _ = compute_gll(nparent)
    nodes_child = (nodes_child / 2) - 0.5  # Scale to [-1, 0]

    # Axis 0: xi_j
    # Axis 1: psi_i
    nodal_basis = lagrange1d(nodes_parent, nodes_child)

    # Axis 0: [xi_{j + 1}, xi_j]
    # Axis 1: psi_i
    coeffs_1_form = np.zeros((nchild, nparent), np.float64)
    for j in range(nchild):
        coeffs_1_form[j, 0] = nodal_basis[j, 0] - nodal_basis[j + 1, 0]
        for i in range(1, nparent):
            coeffs_1_form[j, i] = coeffs_1_form[j, i - 1] + (
                nodal_basis[j, i] - nodal_basis[j + 1, i]
            )

    diffs = nodal_basis[:-1, :] - nodal_basis[+1:, :]
    coeffs_1_fast = np.stack(
        [x for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))],
        axis=-1,
        dtype=np.float64,
    )

    assert np.allclose(coeffs_1_fast, coeffs_1_form)

    return np.astype(nodal_basis, np.float64, copy=False), coeffs_1_form


def endpoints_from_line(
    elm: Element2D, s: Surface, i: int
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    """Get endpoints of the element boundary based on the line index.

    Returns
    -------
    int
        Direction of the normal. Positive means it contributes to gradient.
    (float, float)
        Beginning of the line corresponding to -1 on the reference element.

    (float, float)
        End of the line corresponding to +1 on the reference element.
    """
    if s[0].index == i:
        return (+1, elm.bottom_left, elm.bottom_right)
    if s[1].index == i:
        return (-1, elm.bottom_right, elm.top_right)
    if s[2].index == i:
        return (-1, elm.top_right, elm.top_left)
    if s[3].index == i:
        return (+1, elm.top_left, elm.bottom_left)
    raise ValueError("Line is not in the element.")


def find_boundary_id(s: Surface, i: int) -> Literal[0, 1, 2, 3]:
    """Find what boundary the line with a given index is in the surface."""
    if s[0].index == i:
        return 0
    if s[1].index == i:
        return 1
    if s[2].index == i:
        return 2
    if s[3].index == i:
        return 3
    raise ValueError(f"Line with index {i} is not in the surface {s}.")


def solve_system_2d(
    system: kform.KFormSystem,
    mesh: Mesh2D,
    boundaray_conditions: Sequence[kform.BoundaryCondition2DStrong] | None = None,
    div_predicate: Callable[[Element2D, int], bool] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[kform.KFormUnknown, npt.NDArray[np.float64]],
    pv.UnstructuredGrid,
]:
    """Solve the system on the specified mesh.

    Parameters
    ----------
    system : kforms.KFormSystem
        System of equations to solve.
    mesh : Mesh2D
        Mesh on which to solve the system on.
    rec_order : int
        Order of reconstruction returned.
    boundary_conditions: Sequence of kforms.BoundaryCondition2DStrong, optional
        Sequence of boundary conditions to be applied to the system.
    div_predicate : Callable (Element2D) -> bool
        Callable used to determine if an element should be divided further.

    Returns
    -------
    x : array
        Array of x positions where the reconstructed values were computed.
    y : array
        Array of y positions where the reconstructed values were computed.
    reconstructions : dict of kforms.KFormUnknown to array
        Reconstructed solution for unknowns. The number of points on the element
        where these reconstructions are computed depends on the degree of the element.
    grid : pyvista.UnstructuredGrid
        Reconstructed solution as an unstructured grid of VTK's "lagrange quadrilateral"
        cells.
    """
    # Check that inputs make sense.
    strong_boundary_edges: dict[kform.KFormUnknown, list[npt.NDArray[np.uint64]]] = {}
    for primal in system.unknown_forms:
        if primal.order > 2:
            raise ValueError(
                f"Can not solve the system on a 2D mesh, as it contains a {primal.order}"
                "-form."
            )
        strong_boundary_edges[primal] = []

    # Check boundary conditions are sensible
    if boundaray_conditions is not None:
        for bc in boundaray_conditions:
            if bc.form not in system.unknown_forms:
                raise ValueError(
                    f"Boundary conditions specify form {bc.form}, which is not in the"
                    " system"
                )
            if np.any(bc.indices >= mesh.primal.n_lines):
                raise ValueError(
                    f"Boundary condition on {bc.form} specifies lines which are"
                    " outside not in the mesh (highest index specified was "
                    f"{np.max(bc.indices)}, but mesh has {mesh.primal.n_points}"
                    " lines)."
                )
            strong_boundary_edges[bc.form].append(bc.indices)

    strong_indices = {
        form: np.concatenate(strong_boundary_edges[form])
        if len(strong_boundary_edges[form])
        else np.array([])
        for form in strong_boundary_edges
    }
    del strong_boundary_edges

    cont_indices_edges: list[int] = []
    cont_indices_nodes: list[int] = []
    for form in system.unknown_forms:
        if form.order == 2:
            continue
        idx = system.unknown_forms.index(form)
        if form.order == 1:
            cont_indices_edges.append(idx)
        elif form.order == 0:
            cont_indices_nodes.append(idx)
        else:
            assert False

    # Make elements into a rectree
    element_tree = ElementTree(
        list(mesh.get_element(ie) for ie in range(mesh.n_elements)),
        div_predicate,
        1,
        system.unknown_forms,
    )

    leaf_elements: list[Element2D] = list(element_tree.iter_leaves())

    # Make element matrices and vectors
    cache: dict[int, BasisCache] = dict()
    for order in np.unique(element_tree.orders):
        cache[int(order)] = BasisCache(int(order), 3 * int(order))
    bytecodes = [translate_equation(eq.left, simplify=True) for eq in system.equations]

    t0 = perf_counter()
    element_outputs = tuple(
        element_system(system, e, cache[e.order], None) for e in leaf_elements
    )
    t1 = perf_counter()
    print(f"Element matrices old way: {t1 - t0} seconds.")
    codes = []
    for bite in bytecodes:
        row: list[list | None] = []
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
            else:
                row.append(None)
        codes.append(row)

    bl = np.array([e.bottom_left for e in leaf_elements])
    br = np.array([e.bottom_right for e in leaf_elements])
    tr = np.array([e.top_right for e in leaf_elements])
    tl = np.array([e.top_left for e in leaf_elements])
    orde = np.array([e.order for e in leaf_elements], np.uint32)
    t0 = perf_counter()
    second_matrices = compute_element_matrices(
        [f.order for f in system.unknown_forms],
        codes,
        bl,
        br,
        tr,
        tl,
        orde,
        [cache[o].c_serialization for o in cache],
    )
    t1 = perf_counter()
    print(f"Element matrices new way: {t1 - t0} seconds.")
    t0 = perf_counter()
    third_matrices = compute_element_matrices_2(
        [f.order for f in system.unknown_forms],
        codes,
        bl,
        br,
        tr,
        tl,
        orde,
        [cache[o].c_serialization for o in cache],
    )
    t1 = perf_counter()
    print(f"Element matrices newer way: {t1 - t0} seconds.")
    del bl, br, tr, tl, orde

    # from interplib._eval import element_matrices
    element_matrix: list[npt.NDArray[np.float64]] = [e[0] for e in element_outputs]

    for e, mat1, mat2, mat3 in zip(
        leaf_elements, element_matrix, second_matrices, third_matrices, strict=True
    ):
        assert mat1.shape == mat2.shape
        assert np.allclose(mat2, element_system(system, e, cache[e.order], None)[0])
        assert np.allclose(mat2, mat3)
    element_vectors: list[npt.NDArray[np.float64]] = [e[1] for e in element_outputs]

    # Apply lagrange multipliers for continuity
    equations: list[ConstraintEquation] = list()

    # TODO:
    #
    # Connect children to boundaries of their parents.
    level_offsets = [0] + list(accumulate(element_tree.sizes))[:-1]
    for i_level in reversed(range(element_tree.n_levels - 1)):
        i1 = 0
        i2 = 0
        l1 = element_tree.sizes[i_level]
        l2 = element_tree.sizes[i_level + 1]
        offset_parent = level_offsets[i_level]
        offset_children = level_offsets[i_level + 1]
        assert (l2 & 3) == 0, "Length of lower levels must be divisible by 4"
        while i1 < l1 and i2 < l2:
            idx_parent = i1 + offset_parent
            i1 += 1
            if not element_tree.children[idx_parent]:
                continue
            idx_00 = i2 + offset_children + 0
            idx_01 = i2 + offset_children + 1
            idx_10 = i2 + offset_children + 2
            idx_11 = i2 + offset_children + 3
            i2 += 4

            # Glue children to each other
            child_eq: list[ConstraintEquation] = list()
            if cont_indices_edges:
                # Glue 1-form edges
                child_eq += continuity_element_1_forms(
                    element_tree, cont_indices_edges, idx_01, idx_00, 3, 1
                )
                child_eq += continuity_element_1_forms(
                    element_tree, cont_indices_edges, idx_11, idx_01, 0, 2
                )
                child_eq += continuity_element_1_forms(
                    element_tree, cont_indices_edges, idx_10, idx_11, 1, 3
                )
                child_eq += continuity_element_1_forms(
                    element_tree, cont_indices_edges, idx_00, idx_10, 2, 0
                )

                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_00, 0, False
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_01, 0, True
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_01, 1, False
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_11, 1, True
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_11, 2, False
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_10, 2, True
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_10, 3, False
                )
                child_eq += continuity_parent_child_edges(
                    element_tree, cont_indices_edges, idx_parent, idx_00, 3, True
                )

            if cont_indices_nodes:
                # Glue 0-form edges
                child_eq += continuity_0_forms_inner(
                    element_tree, cont_indices_nodes, 3, 1, idx_01, idx_00
                )
                child_eq += continuity_0_forms_inner(
                    element_tree, cont_indices_nodes, 0, 2, idx_11, idx_01
                )
                child_eq += continuity_0_forms_inner(
                    element_tree, cont_indices_nodes, 1, 3, idx_10, idx_11
                )
                child_eq += continuity_0_forms_inner(
                    element_tree, cont_indices_nodes, 2, 0, idx_00, idx_10
                )
                # Glue the corner they all share
                child_eq += continuity_0_form_corner(
                    element_tree, cont_indices_nodes, 1, 3, idx_00, idx_01
                )
                child_eq += continuity_0_form_corner(
                    element_tree, cont_indices_nodes, 2, 0, idx_01, idx_11
                )
                child_eq += continuity_0_form_corner(
                    element_tree, cont_indices_nodes, 3, 1, idx_11, idx_10
                )
                # Don't add the fourth equation!

                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_00, 0, False
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_01, 0, True
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_01, 1, False
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_11, 1, True
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_11, 2, False
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_10, 2, True
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_10, 3, False
                )
                child_eq += continuity_parent_child_nodes(
                    element_tree, cont_indices_nodes, idx_parent, idx_00, 3, True
                )

            equations.extend(child_eq)

    t0 = perf_counter()
    # Continuity of 1-forms
    if cont_indices_edges:
        for il in range(mesh.dual.n_lines):
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end
            if not idx_self or not idx_neighbour:
                continue

            # For each variable which must be continuous, get locations in left and right
            s_other = mesh.primal.get_surface(idx_neighbour)
            s_self = mesh.primal.get_surface(idx_self)

            equations.extend(
                continuity_element_1_forms(
                    element_tree,
                    cont_indices_edges,
                    idx_neighbour.index,
                    idx_self.index,
                    find_boundary_id(s_other, il),
                    find_boundary_id(s_self, il),
                )
            )
    t1 = perf_counter()
    print(f"Continuity of 1-forms: {t1 - t0} seconds.")

    t0 = perf_counter()
    # Continuity of 0-forms on the non-corner DoFs
    if cont_indices_nodes and np.any(element_tree.orders > 1):
        for il in range(mesh.dual.n_lines):
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end

            if not idx_neighbour or not idx_self:
                continue

            s_other = mesh.primal.get_surface(idx_neighbour)
            s_self = mesh.primal.get_surface(idx_self)

            equations.extend(
                continuity_0_forms_inner(
                    element_tree,
                    cont_indices_nodes,
                    find_boundary_id(s_other, il),
                    find_boundary_id(s_self, il),
                    idx_neighbour.index,
                    idx_self.index,
                )
            )

    # Continuity of 0-forms on the corner DoFs
    if cont_indices_nodes:
        for i_surf in range(mesh.dual.n_surfaces):
            dual_surface = mesh.dual.get_surface(i_surf + 1)
            closed = True
            valid: list[int] = list()
            for i_ln in range(len(dual_surface)):
                id_line = dual_surface[i_ln]
                dual_line = mesh.dual.get_line(id_line)
                idx_neighbour = dual_line.begin
                idx_self = dual_line.end
                if not idx_neighbour or not idx_self:
                    closed = False
                else:
                    valid.append(i_ln)

            if closed:
                valid.pop()

            for i_ln in valid:
                id_line = dual_surface[i_ln]
                dual_line = mesh.dual.get_line(id_line)
                idx_neighbour = dual_line.begin
                idx_self = dual_line.end

                if not idx_neighbour or not idx_self:
                    continue

                s_other = mesh.primal.get_surface(idx_neighbour)
                s_self = mesh.primal.get_surface(idx_self)

                equations.extend(
                    continuity_0_form_corner(
                        element_tree,
                        cont_indices_nodes,
                        find_boundary_id(s_other, id_line.index),
                        find_boundary_id(s_self, id_line.index),
                        idx_neighbour.index,
                        idx_self.index,
                    )
                )

    t1 = perf_counter()
    print(f"Continuity of 0-forms: {t1 - t0} seconds.")

    # Strong boundary conditions
    if boundaray_conditions is not None:
        set_nodes: set[int] = set()
        for bc in boundaray_conditions:
            self_var_offset = element_tree.dof_offsets[
                system.unknown_forms.index(bc.form)
            ]

            for idx in bc.indices:
                dof_offsets: npt.NDArray[np.integer] | None = None
                surf_id: GeoID | None = None
                x0: float
                x1: float
                y0: float
                y1: float

                dual_line = mesh.dual.get_line(idx + 1)
                if dual_line.begin and dual_line.end:
                    raise ValueError(
                        f"Boundary condition for {bc.form} was specified for"
                        f" line {idx}, which is not on the boundary."
                    )
                surf_id = dual_line.begin if dual_line.begin else dual_line.end
                primal_surface = mesh.primal.get_surface(surf_id)
                assert len(primal_surface) == 4

                for i_side in range(4):
                    if primal_surface[i_side].index == idx:
                        if bc.form.order == 0:
                            dof_offsets = element_tree.element_node_dofs(
                                surf_id.index, i_side
                            )
                        else:
                            dof_offsets = element_tree.element_edge_dofs(
                                surf_id.index, i_side
                            )
                        break
                assert i_side != 4
                assert dof_offsets is not None
                assert surf_id is not None

                dof_offsets = np.astype(
                    dof_offsets
                    + self_var_offset[surf_id.index]
                    + element_tree.element_offsets[surf_id.index],
                    np.uint32,
                )
                assert dof_offsets is not None

                primal_line = mesh.primal.get_line(primal_surface[i_side])
                x0, y0 = mesh.positions[primal_line.begin.index, :]
                x1, y1 = mesh.positions[primal_line.end.index, :]

                elem_cache = cache[element_tree.orders[surf_id.index]]
                comp_nodes = elem_cache.nodes_1d
                xv = (x1 + x0) / 2 + (x1 - x0) / 2 * comp_nodes
                yv = (y1 + y0) / 2 + (y1 - y0) / 2 * comp_nodes

                vals = np.empty_like(dof_offsets, np.float64)
                if bc.form.order == 0:
                    vals[:] = bc.func(xv, yv)

                    if primal_line.begin.index in set_nodes:
                        vals = vals[1:]
                        dof_offsets = dof_offsets[1:]
                    else:
                        set_nodes.add(primal_line.begin.index)

                    if primal_line.end.index in set_nodes:
                        vals = vals[:-1]
                        dof_offsets = dof_offsets[:-1]
                    else:
                        set_nodes.add(primal_line.end.index)

                elif bc.form.order == 1:
                    # TODO: this might be more efficiently done as some sort of projection
                    lnds = elem_cache.int_nodes_1d
                    wnds = elem_cache.int_weights_1d
                    for i in range(bc.form.order):
                        xc = (xv[i + 1] + xv[i]) / 2 + (xv[i + 1] - xv[i]) / 2 * lnds
                        yc = (yv[i + 1] + yv[i]) / 2 + (yv[i + 1] - yv[i]) / 2 * lnds
                        dx = (xv[i + 1] - xv[i]) / 2
                        dy = (yv[i + 1] - yv[i]) / 2
                        if i_side == 0:
                            normal = np.array((-dy, dx))
                        elif i_side == 1:
                            normal = np.array((dy, -dx))
                        elif i_side == 2:
                            normal = np.array((dy, -dx))
                        elif i_side == 3:
                            normal = np.array((-dy, dx))
                        else:
                            assert False
                        fvals = bc.func(xc, yc)
                        fvals = fvals[..., 0] * normal[0] + fvals[..., 1] * normal[1]
                        vals[i] = np.sum(fvals * wnds)
                else:
                    assert False

                assert vals.size == dof_offsets.size
                for r, v in zip(dof_offsets, vals, strict=True):
                    equations.append(ConstraintEquation((r,), (1,), v))

    # Weak boundary conditions
    for eq in system.equations:
        rhs = eq.right
        for c, kp in rhs.pairs:
            if not isinstance(kp, KBoundaryProjection) or kp.func is None or c == 0:
                continue
            w_form = kp.weight.base_form
            edges = mesh.boundary_indices
            if w_form in strong_indices:
                edges = np.astype(
                    np.setdiff1d(edges, strong_indices[w_form]),  # type: ignore
                    np.int32,
                    copy=False,
                )
            if edges.size == 0:
                continue
            for edge in edges:
                dual_line = mesh.dual.get_line(edge + 1)
                primal_line = mesh.primal.get_line(edge + 1)
                if dual_line.begin:
                    id_surf = dual_line.begin
                elif dual_line.end:
                    id_surf = dual_line.end
                else:
                    assert False
                e = leaf_elements[id_surf.index]
                basis_cache = cache[e.order]
                primal_surface = mesh.primal.get_surface(id_surf)
                ndir, p0, p1 = endpoints_from_line(e, primal_surface, edge)
                dx = (p1[0] - p0[0]) / 2
                xv = (p1[0] + p0[0]) / 2 + dx * basis_cache.int_nodes_1d
                dy = (p1[1] - p0[1]) / 2
                yv = (p1[1] + p0[1]) / 2 + dy * basis_cache.int_nodes_1d
                f_vals = kp.func(xv, yv)
                if w_form.order == 0:
                    # dofs = node_dof_indices_from_line(e, primal_surface, edge)
                    dofs = element_tree.element_node_dofs(
                        id_surf.index, find_boundary_id(primal_surface, edge)
                    )
                    # Tangental integral of function with the 0 basis
                    basis = basis_cache.nodal_1d
                    f_vals = (
                        f_vals[..., 0] * dx + f_vals[..., 1] * dy
                    ) * basis_cache.int_weights_1d

                elif w_form.order:
                    # dofs = edge_dof_indices_from_line(e, primal_surface, edge)
                    dofs = element_tree.element_edge_dofs(
                        id_surf.index, find_boundary_id(primal_surface, edge)
                    )
                    # Integral with the normal basis
                    basis = basis_cache.edge_1d
                    f_vals *= basis_cache.int_weights_1d * ndir  # * np.hypot(dx, dy)

                else:
                    assert False
                dofs = (
                    dofs
                    + element_tree.dof_offsets[system.unknown_forms.index(w_form)][
                        id_surf.index
                    ]
                )
                vals = c * np.sum(f_vals[..., None] * basis, axis=0)
                element_vectors[id_surf.index][dofs] += vals

    sys_mat = sp.block_diag(element_matrix)
    if equations:
        lag_rows: list[npt.NDArray[np.uint32]] = list()
        lag_cols: list[npt.NDArray[np.uint32]] = list()
        lag_vals: list[npt.NDArray[np.float64]] = list()
        lag_rhs: list[np.float64] = list()

        for i, lag_eq in enumerate(equations):
            print(lag_eq)
            lag_rows.append(np.full_like(lag_eq.indices, i + element_tree.n_dof))
            lag_cols.append(lag_eq.indices)
            lag_vals.append(lag_eq.values)
            lag_rhs.append(lag_eq.rhs)

        mat_rows = np.concatenate(lag_rows + lag_cols, dtype=int)
        mat_cols = np.concatenate(lag_cols + lag_rows, dtype=int)
        mat_vals = np.concatenate(lag_vals + lag_vals)

        lagrange = sp.coo_array((mat_vals, (mat_rows, mat_cols)), dtype=np.float64)
        element_vectors.append(np.array(lag_rhs, np.float64))
        # Make the big matrix
        sys_mat.resize(lagrange.shape)

        # Create the system matrix and vector
        matrix = sp.csr_array(sys_mat + lagrange)
    else:
        matrix = sp.csr_array(sys_mat)
    vector = np.concatenate(element_vectors, dtype=np.float64)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.spy(matrix)
    # plt.show()

    # exit()
    # Solve the system
    solution = sla.spsolve(matrix, vector)
    # print(solution)

    xvals: list[npt.NDArray[np.float64]] = list()
    yvals: list[npt.NDArray[np.float64]] = list()

    # Prepare to build up the 1D Splines
    build: dict[kform.KFormUnknown, list[npt.NDArray[np.float64]]] = {
        form: [] for form in system.unknown_forms
    }

    node_array: list[npt.NDArray[np.int32]] = list()
    offset_nodes = 0
    # Loop over element
    for ie, elm in zip(element_tree.leave_indices, leaf_elements):
        # Extract element DoFs
        element_dofs = solution[element_tree.element_offsets[ie] :]
        recon_nodes_1d = cache[elm.order].nodes_1d
        ordering = Element2D.vtk_lagrange_ordering(elm.order) + offset_nodes
        node_array.append(np.concatenate(((ordering.size,), ordering)))
        offset_nodes += ordering.size

        ex = elm.poly_x(recon_nodes_1d[None, :], recon_nodes_1d[:, None])
        ey = elm.poly_y(recon_nodes_1d[None, :], recon_nodes_1d[:, None])

        xvals.append(ex.flatten())
        yvals.append(ey.flatten())

        # Loop over each of the primal forms
        for idx, form in enumerate(system.unknown_forms):
            form_offset = element_tree.dof_offsets[idx][ie]
            form_offset_end = element_tree.dof_offsets[idx + 1][ie]
            form_dofs = element_dofs[form_offset:form_offset_end]
            if not form.is_primal:
                mass: npt.NDArray[np.float64]
                if form.order == 0:
                    mass = elm.mass_matrix_surface(cache[elm.order])
                elif form.order == 1:
                    mass = elm.mass_matrix_edge(cache[elm.order])
                elif form.order == 2:
                    mass = elm.mass_matrix_node(cache[elm.order])
                else:
                    assert False
                form_dofs = np.linalg.solve(mass, form_dofs)
            # Reconstruct unknown
            recon_v = elm.reconstruct(
                form.order, form_dofs, recon_nodes_1d[None, :], recon_nodes_1d[:, None]
            )
            shape = (-1, 2) if form.order == 1 else (-1,)
            build[form].append(np.reshape(recon_v, shape))

    out: dict[kform.KFormUnknown, npt.NDArray[np.float64]] = dict()

    x = np.concatenate(xvals)
    y = np.concatenate(yvals)

    grid = pv.UnstructuredGrid(
        np.concatenate(node_array),
        np.full(len(node_array), pv.CellType.LAGRANGE_QUADRILATERAL),
        np.stack((x, y, np.zeros_like(x)), axis=-1),
    )

    grid.cell_data["order"] = [e.order for e in leaf_elements]

    # Build the outputs
    for form in build:
        vf = np.concatenate(build[form], axis=0, dtype=np.float64)
        out[form] = vf
        grid.point_data[form.label] = vf

    return (x, y, out, grid)


def continuity_0_form_corner(
    element_tree: ElementTree,
    cont_indices: list[int],
    side_other: int,
    side_self: int,
    i_other: int,
    i_self: int,
) -> list[ConstraintEquation]:
    """Generate equations for 0-form continuity between elements for a corner.

    Parameters
    ----------
    element_tree : ElementTree
        Element structure for which to generate these equations.
    cont_indices : list[int]
        List of indices of forms for which this continuity should be applied for.
    i_other : int
        Index of the second element which to connect.
    i_self : int
        Index of the first element to connect.
    side_other : int
        Index of the side of the second element which is connected.
    side_self : int
        Index of the side of the first element which is connected.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 0 forms at a corner.
    """
    equations: list[ConstraintEquation] = list()
    offset_self = element_tree.element_offsets[i_self]
    offset_other = element_tree.element_offsets[i_other]
    dofs_other = element_tree.element_node_dofs(i_other, side_other)[-1]
    ds = element_tree.element_node_dofs(i_self, side_self)[0]

    for var_idx in cont_indices:
        # Left one is from the first DoF of that variable
        self_var_offset = element_tree.dof_offsets[var_idx][i_self]
        # Right one is from the first DoF of that variable
        other_var_offset = element_tree.dof_offsets[var_idx][i_other]

        col_off_self = offset_self + self_var_offset
        col_off_other = offset_other + other_var_offset

        equations.append(
            ConstraintEquation(
                (col_off_self + ds, col_off_other + dofs_other), (+1, -1), 0.0
            )
        )
    return equations


def continuity_0_forms_inner(
    element_tree: ElementTree,
    cont_indices_nodes: list[int],
    side_other: int,
    side_self: int,
    i_other: int,
    i_self: int,
):
    """Generate equations for 0-form continuity between elements with no corners.

    Parameters
    ----------
    element_tree : ElementTree
        Element structure for which to generate these equations.
    cont_indices : list[int]
        List of indices of forms for which this continuity should be applied for.
    i_other : int
        Index of the second element which to connect.
    i_self : int
        Index of the first element to connect.
    side_other : int
        Index of the side of the second element which is connected.
    side_self : int
        Index of the side of the first element which is connected.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 0 forms without the
        corner.
    """
    equations: list[ConstraintEquation] = list()
    offset_self = element_tree.element_offsets[i_self]
    offset_other = element_tree.element_offsets[i_other]
    base_other_dofs = np.flip(element_tree.element_node_dofs(i_other, side_other))
    base_self_dofs = element_tree.element_node_dofs(i_self, side_self)
    order_self = element_tree.orders[i_self]
    order_other = element_tree.orders[i_other]

    for var_idx in cont_indices_nodes:
        self_var_offset = element_tree.dof_offsets[var_idx][i_self]
        other_var_offset = element_tree.dof_offsets[var_idx][i_other]

        col_off_self = offset_self + self_var_offset
        col_off_other = offset_other + other_var_offset

        dofs_other = base_other_dofs + col_off_other

        ds = base_self_dofs + col_off_self
        if order_self == order_other:
            dofs_other = dofs_other[1:-1]
            ds = ds[1:-1]
            assert dofs_other.size == ds.size

            for v1, v2 in zip(ds, dofs_other, strict=True):
                equations.append(ConstraintEquation((v1, v2), (+1, -1), 0.0))

        else:
            if order_self > order_other:
                order_high = int(order_self)
                order_low = int(order_other)
                dofs_high = ds
                dofs_low = dofs_other
            else:
                order_low = int(order_self)
                order_high = int(order_other)
                dofs_high = dofs_other
                dofs_low = ds

            dofs_high = dofs_high[1:-1]

            coeffs_0, _ = continuity_matrices(order_high, order_low)

            for i_h, v_h in zip(range(order_high - 1), dofs_high, strict=True):
                coefficients = coeffs_0[i_h + 1, ...]
                equations.append(
                    ConstraintEquation(
                        np.concatenate(((v_h,), dofs_low)),
                        np.concatenate(((-1,), coefficients)),
                        0.0,
                    )
                )

    return equations


def continuity_element_1_forms(
    element_tree: ElementTree,
    cont_indices: list[int],
    i_other: int,
    i_self: int,
    side_other: int,
    side_self: int,
) -> list[ConstraintEquation]:
    """Generate equations for 1-form continuity between elements.

    Parameters
    ----------
    element_tree : ElementTree
        Element structure for which to generate these equations.
    cont_indices : list[int]
        List of indices of forms for which this continuity should be applied for.
    i_other : int
        Index of the second element which to connect.
    i_self : int
        Index of the first element to connect.
    side_other : int
        Index of the side of the second element which is connected.
    side_self : int
        Index of the side of the first element which is connected.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 1 forms.
    """
    equations: list[ConstraintEquation] = list()
    offset_self = element_tree.element_offsets[i_self]
    offset_other = element_tree.element_offsets[i_other]
    base_other_dofs = np.flip(element_tree.element_edge_dofs(i_other, side_other))
    base_self_dofs = element_tree.element_edge_dofs(i_self, side_self)
    order_self = element_tree.orders[i_self]
    order_other = element_tree.orders[i_other]

    for var_idx in cont_indices:
        self_var_offset = element_tree.dof_offsets[var_idx][i_self]
        other_var_offset = element_tree.dof_offsets[var_idx][i_other]

        col_off_self = offset_self + self_var_offset
        col_off_other = offset_other + other_var_offset

        dofs_other = base_other_dofs + col_off_other
        ds = base_self_dofs + col_off_self

        if order_self == order_other:
            assert base_other_dofs.size == base_self_dofs.size
            for v1, v2 in zip(ds, dofs_other, strict=True):
                equations.append(ConstraintEquation((v1, v2), (+1, -1), 0.0))

        else:
            if order_self > order_other:
                order_high = int(order_self)
                order_low = int(order_other)
                dofs_high = ds
                dofs_low = dofs_other
            else:
                order_low = int(order_self)
                order_high = int(order_other)
                dofs_high = dofs_other
                dofs_low = ds

            _, coeffs_1 = continuity_matrices(order_high, order_low)

            for i_h, v_h in zip(range(order_high), dofs_high, strict=True):
                coefficients = coeffs_1[i_h, ...]

                equations.append(
                    ConstraintEquation(
                        np.concatenate(((v_h,), dofs_low)),
                        np.concatenate(((-1,), coefficients)),
                        0.0,
                    )
                )

    return equations


def continuity_parent_child_nodes(
    element_tree: ElementTree,
    cont_indices: list[int],
    i_parent: int,
    i_child: int,
    i_boundary: int,
    flipped: bool,
) -> list[ConstraintEquation]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    element_tree : ElementTree
        Element tree for which these equations should be generated.
    cont_indices : list[int]
        Indices of all 0-forms for which this should be applied.
    i_parent : int
        Index of the parent element.
    i_child : int
        Index of the child element.
    i_boundary : int
        Index of the boundary which is to be connected.
    flipped : bool
        Determines if the child is on the second half of the boundary instead of the
        first. This in practice means that coefficients in equations must be flipped.

    Returns
    -------
    list of ConstraintEquation
        Equations which enforce continuity of the 0-forms on the boundary of a parent and
        a child.
    """
    dofs_parent = (
        element_tree.element_node_dofs(i_parent, i_boundary)
        + element_tree.element_offsets[i_parent]
    )
    dofs_child = (
        element_tree.element_node_dofs(i_child, i_boundary)
        + element_tree.element_offsets[i_child]
    )
    coeff_0, _ = continuity_child_matrices(
        element_tree.orders[i_child], element_tree.orders[i_parent]
    )
    if flipped:
        coeff_0 = np.flip(coeff_0, axis=0)
        coeff_0 = np.flip(coeff_0, axis=1)
        # Only do the corner on non-flipped, so
        # that we do not double constraints for it.
        dofs_child = dofs_child[:-1]
        coeff_0 = coeff_0[:-1, :]

    equations: list[ConstraintEquation] = list()
    for var_idx in cont_indices:
        var_parent_offset = element_tree.dof_offsets[var_idx][i_parent]
        var_child_offset = element_tree.dof_offsets[var_idx][i_child]

        dp = var_parent_offset + dofs_parent
        dc = var_child_offset + dofs_child

        for i_c, v_c in enumerate(dc):
            coeffs = coeff_0[i_c, :]
            equations.append(
                ConstraintEquation(
                    np.concatenate(((v_c,), dp)), np.concatenate(((-1,), coeffs)), 0.0
                )
            )

    return equations


def continuity_parent_child_edges(
    element_tree: ElementTree,
    cont_indices: list[int],
    i_parent: int,
    i_child: int,
    i_boundary: int,
    flipped: bool,
) -> list[ConstraintEquation]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    element_tree : ElementTree
        Element tree for which these equations should be generated.
    cont_indices : list[int]
        Indices of all 1-forms for which this should be applied.
    i_parent : int
        Index of the parent element.
    i_child : int
        Index of the child element.
    i_boundary : int
        Index of the boundary which is to be connected.
    flipped : bool
        Determines if the child is on the second half of the boundary instead of the
        first. This in practice means that coefficients in equations must be flipped.

    Returns
    -------
    list of ConstraintEquation
        Equations which enforce continuity of the 1-forms on the boundary of a parent and
        a child.
    """
    dofs_parent = (
        element_tree.element_edge_dofs(i_parent, i_boundary)
        + element_tree.element_offsets[i_parent]
    )
    dofs_child = (
        element_tree.element_edge_dofs(i_child, i_boundary)
        + element_tree.element_offsets[i_child]
    )
    _, coeff_1 = continuity_child_matrices(
        element_tree.orders[i_child], element_tree.orders[i_parent]
    )
    if flipped:
        coeff_1 = np.flip(coeff_1, axis=0)
        coeff_1 = np.flip(coeff_1, axis=1)

    equations: list[ConstraintEquation] = list()
    for var_idx in cont_indices:
        var_parent_offset = element_tree.dof_offsets[var_idx][i_parent]
        var_child_offset = element_tree.dof_offsets[var_idx][i_child]

        dp = var_parent_offset + dofs_parent
        dc = var_child_offset + dofs_child

        for i_c, v_c in enumerate(dc):
            coeffs = coeff_1[i_c, :]
            equations.append(
                ConstraintEquation(
                    np.concatenate(((v_c,), dp)), np.concatenate(((-1,), coeffs)), 0.0
                )
            )

    return equations
