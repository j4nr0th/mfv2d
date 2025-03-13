"""Functionality related to creating a full system of equations."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
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
    # compute_element_matrices,
    compute_element_matrices_2,
    #     continuity,
)
from interplib.kforms.eval import _ctranslate, translate_equation
from interplib.kforms.kform import KBoundaryProjection
from interplib.mimetic.mimetic2d import BasisCache, Element2D, Mesh2D, element_rhs
from interplib.prof import PerfTimer


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
    # Orders of each element
    orders: npt.NDArray[np.uint32]
    # True if the element is not a leaf, otherwise false
    children: npt.NDArray[np.bool]
    # Offset of the first DoF of a variable within an element
    dof_offsets: tuple[npt.NDArray[np.uint32], ...]
    # Total number of degrees of freedom
    n_dof: int
    # Total number of degrees of freedom in the leaf nodes
    n_dof_leaves: int
    # Total number of level-zero elements
    n_base_elements: int
    # Indices of elements on the highest level
    top_indices: npt.NDArray[np.intp]
    # Level of the elements
    levels: npt.NDArray[np.uint32]

    def __init__(
        self,
        elements: Sequence[Element2D],
        predicate: None | Callable[[Element2D, int], bool],
        max_levels: int,
        unknowns: Sequence[kform.KFormUnknown],
    ) -> None:
        # Divide the elements as long as predicate is true.
        all_elems: list[Element2D] = list()
        levels: npt.NDArray[np.uint32]
        self.n_base_elements = len(elements)

        def check_refinement(
            pred: Callable[[Element2D, int], bool] | None,
            e: Element2D,
            level: int,
            max_level: int,
        ) -> list[Element2D]:
            """Return element and potentially its children."""
            out = [e]

            if level < max_level and pred is not None and pred(e, level):
                # TODO: have to configure how orders change
                (ebl, ebr), (etl, etr) = e.divide(e.order)
                e.order *= 2
                out += check_refinement(pred, ebl, level + 1, max_level)
                out += check_refinement(pred, ebr, level + 1, max_level)
                out += check_refinement(pred, etl, level + 1, max_level)
                out += check_refinement(pred, etr, level + 1, max_level)

            return out

        # Depth-first refinement
        for e in elements:
            all_elems += check_refinement(predicate, e, 0, max_levels)

        levels = np.array([e.level for e in all_elems], np.uint32)
        self.levels = levels

        self.elements = tuple(all_elems)
        del all_elems

        # Check if elements have children
        n_total = len(self.elements)
        self.children = np.zeros(n_total, np.bool)
        self.children[:-1] = levels[1:] > levels[:-1]

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
        # off = np.pad(np.cumsum(self.dof_offsets[-1][rindices]), (1, 0))
        # self.element_offsets = off[indices]
        self.n_dof_leaves = int(np.sum(self.dof_offsets[-1][~self.children]))
        self.n_dof = int(np.sum(self.dof_offsets[-1]))
        self.top_indices = np.flatnonzero(levels == 0)

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
    def leaf_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of leaf elements."""
        return np.astype(np.nonzero(~self.children)[0], np.uint32)

    @property
    def n_elements(self) -> int:
        """Number of elements in the ElementTree."""
        return len(self.elements)

    def get_children(self, i: int, /) -> tuple[int, int, int, int]:
        """Get children of a non-leaf element.

        Children are in the following order:

        1. bottom left
        2. bottom right
        3. top left
        4. top right
        """
        if not self.children[i]:
            raise ValueError("Leaf element has no children.")
        indices = np.flatnonzero(self.levels[i:] == self.levels[i] + 1)[:4] + i
        assert len(indices) == 4

        return int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])


@dataclass(frozen=True)
class SolutionStatistics:
    """Information about the solution."""

    element_orders: dict[int, int]
    n_total_dofs: int
    n_leaf_dofs: int
    n_lagrange: int
    n_elems: int
    n_leaves: int


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
    # assert nchild >= nparent
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
    refinement_levels: int = 0,
    div_predicate: Callable[[Element2D, int], bool] | None = None,
    *,
    timed: bool = False,
    recon_order: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[kform.KFormUnknown, npt.NDArray[np.float64]],
    pv.UnstructuredGrid,
    SolutionStatistics,
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
    refinement_levels : int, default: 0
        Number of mesh refinement levels which can be done. When zero
        (default value) no refinement is done.
    div_predicate : Callable (Element2D) -> bool
        Callable used to determine if an element should be divided further.
    timed : bool, default: False
        Report time taken for different parts of the code.
    recon_order : int, optional
        When specified, all elements will be reconstructed using this polynomial order.
        Otherwise, they are reconstructed with their own order.

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
        cells. This reconstruction is done on the nodal basis for all unknowns.
    stats : SolutionStatistics
        Statistics about the solution. This can be used for convergence tests or timing.
    """
    base_timer = PerfTimer()
    # Check that inputs make sense.
    strong_boundary_edges: dict[kform.KFormUnknown, list[npt.NDArray[np.uint64]]] = {}
    for primal in system.unknown_forms:
        if primal.order > 2:
            raise ValueError(
                f"Can not solve the system on a 2D mesh, as it contains a {primal.order}"
                "-form."
            )
        strong_boundary_edges[primal] = []

    refinement_levels = int(refinement_levels)
    if refinement_levels < 0:
        raise ValueError(
            f"Can not have less than 0 refinement levels ({refinement_levels} was given)."
        )

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
        refinement_levels,
        system.unknown_forms,
    )
    if timed:
        base_timer.stop("Creating element tree took {} seconds.")
        base_timer.set()

    leaf_elements: list[Element2D] = list(element_tree.iter_leaves())

    # Make element matrices and vectors
    cache: dict[int, BasisCache] = dict()
    for order in np.unique(element_tree.orders):
        cache[int(order)] = BasisCache(int(order), int(order) + 2)

    if recon_order is not None and recon_order not in cache:
        cache[int(recon_order)] = BasisCache(int(recon_order), int(recon_order))

    bytecodes = [translate_equation(eq.left, simplify=True) for eq in system.equations]

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
    c_ser: list[
        tuple[
            int,
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
    ] = list()

    # Release cache element memory. If they will be needed in the future,
    # they will be recomputed, but they consume LOTS of memory
    for o in np.unique(element_tree.orders[~element_tree.children]):
        c_ser.append(cache[o].c_serialization())
        cache[o].clean()

    if timed:
        base_timer.stop("Pre-processing inputs for the C code took {} seconds.")
        base_timer.set()

    element_matrices = {
        int(ileaf): m
        for ileaf, m in zip(
            element_tree.leaf_indices,
            compute_element_matrices_2(
                [f.order for f in system.unknown_forms],
                codes,
                bl,
                br,
                tr,
                tl,
                orde,
                c_ser,
            ),
            strict=True,
        )
    }
    del bl, br, tr, tl, orde, c_ser

    if timed:
        base_timer.stop("Computing element matrices took {} seconds.")
        base_timer.set()

    element_vectors: dict[int, npt.NDArray[np.float64]] = {
        int(ileaf): element_rhs(system, e, cache[e.order])
        for ileaf, e in zip(element_tree.leaf_indices, leaf_elements, strict=True)
    }

    if timed:
        base_timer.stop("Computing the RHS took {} seconds.")
        base_timer.set()

    base_element_offsets = np.zeros(element_tree.n_base_elements + 1, np.uint32)
    matrices: list[sp.coo_array] = list()
    vec: list[npt.NDArray[np.float64]] = list()
    element_begin = np.zeros(
        element_tree.n_elements + 1, np.uint32
    )  # Element beginning offsets
    for i, itop in enumerate(element_tree.top_indices):
        bvals, em, ev = element_matrix(
            cont_indices_edges,
            cont_indices_nodes,
            element_tree,
            itop,
            element_matrices,
            element_vectors,
            element_tree.dof_offsets[-1],
        )
        matrices.append(em)
        vec.append(ev)
        n_bvals = len(bvals)
        element_begin[itop + 1 : itop + n_bvals + 1] = element_begin[itop] + bvals
        base_element_offsets[i + 1] = base_element_offsets[i] + ev.size

    main_mat = sp.block_diag(matrices)
    main_vec = np.concatenate(vec)
    del matrices, vec, element_matrices, element_vectors

    if timed:
        base_timer.stop("Assembling the main matrix took {} seconds.")
        base_timer.set()

    # Apply lagrange multipliers for continuity
    continuity_equations: list[ConstraintEquation] = list()

    # Continuity of 1-forms on top level
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

            i_other = int(element_tree.top_indices[idx_neighbour.index])
            i_self = int(element_tree.top_indices[idx_self.index])
            continuity_equations.extend(
                continuity_element_1_forms(
                    element_tree,
                    cont_indices_edges,
                    i_other,
                    i_self,
                    find_boundary_id(s_other, il),
                    find_boundary_id(s_self, il),
                    base_element_offsets[idx_neighbour.index],
                    base_element_offsets[idx_self.index],
                )
            )

    if timed:
        base_timer.stop("Continuity of 1-forms took {} seconds.")
        base_timer.set()

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

            i_other = int(element_tree.top_indices[idx_neighbour.index])
            i_self = int(element_tree.top_indices[idx_self.index])
            continuity_equations.extend(
                continuity_0_forms_inner(
                    element_tree,
                    cont_indices_nodes,
                    find_boundary_id(s_other, il),
                    find_boundary_id(s_self, il),
                    i_other,
                    i_self,
                    base_element_offsets[idx_neighbour.index],
                    base_element_offsets[idx_self.index],
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

                i_other = int(element_tree.top_indices[idx_neighbour.index])
                i_self = int(element_tree.top_indices[idx_self.index])
                continuity_equations.extend(
                    continuity_0_form_corner(
                        element_tree,
                        cont_indices_nodes,
                        find_boundary_id(s_other, id_line.index),
                        find_boundary_id(s_self, id_line.index),
                        i_other,
                        i_self,
                        base_element_offsets[idx_neighbour.index],
                        base_element_offsets[idx_self.index],
                    )
                )

    if timed:
        base_timer.stop("Continuity of 0-forms took {} seconds.")
        base_timer.set()

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
                i_element = element_tree.top_indices[surf_id.index]

                for i_side in range(4):
                    if primal_surface[i_side].index == idx:
                        if bc.form.order == 0:
                            dof_offsets = element_tree.element_node_dofs(
                                i_element, i_side
                            )
                        else:
                            dof_offsets = element_tree.element_edge_dofs(
                                i_element, i_side
                            )
                        break
                assert i_side != 4
                assert dof_offsets is not None
                assert surf_id is not None

                dof_offsets = np.astype(
                    dof_offsets
                    + self_var_offset[i_element]
                    + base_element_offsets[surf_id.index],
                    np.uint32,
                )
                assert dof_offsets is not None

                primal_line = mesh.primal.get_line(primal_surface[i_side])
                x0, y0 = mesh.positions[primal_line.begin.index, :]
                x1, y1 = mesh.positions[primal_line.end.index, :]

                elem_cache = cache[int(element_tree.orders[i_element])]
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
                    continuity_equations.append(ConstraintEquation((r,), (1,), v))

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
                i_element = element_tree.top_indices[id_surf.index]
                e = leaf_elements[i_element]
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
                        i_element, find_boundary_id(primal_surface, edge)
                    )
                    # Tangental integral of function with the 0 basis
                    basis = basis_cache.nodal_1d
                    f_vals = (
                        f_vals[..., 0] * dx + f_vals[..., 1] * dy
                    ) * basis_cache.int_weights_1d

                elif w_form.order:
                    # dofs = edge_dof_indices_from_line(e, primal_surface, edge)
                    dofs = element_tree.element_edge_dofs(
                        i_element, find_boundary_id(primal_surface, edge)
                    )
                    # Integral with the normal basis
                    basis = basis_cache.edge_1d
                    f_vals *= basis_cache.int_weights_1d * ndir  # * np.hypot(dx, dy)

                else:
                    assert False
                dofs = (
                    dofs
                    + element_tree.dof_offsets[system.unknown_forms.index(w_form)][
                        i_element
                    ]
                )
                vals = c * np.sum(f_vals[..., None] * basis, axis=0)
                # element_vectors[i_element][dofs] += vals
                main_vec[element_begin[i_element] : element_begin[i_element + 1]][
                    dofs
                ] += vals

    if timed:
        base_timer.stop("Boundary conditions took {} seconds.")
        base_timer.set()

    # TODO: Assemble the system matrix

    n_lagrange_eq = len(continuity_equations)

    if continuity_equations:
        lag_rows: list[npt.NDArray[np.uint32]] = list()
        lag_cols: list[npt.NDArray[np.uint32]] = list()
        lag_vals: list[npt.NDArray[np.float64]] = list()
        lag_rhs: list[np.float64] = list()

        for ieq, lag_eq in enumerate(continuity_equations):
            lag_rows.append(np.full_like(lag_eq.indices, ieq))
            lag_cols.append(lag_eq.indices)
            lag_vals.append(lag_eq.values)
            lag_rhs.append(lag_eq.rhs)

        mat_rows = np.concatenate(lag_rows, dtype=int)
        mat_cols = np.concatenate(lag_cols, dtype=int)
        mat_vals = np.concatenate(lag_vals)

        lagrange_mat = sp.csc_array((mat_vals, (mat_rows, mat_cols)), dtype=np.float64)
        lagrange_mat.resize(n_lagrange_eq, element_begin[-1])
        del mat_rows, mat_cols, mat_vals
        main_mat = sp.block_array([[main_mat, lagrange_mat.T], [lagrange_mat, None]])
        del lagrange_mat
        if timed:
            base_timer.stop("Preparing the system took {} seconds.")
            base_timer.set()
        main_vec = np.concatenate((main_vec, lag_rhs))

    main_mat = sp.csc_array(main_mat)
    solution = sla.spsolve(main_mat, main_vec)

    del main_mat, main_vec, continuity_equations

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.spy(matrix)
    # plt.show()

    # exit()

    # np.savetxt("test_mat.dat", matrix.toarray())
    # Solve the system

    if timed:
        base_timer.stop("Solving took {} seconds.")
        base_timer.set()

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
    for ie, elm in zip(element_tree.leaf_indices, leaf_elements):
        # Extract element DoFs
        element_dofs = solution[element_begin[ie] : element_begin[ie + 1]]
        element_order = elm.order if recon_order is None else int(recon_order)
        recon_nodes_1d = cache[element_order].nodes_1d
        ordering = Element2D.vtk_lagrange_ordering(element_order) + offset_nodes
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
                form.order,
                form_dofs,
                recon_nodes_1d[None, :],
                recon_nodes_1d[:, None],
                cache[elm.order],
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

    orders_cnt = np.unique_counts(element_tree.orders)

    if timed:
        base_timer.stop("Reconstruction took {} seconds.")
        base_timer.set()

    stats = SolutionStatistics(
        element_orders={
            int(i1): i2 for i1, i2 in zip(orders_cnt.values, orders_cnt.counts)
        },
        n_total_dofs=element_tree.n_dof,
        n_lagrange=n_lagrange_eq,
        n_elems=len(element_tree.elements),
        n_leaves=len(leaf_elements),
        n_leaf_dofs=element_tree.n_dof_leaves,
    )

    return (x, y, out, grid, stats)


def element_matrix(
    cont_indices_edges: list[int],
    cont_indices_nodes: list[int],
    element_tree: ElementTree,
    i: int,
    element_matrices: dict[int, npt.NDArray[np.float64]],
    element_vecs: dict[int, npt.NDArray[np.float64]],
    element_sizes: npt.NDArray[np.uint32],
) -> tuple[
    npt.NDArray[np.uint32],
    sp.coo_array,
    npt.NDArray[np.float64],
]:
    """Add element matrix of the element with the specified index."""
    # Add offset for the next element
    size = element_sizes[i]
    curr_size = np.array([size], np.uint32)

    # Check if leaf element
    if (
        i + 1 == element_tree.n_elements
        or element_tree.elements[i].level >= element_tree.elements[i + 1].level
    ):
        # Leaf element, meaning only the element matrix is added
        assert i in element_matrices and i in element_vecs
        m = element_matrices[i]
        v = element_vecs[i]
        assert m.shape[0] == m.shape[1] and m.shape[0] == size
        assert m.shape[0] == v.size
        return (curr_size, sp.coo_array(m), v)

    vec: list[npt.NDArray[np.float64]] = list()
    # A non-leaf element, meaning its four children must be found
    vec.append(np.zeros(size))
    # Add the bottom left
    child_indices = element_tree.get_children(i)

    mats: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]

    vecs: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]

    sizes, mats, vecs = zip(
        *(
            element_matrix(
                cont_indices_edges,
                cont_indices_nodes,
                element_tree,
                c_idx,
                element_matrices,
                element_vecs,
                element_sizes,
            )
            for c_idx in child_indices
        )
    )

    for m, v in zip(mats, vecs):
        assert m.shape[0] == m.shape[1] and m.shape[0] and v.size

    vec.extend(vecs)
    offsets = np.pad(np.cumsum([size] + [v.size for v in vecs]), (1, 0))

    # Get the continuity equations
    cont = parent_child_equations(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i,
        *child_indices,
        *offsets[:-1],
    )
    # n_cont = len(cont)

    # lag_mult = np.sum([v.size for v in vecs]) + size

    # mat.resize(mat.shape[0] + n_cont, mat.shape[1] + n_cont)
    vals: list[npt.NDArray[np.float64]] = list()
    rows: list[npt.NDArray[np.uint32]] = list()
    cols: list[npt.NDArray[np.uint32]] = list()
    rhs: list[np.float64] = list()
    for j, eq in enumerate(cont):
        vals.append(eq.values)
        cols.append(eq.indices)
        rows.append(np.full_like(eq.indices, j))
        rhs.append(eq.rhs)

    vv = np.concatenate(vals)
    rv = np.concatenate(rows)
    cv = np.concatenate(cols)

    lag_mat = sp.coo_array((vv, (rv, cv)))
    lag_mat.resize((len(cont), offsets[-1]))
    combined = sp.block_diag([sp.coo_array((size, size)), *mats])

    resulting = sp.block_array([[combined, lag_mat.T], [lag_mat, None]])
    assert isinstance(resulting, sp.coo_array)

    vec.append(np.array(rhs, np.float64))

    size_list = [curr_size]
    for s in sizes:
        size_list.append(s + size_list[-1][-1])
    size_list[-1][-1] += len(cont)

    return np.concatenate(size_list), resulting, np.concatenate(vec)


def add_element_matrix(
    cont_indices_edges: list[int],
    cont_indices_nodes: list[int],
    element_tree: ElementTree,
    i: int,
    element_matrices: list[npt.NDArray[np.float64]],
    element_vecs: list[npt.NDArray[np.float64]],
    element_offsets: npt.NDArray[np.uint32],
    element_sizes: npt.NDArray[np.uint32],
    mat: tuple[
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.uint32]],
        list[npt.NDArray[np.uint32]],
    ],
    vec: list[npt.NDArray[np.float64]],
) -> int:
    """Add element matrix of the element with the specified index."""
    offset = element_offsets[i]
    # Add offset for the next element
    size = element_sizes[i]
    element_offsets[i + 1] = offset + size

    # Check if leaf element
    if (
        i + 1 == element_tree.n_elements
        or element_tree.elements[i].level >= element_tree.elements[i + 1].level
    ):
        # Leaf element, meaning only the element matrix is added
        assert len(element_matrices) and len(element_matrices) == len(element_vecs)
        m = element_matrices.pop()
        v = element_vecs.pop()
        assert m.shape[0] == m.shape[1] and m.shape[0] == size
        idx = offset + np.arange(size)
        # mat.resize(idx[-1] + 1, idx[-1] + 1)
        # mat += sp.coo_array((m.flatten(), (np.repeat(idx, size), np.tile(idx, size))))
        mat[0].append(m.flatten())
        mat[1].append(np.repeat(idx, size))
        mat[2].append(np.tile(idx, size))
        vec.append(v)
        return 1

    # A non-leaf element, meaning its four children must be found
    n = 1
    vec.append(np.zeros(size))
    # Add the bottom left
    i_bl = i + n
    dn = add_element_matrix(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i_bl,
        element_matrices,
        element_vecs,
        element_offsets,
        element_sizes,
        mat,
        vec,
    )
    n += dn
    # Add the bottom right
    i_br = i + n
    dn = add_element_matrix(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i_br,
        element_matrices,
        element_vecs,
        element_offsets,
        element_sizes,
        mat,
        vec,
    )
    n += dn
    # Add the top left
    i_tl = i + n
    dn = add_element_matrix(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i_tl,
        element_matrices,
        element_vecs,
        element_offsets,
        element_sizes,
        mat,
        vec,
    )
    n += dn
    # Add the top right
    i_tr = i + n
    dn = add_element_matrix(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i_tr,
        element_matrices,
        element_vecs,
        element_offsets,
        element_sizes,
        mat,
        vec,
    )
    n += dn
    # Get the continuity equations
    cont = parent_child_equations(
        cont_indices_edges,
        cont_indices_nodes,
        element_tree,
        i,
        i_bl,
        i_br,
        i_tl,
        i_tr,
        element_offsets[i],
        element_offsets[i_bl],
        element_offsets[i_br],
        element_offsets[i_tl],
        element_offsets[i_tr],
    )
    n_cont = len(cont)

    lag_mult = element_offsets[i + n]

    # mat.resize(mat.shape[0] + n_cont, mat.shape[1] + n_cont)
    vals: list[npt.NDArray[np.float64]] = list()
    rows: list[npt.NDArray[np.uint32]] = list()
    cols: list[npt.NDArray[np.uint32]] = list()
    rhs: list[np.float64] = list()
    for j, eq in enumerate(cont):
        vals.append(eq.values)
        cols.append(eq.indices)
        rows.append(np.full_like(eq.indices, lag_mult + j))
        rhs.append(eq.rhs)

    vv = np.concatenate(vals + vals)
    rv = np.concatenate(rows + cols)
    cv = np.concatenate(cols + rows)

    # mat += sp.coo_array((vv, (rv, cv)))
    mat[0].append(vv)
    mat[1].append(rv)
    mat[2].append(cv)
    vec.append(np.array(rhs, np.float64))

    # Next element is further offset due to Lagrange multipliers
    element_offsets[i + n] += n_cont

    return n


def parent_child_equations(
    cont_indices_edges: list[int],
    cont_indices_nodes: list[int],
    element_tree: ElementTree,
    idx_parent: int,
    idx_00: int,
    idx_01: int,
    idx_10: int,
    idx_11: int,
    offset_parent: int,
    offset_00: int,
    offset_01: int,
    offset_10: int,
    offset_11: int,
) -> list[ConstraintEquation]:
    """Create constraint equations for the parent-child and child-child continuity.

    Parameters
    ----------
    const_indices_edges : list of int
        List of 1-form indices for which continuity must be ensured.
    const_indices_nodes : list of int
        List of 0-form indices for which continuity must be ensured.
    element_offsets : array
        Array of offsets of element DoFs.
    element_tree : ElementTree
        Element tree in which the elements are defined.
    idx_parent : int
        Index of the parent element.
    idx_00 : int
        Index of the bottom left child element
    idx_01 : int
        Index of the bottom right child element.
    idx_10 : int
        Index of the top left child element
    idx_11 : int
        Index of the top right child element.
    offset_parent : int
        Offset of the first degree of freedom in the parent element.
    offset_00 : int
        Offset of the first degree of freedom in the bottom left child element
    offset_01 : int
        Offset of the first degree of freedom in the bottom right child element.
    offset_10 : int
        Offset of the first degree of freedom in the top left child element
    offset_11 : int
        Offset of the first degree of freedom in the top right child element.

    Returns
    -------
    list of ConstraintEquation
        List of the constraint equations which ensure continuity between these elements.
    """
    child_child: list[ConstraintEquation] = list()
    parent_child: list[ConstraintEquation] = list()
    if cont_indices_edges:
        # Glue 1-form edges
        child_child += continuity_element_1_forms(
            element_tree, cont_indices_edges, idx_01, idx_00, 3, 1, offset_01, offset_00
        )
        child_child += continuity_element_1_forms(
            element_tree, cont_indices_edges, idx_11, idx_01, 0, 2, offset_11, offset_01
        )
        child_child += continuity_element_1_forms(
            element_tree, cont_indices_edges, idx_10, idx_11, 1, 3, offset_10, offset_11
        )
        child_child += continuity_element_1_forms(
            element_tree, cont_indices_edges, idx_00, idx_10, 2, 0, offset_00, offset_10
        )

        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_00,
            offset_parent,
            offset_00,
            0,
            False,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_01,
            offset_parent,
            offset_01,
            0,
            True,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_01,
            offset_parent,
            offset_01,
            1,
            False,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_11,
            offset_parent,
            offset_11,
            1,
            True,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_11,
            offset_parent,
            offset_11,
            2,
            False,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_10,
            offset_parent,
            offset_10,
            2,
            True,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_10,
            offset_parent,
            offset_10,
            3,
            False,
        )
        parent_child += continuity_parent_child_edges(
            element_tree,
            cont_indices_edges,
            idx_parent,
            idx_00,
            offset_parent,
            offset_00,
            3,
            True,
        )

    if cont_indices_nodes:
        # Glue 0-form edges
        child_child += continuity_0_forms_inner(
            element_tree, cont_indices_nodes, 3, 1, idx_01, idx_00, offset_01, offset_00
        )
        child_child += continuity_0_forms_inner(
            element_tree, cont_indices_nodes, 0, 2, idx_11, idx_01, offset_11, offset_01
        )
        child_child += continuity_0_forms_inner(
            element_tree, cont_indices_nodes, 1, 3, idx_10, idx_11, offset_10, offset_11
        )
        child_child += continuity_0_forms_inner(
            element_tree, cont_indices_nodes, 2, 0, idx_00, idx_10, offset_00, offset_10
        )
        # Glue the corner they all share
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 1, 3, idx_00, idx_01, offset_00, offset_01
        )
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 2, 0, idx_01, idx_11, offset_01, offset_11
        )
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 3, 1, idx_11, idx_10, offset_11, offset_10
        )

        # Glue the child corners too
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 3, 1, idx_01, idx_00, offset_01, offset_00
        )
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 0, 2, idx_11, idx_01, offset_11, offset_01
        )
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 1, 3, idx_10, idx_11, offset_10, offset_11
        )
        child_child += continuity_0_form_corner(
            element_tree, cont_indices_nodes, 2, 0, idx_00, idx_10, offset_00, offset_10
        )

        # Don't add the fourth equation!

        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_00,
            offset_parent,
            offset_00,
            0,
            False,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_01,
            offset_parent,
            offset_01,
            0,
            True,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_01,
            offset_parent,
            offset_01,
            1,
            False,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_11,
            offset_parent,
            offset_11,
            1,
            True,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_11,
            offset_parent,
            offset_11,
            2,
            False,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_10,
            offset_parent,
            offset_10,
            2,
            True,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_10,
            offset_parent,
            offset_10,
            3,
            False,
        )
        parent_child += continuity_parent_child_nodes(
            element_tree,
            cont_indices_nodes,
            idx_parent,
            idx_00,
            offset_parent,
            offset_00,
            3,
            True,
        )

    return child_child + parent_child


def continuity_0_form_corner(
    element_tree: ElementTree,
    cont_indices: list[int],
    side_other: int,
    side_self: int,
    i_other: int,
    i_self: int,
    offset_other: int,
    offset_self: int,
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
    offset_other : int
        Offset of the first degree of freedom in the system for the first element.
    offset_self : int
        Offset of the first degree of freedom in the system for the second element.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 0 forms at a corner.
    """
    equations: list[ConstraintEquation] = list()
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
    offset_other: int,
    offset_self: int,
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
    offset_other : int
        Offset of the first degree of freedom in the system for the first element.
    offset_self : int
        Offset of the first degree of freedom in the system for the second element.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 0 forms without the
        corner.
    """
    equations: list[ConstraintEquation] = list()
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
    offset_other: int,
    offset_self: int,
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
    offset_other : int
        Offset of the first degree of freedom in the system for the first element.
    offset_self : int
        Offset of the first degree of freedom in the system for the second element.

    Returns
    -------
    list of ConstraintEquation
        List of constraint equations, which enforce continuity of 1 forms.
    """
    equations: list[ConstraintEquation] = list()
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
    offset_parent: int,
    offset_child: int,
    i_boundary: int,
    flipped: bool,
) -> list[ConstraintEquation]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    element_tree : ElementTree
        Element tree for which these equations should be generated.
    element_offsets : array
        Array of offsets of element DoFs.
    cont_indices : list[int]
        Indices of all 0-forms for which this should be applied.
    i_parent : int
        Index of the parent element.
    i_child : int
        Index of the child element.
    offset_parent : int
        Offset of the first degree of freedom in parent element.
    offset_child : int
        Offset of the first degree of freedom in child element.
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
    dofs_parent = element_tree.element_node_dofs(i_parent, i_boundary) + offset_parent
    dofs_child = element_tree.element_node_dofs(i_child, i_boundary) + offset_child
    coeff_0, _ = continuity_child_matrices(
        element_tree.orders[i_child], element_tree.orders[i_parent]
    )
    if flipped:
        coeff_0 = np.flip(coeff_0, axis=0)
        coeff_0 = np.flip(coeff_0, axis=1)
        # Only do the corner on non-flipped, so
        # that we do not double constraints for it.
        dofs_child = dofs_child[1:-1]
        coeff_0 = coeff_0[1:-1, :]

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
    offset_parent: int,
    offset_child: int,
    i_boundary: int,
    flipped: bool,
) -> list[ConstraintEquation]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    element_tree : ElementTree
        Element tree for which these equations should be generated.
    element_offsets : array
        Array of offsets of element DoFs.
    cont_indices : list[int]
        Indices of all 1-forms for which this should be applied.
    i_parent : int
        Index of the parent element.
    i_child : int
        Index of the child element.
    offset_parent : int
        Offset of the first degree of freedom in parent element.
    offset_child : int
        Offset of the first degree of freedom in child element.
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
    dofs_parent = element_tree.element_edge_dofs(i_parent, i_boundary) + offset_parent
    dofs_child = element_tree.element_edge_dofs(i_child, i_boundary) + offset_child
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
