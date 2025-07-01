"""File containing implementation of boundary conditions.

For now the only one really used anywhere is the :class:`BoundaryCondition2DSteady`,
but perhaps in the future, for unsteady problems unsteady BCs may be used.

So far the intention is to use these to prescribe strong Dirichelt or Neumann conditions,
while weak boundary conditions are introduced in the equation as boundary integral terms.
That simplifies evaulation makes tracking all of these much simpler.

So far there is no plans to introduce any other types of strong boundary conditions,
though based on what is already supported, it would not be too much of a stretch to
add support for prescribing arbitrary relations on the boundary.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mfv2d.element import (
    ElementCollection,
    ElementConstraint,
    FixedElementArray,
    _element_node_children_on_side,
)
from mfv2d.kform import (
    Function2D,
    KBoundaryProjection,
    KFormUnknown,
    KSum,
    UnknownFormOrder,
    UnknownOrderings,
)
from mfv2d.mimetic2d import ElementSide, FemCache, Mesh2D, find_surface_boundary_id_line


@dataclass(frozen=True, init=False)
class BoundaryCondition2D:
    """Base class for 2D boundary conditions."""

    form: KFormUnknown
    indices: npt.NDArray[np.uint64]

    def __init__(self, form: KFormUnknown, indices: npt.ArrayLike) -> None:
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "indices", np.array(indices, np.uint64))
        if self.indices.ndim != 1:
            raise ValueError("Indices array is not a 1D array.")
        object.__setattr__(self, "indices", np.unique(self.indices))


@dataclass(frozen=True)
class BoundaryCondition2DSteady(BoundaryCondition2D):
    """Boundary condition for a 2D problem with no time dependence.

    These boundary conditions specifiy values of differential forms on the
    the given indices directly. These are enforced "strongly" by adding
    a Lagrange multiplier to the system.

    Parameters
    ----------
    form : KFormUnknown
        Form for which the value is to be prescribed.

    indices : array_like
        One dimensional array of edges on which this is prescribed.
        TODO: check if 0-based on 1-based

    func : (array, array) -> array_like
        Function that can be evaluated to obtain values of differential forms
        at those points. For 1-froms it should return an array_like with an
        extra last dimension, which contains the two vector components.
    """

    func: Function2D

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Function2D,
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)


@dataclass(init=False, frozen=True)
class BoundaryCondition2DUnsteady(BoundaryCondition2D):
    """Unsteady boundary condition for a 2D problem."""

    func: Function2D

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]],
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)


def _element_weak_boundary_condition(
    elements: ElementCollection,
    ie: int,
    side: ElementSide,
    unknown_orders: UnknownOrderings,
    unknown_index: int,
    dof_offsets: FixedElementArray[np.uint32],
    weak_terms: Sequence[tuple[float, KBoundaryProjection]],
    basis_cache: FemCache,
) -> tuple[ElementConstraint, ...]:
    """Determine boundary conditions given an element and a side."""
    children = elements.child_array[ie]

    if children.size != 0:
        # Node, has children
        current: tuple[ElementConstraint, ...] = tuple()
        if side == ElementSide.SIDE_LEFT or side == ElementSide.SIDE_BOTTOM:
            current += _element_weak_boundary_condition(
                elements,
                int(children[0]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                weak_terms,
                basis_cache,
            )
        if side == ElementSide.SIDE_BOTTOM or side == ElementSide.SIDE_RIGHT:
            current += _element_weak_boundary_condition(
                elements,
                int(children[1]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                weak_terms,
                basis_cache,
            )
        if side == ElementSide.SIDE_RIGHT or side == ElementSide.SIDE_TOP:
            current += _element_weak_boundary_condition(
                elements,
                int(children[2]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                weak_terms,
                basis_cache,
            )
        if side == ElementSide.SIDE_TOP or side == ElementSide.SIDE_LEFT:
            current += _element_weak_boundary_condition(
                elements,
                int(children[3]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                weak_terms,
                basis_cache,
            )

        return current

    side_order = elements.get_element_order_on_side(ie, side)

    basis_1d = basis_cache.get_basis1d(side_order)
    ndir = 2 * ((side.value & 2) >> 1) - 1
    i0 = side.value - 1
    i1 = side.value & 3
    corners = elements.corners_array[ie]
    p0: tuple[float, float] = corners[i0]
    p1: tuple[float, float] = corners[i1]
    dx = (p1[0] - p0[0]) / 2
    xv = (p1[0] + p0[0]) / 2 + dx * basis_1d.rule.nodes
    dy = (p1[1] - p0[1]) / 2
    yv = (p1[1] + p0[1]) / 2 + dy * basis_1d.rule.nodes
    form_order = unknown_orders.form_orders[unknown_index]
    dofs = elements.get_boundary_dofs(ie, form_order, side)
    dofs = dofs + dof_offsets[ie][unknown_index]
    vals = np.zeros_like(dofs, np.float64)

    for k, bp in weak_terms:
        func = bp.func
        # We check that bp.func is not None before calling this.
        assert func is not None
        f_vals = np.asarray(func(xv, yv), np.float64, copy=False)
        if form_order == UnknownFormOrder.FORM_ORDER_0:
            # Tangental integral of function with the 0 basis
            basis = basis_1d.node
            f_vals = -(f_vals[..., 0] * dx + f_vals[..., 1] * dy) * basis_1d.rule.weights

        elif form_order == UnknownFormOrder.FORM_ORDER_1:
            # Integral with the normal basis
            basis = basis_1d.edge
            f_vals *= -basis_1d.rule.weights * ndir

        else:
            raise ValueError(f"Unknown/Invalid weak form order {form_order=}.")

        vals[:] += np.sum(f_vals[None, ...] * basis, axis=1) * k

    return (ElementConstraint(ie, dofs, vals),)


def _element_strong_boundary_condition(
    elements: ElementCollection,
    ie: int,
    side: ElementSide,
    unknown_orders: UnknownOrderings,
    unknown_index: int,
    dof_offsets: FixedElementArray[np.uint32],
    strong_bc: BoundaryCondition2DSteady,
    basis_cache: FemCache,
    skip_first: bool,
    skip_last: bool,
) -> tuple[ElementConstraint, ...]:
    """Determine boundary conditions given an element and a side."""
    children = elements.child_array[ie]
    if children.size != 0:
        # Node, has children
        c1, c2 = _element_node_children_on_side(side, children)

        return _element_strong_boundary_condition(
            elements,
            c1,
            side,
            unknown_orders,
            unknown_index,
            dof_offsets,
            strong_bc,
            basis_cache,
            skip_first,
            False,
        ) + _element_strong_boundary_condition(
            elements,
            c2,
            side,
            unknown_orders,
            unknown_index,
            dof_offsets,
            strong_bc,
            basis_cache,
            False,
            skip_last,
        )

    side_order = elements.get_element_order_on_side(ie, side)

    basis_1d = basis_cache.get_basis1d(side_order)
    ndir = 2 * ((side.value & 2) >> 1) - 1
    i0 = side.value - 1
    i1 = side.value & 3
    corners = elements.corners_array[ie]
    p0: tuple[float, float] = corners[i0]
    p1: tuple[float, float] = corners[i1]
    # ndir, p0, p1 = _endpoints_from_line(e, i_side)
    dx = (p1[0] - p0[0]) / 2
    dy = (p1[1] - p0[1]) / 2
    xv = np.astype((p1[0] + p0[0]) / 2 + dx * basis_1d.roots, np.float64, copy=False)
    yv = np.astype((p1[1] + p0[1]) / 2 + dy * basis_1d.roots, np.float64, copy=False)
    form_order = unknown_orders.form_orders[unknown_index]
    dofs = elements.get_boundary_dofs(ie, form_order, side)
    dofs = dofs + dof_offsets[ie][unknown_index]
    vals = np.zeros_like(dofs, np.float64)

    if form_order == UnknownFormOrder.FORM_ORDER_0:
        vals[:] = strong_bc.func(xv, yv)

        if skip_first:
            vals = vals[1:]
            dofs = dofs[1:]

        if skip_last:
            vals = vals[:-1]
            dofs = dofs[:-1]
        if len(vals) == 0:
            assert len(dofs) == 0
            return tuple()

    elif form_order == UnknownFormOrder.FORM_ORDER_1:
        # TODO: this might be more efficiently done as some sort of projection
        lnds = basis_1d.rule.nodes
        wnds = basis_1d.rule.weights
        for i in range(side_order):
            xc = (xv[i + 1] + xv[i]) / 2 + (xv[i + 1] - xv[i]) / 2 * lnds
            yc = (yv[i + 1] + yv[i]) / 2 + (yv[i + 1] - yv[i]) / 2 * lnds
            dx = (xv[i + 1] - xv[i]) / 2
            dy = (yv[i + 1] - yv[i]) / 2
            normal = ndir * np.array((dy, -dx))
            fvals = np.asarray(strong_bc.func(xc, yc), np.float64, copy=None)
            fvals = fvals[..., 0] * normal[0] + fvals[..., 1] * normal[1]
            vals[i] = np.sum(fvals * wnds)
    else:
        assert False

    # TODO: If good, replace old way with this
    # xv = np.astype((p1[0] + p0[0]) / 2 + dx * basis_1d.rule.nodes, np.float64,
    # copy=False)
    # yv = np.astype((p1[1] + p0[1]) / 2 + dy * basis_1d.rule.nodes, np.float64,
    # copy=False)
    # func_values = np.asarray(strong_bc.func(xv, yv), np.float64, copy=None)

    # if form_order == UnknownFormOrder.FORM_ORDER_0:
    #     weights = basis_1d.rule.weights
    #     basis = basis_1d.node
    #     inverse = basis_cache.get_mass_inverse_1d_node(basis_1d.order)
    #     new_vals = inverse @ np.sum(
    #         func_values[None, :] * weights[None, :] * basis, axis=1
    #     )

    #     # TODO: when replacing old, do not forget to also do this with DoFs
    #     # if skip_first:
    #     #     new_vals = new_vals[1:]

    #     # if skip_last:
    #     #     new_vals = new_vals[:-1]

    # elif form_order == UnknownFormOrder.FORM_ORDER_1:
    #     weights = basis_1d.rule.weights
    #     basis = basis_1d.edge
    #     inverse = basis_cache.get_mass_inverse_1d_edge(basis_1d.order)
    #     normal = ndir * np.array((dy, -dx))
    #     new_vals = inverse @ np.sum(
    #         (func_values[:, 0] * normal[0] + func_values[:, 1] * normal[1])[None, :]
    #         * weights[None, :]
    #         * basis,
    #         axis=1,
    #     )
    # else:
    #     assert False

    assert vals.size == dofs.size
    # assert np.allclose(vals, new_vals)
    return (ElementConstraint(ie, dofs, vals),)


def mesh_boundary_conditions(
    evaluatable_terms: Sequence[KSum],
    mesh: Mesh2D,
    unknown_order: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    top_indices: npt.NDArray[np.uint32],
    strong_bcs: Sequence[Sequence[BoundaryCondition2DSteady]],
    basis_cache: FemCache,
) -> tuple[tuple[ElementConstraint, ...], tuple[ElementConstraint, ...]]:
    """Compute boundary condition contributions and constraints.

    Parameters
    ----------
    evaluatable_terms : Sequence of KSum
        Right sides of equations that contain boundary projections to be evaluated. Must
        be ordered according to weights.

    mesh : Mesh2D
        Mesh of the top level elements.

    unknown_order : UnknownOrderings
        Orders of unknown forms.

    elements : ElementCollection
        Collection of elements to use.

    dof_offsets : FixedElementArray[np.uint32]
        Offsets of DoFs within each element.

    top_indices : npt.NDArray[np.uint32]
        Array with corrected indices for top level elements.

    strong_bcs : Sequence of Sequence of BoundaryCondition2DSteady
        Boundary conditions grouped per weight functions and correctly ordered to match
        the order of weight functions in the system.

    caches : MutableMapping of (int, BasisCache)
        Caches which can be used and appended to.

    Returns
    -------
    tuple of ElementConstraint
        Strong boundary conditions in a specific notation. Each of these means
        that for element given by ``ElementConstraint.i_e``, all dofs with
        indices ``ElementConstraint.dofs`` should be constrained to value
        ``ElementConstraint.coeffs``.

    tuple of ElementConstraint
        Weak boundary conditions in a specific notation. Each of these means
        that for element given by ``ElementConstraint.i_e``, all equations with
        indices ``ElementConstraint.dofs`` should have the value
        ``ElementConstraint.coeffs`` added to them.
    """
    i_boundary: int
    w_bcs: list[ElementConstraint] = list()
    s_bcs: list[ElementConstraint] = list()
    projections = [
        [
            (k, v)
            for k, v in weak_term.pairs
            if (type(v) is KBoundaryProjection and v.func is not None)
        ]
        for weak_term in evaluatable_terms
    ]
    del evaluatable_terms
    set_nodes: set[int] = set()

    for i_boundary in mesh.boundary_indices:
        dual_line = mesh.dual.get_line(i_boundary + 1)
        if dual_line.begin:
            id_surf = dual_line.begin
        elif dual_line.end:
            id_surf = dual_line.end
        else:
            raise ValueError("Dual line should be on the boundary.")

        # primal_line = mesh.primal.get_line(i_boundary + 1)
        i_element = top_indices[id_surf.index]
        primal_surface = mesh.primal.get_surface(id_surf)
        i_side = find_surface_boundary_id_line(primal_surface, i_boundary)
        primal_line = mesh.primal.get_line(primal_surface[i_side.value - 1])
        for idx, (weak_term, strong_terms) in enumerate(
            zip(projections, strong_bcs, strict=True)
        ):
            strong_term = None
            for strong in strong_terms:
                if i_boundary in strong.indices:
                    strong_term = strong
                    break
            if strong_term is not None:
                # Strong BC
                p0 = primal_line.begin.index
                p1 = primal_line.end.index

                s_bcs.extend(
                    _element_strong_boundary_condition(
                        elements,
                        i_element,
                        i_side,
                        unknown_order,
                        idx,
                        dof_offsets,
                        strong_term,
                        basis_cache,
                        p0 in set_nodes,
                        p1 in set_nodes,
                    )
                )
                set_nodes |= {p0, p1}

            elif len(weak_term):
                # Weak BC

                w_bcs.extend(
                    _element_weak_boundary_condition(
                        elements,
                        i_element,
                        i_side,
                        unknown_order,
                        idx,
                        dof_offsets,
                        weak_term,
                        basis_cache,
                    )
                )

            else:
                # Strong not given, but also no weak ones.
                pass

    return tuple(s_bcs), tuple(w_bcs)
