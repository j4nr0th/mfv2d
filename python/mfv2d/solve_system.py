"""Implementation of solve functions.

Functions in this file all deal with solving the full system. Examples
of these include the assembly of the global matrix, the application of the
boundary conditions, and computing the right side of the system.
"""

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from functools import cache
from itertools import accumulate
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import (
    Basis2D,
    Surface,
    compute_element_mass_matrix,
    compute_element_matrix,
    compute_element_vector,
    compute_gll,
    lagrange1d,
)
from mfv2d.boundary import BoundaryCondition2DSteady
from mfv2d.element import (
    ElementCollection,
    ElementSide,
    FixedElementArray,
    FlexibleElementArray,
    _compute_element_lagrange_multipliers,
    call_per_element_fix,
    call_per_element_flex,
    call_per_leaf_flex,
    element_dual_dofs,
    poly_x,
    poly_y,
    reconstruct,
)
from mfv2d.eval import CompiledSystem, _CompiledCodeMatrix
from mfv2d.kform import (
    Function2D,
    KBoundaryProjection,
    KElementProjection,
    KExplicit,
    KFormSystem,
    KFormUnknown,
    KSum,
    KWeight,
    UnknownFormOrder,
    UnknownOrderings,
)
from mfv2d.mimetic2d import (
    Element2D,
    ElementLeaf2D,
    FemCache,
    Mesh2D,
    vtk_lagrange_ordering,
)
from mfv2d.progress import ProgressTracker

OrderDivisionFunction = Callable[
    [int, int, int], tuple[int | None, tuple[int, int, int, int]]
]


# TODO: REPLACE
def check_and_refine(
    pred: Callable[[ElementLeaf2D, int], bool] | None,
    order_div: OrderDivisionFunction,
    e: ElementLeaf2D,
    level: int,
    max_level: int,
) -> list[Element2D]:
    """Return element and potentially its children."""
    out: list[Element2D]
    if level < max_level and pred is not None and pred(e, level):
        # TODO: Make this nicer without this stupid Method mumbo jumbo
        parent_order, child_orders = order_div(e.order, level, max_level)
        new_e, ((ebl, ebr), (etl, etr)) = e.divide(*child_orders, parent_order)
        cbl = check_and_refine(pred, order_div, ebl, level + 1, max_level)
        cbr = check_and_refine(pred, order_div, ebr, level + 1, max_level)
        ctl = check_and_refine(pred, order_div, etl, level + 1, max_level)
        ctr = check_and_refine(pred, order_div, etr, level + 1, max_level)
        object.__setattr__(new_e, "child_bl", cbl[0])
        object.__setattr__(new_e, "child_br", cbr[0])
        object.__setattr__(new_e, "child_tl", ctl[0])
        object.__setattr__(new_e, "child_tr", ctr[0])
        out = [new_e] + cbl + cbr + ctr + ctl

    else:
        out = [e]

    return out


def compute_element_vector_fields_nonlin(
    system: KFormSystem,
    element_basis: Basis2D,
    output_basis: Basis2D,
    vector_fields: Sequence[Function2D | KFormUnknown],
    element_corners: npt.NDArray[np.float64],
    unknown_offsets: npt.NDArray[np.uint32],
    solution: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Evaluate vector fields which may be non-linear.

    Parameters
    ----------
    system : KFormSystem
        System for which these vector fields should be computed.
        TODO: REPLACE WITH JUST UNKNOWN FORMS

    element_basis : Basis2D
        Basis functions that the element uses. This needs to match
        with the number of degrees of freedom of the element.

    output_basis : Basis2D
        Basis onto which the result is to be computed.

    vector_fields : Sequence of Function2D or KFormUnknown
        Description of the vector fields. Can be a callable which gives its
        value at a point, or instead it can be an unknown in the system.

    element_corners : (4, 2) array
        Array of the element corner points.

    unknown_offsets : array
        Array with offsets of the degrees of freedom within the element. This
        is used to pick correct degrees of freedom from the element DoFs vector.

    solution : array, optional
        Array of the element degrees of freedom. If not provided, all are assumed
        to be zero instead.

    Returns
    -------
    tuple of array
        Tuple with arrays with values of the vector field at each point of the 2D
        basis integration rules.
    """
    vec_field_lists: list[npt.NDArray[np.float64]] = list()
    # Extract element DoFs

    out_xi = output_basis.basis_xi.rule.nodes[None, :]
    out_eta = output_basis.basis_eta.rule.nodes[:, None]

    for i, vec_fld in enumerate(vector_fields):
        if isinstance(vec_fld, KFormUnknown):
            if solution is not None:
                i_form = system.unknown_forms.index(vec_fld)
                element_dofs = solution
                form_offset = unknown_offsets[i_form]
                form_offset_end = unknown_offsets[i_form + 1]
                form_dofs = element_dofs[form_offset:form_offset_end]
                vf = reconstruct(
                    element_corners,
                    vec_fld.order,
                    form_dofs,
                    out_xi,
                    out_eta,
                    element_basis,
                )
                if vec_fld.order != 1:
                    vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
            else:
                # if vec_fld.order == 1:
                vf = np.zeros(
                    (
                        out_xi.size,
                        out_eta.size,
                        2,
                    ),
                    np.float64,
                )
                # else:
                #     vf = np.zeros(
                #         (
                #             output_basis.basis_xi.order + 1,
                #             output_basis.basis_eta.order + 1,
                #         ),
                #         np.float64,
                #     )
        else:
            x = poly_x(
                element_corners[:, 0],
                out_xi,
                out_eta,
            )
            y = poly_y(
                element_corners[:, 1],
                out_xi,
                out_eta,
            )
            vf = np.asarray(vec_fld(x, y), np.float64, copy=None)
        vec_field_lists.append(vf.reshape((-1, 2)))

    return tuple(vec_field_lists)


# TODO: REMOVE
def compute_vector_fields_nonlin(
    system: KFormSystem,
    leaf_elements: Sequence[int] | npt.NDArray[np.integer],
    cache,
    vector_fields: Sequence[Function2D | KFormUnknown],
    corners: FixedElementArray[np.float64],
    element_orders: FixedElementArray[np.uint32],
    output_orders: FixedElementArray[np.uint32],
    unknown_offsets: FixedElementArray[np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32] | None,
) -> tuple[npt.NDArray[np.uint64], tuple[npt.NDArray[np.float64], ...]]:
    """Evaluate vector fields which may be non-linear."""
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(len(leaf_elements) + 1, np.uint64)

    for i_out, e in enumerate(leaf_elements):
        order_1, order_2 = element_orders[e]
        out_order_1, out_order_2 = output_orders[e]
        e_cache_1 = cache[order_1]
        e_cache_2 = cache[order_2]
        o_cache_1 = cache[out_order_1]
        o_cache_2 = cache[out_order_2]

        e_corners = corners[e]
        # Extract element DoFs
        for i, vec_fld in enumerate(vector_fields):
            if isinstance(vec_fld, KFormUnknown):
                if solution is not None:
                    i_form = system.unknown_forms.index(vec_fld)
                    element_dofs = solution[e]
                    form_offset = unknown_offsets[e][i_form]
                    form_offset_end = unknown_offsets[e][i_form + 1]
                    form_dofs = element_dofs[form_offset:form_offset_end]
                    vf = reconstruct(
                        e_corners,
                        vec_fld.order,
                        form_dofs,
                        o_cache_1.int_nodes_1d[None, :],
                        o_cache_2.int_nodes_1d[:, None],
                        e_cache_1,
                        e_cache_2,
                    )  # type: ignore
                    if vec_fld.order != 1:
                        vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
                else:
                    vf = np.zeros(
                        (
                            o_cache_2.integration_order + 1,
                            o_cache_1.integration_order + 1,
                            2,
                        ),
                        np.float64,
                    )
            else:
                x = poly_x(
                    e_corners[:, 0],
                    o_cache_1.int_nodes_1d[None, :],
                    o_cache_2.int_nodes_1d[:, None],
                )
                y = poly_y(
                    e_corners[:, 1],
                    o_cache_1.int_nodes_1d[None, :],
                    o_cache_2.int_nodes_1d[:, None],
                )
                vf = vec_fld(x, y)  # type: ignore
            vec_field_lists[i].append(np.reshape(vf, (-1, 2)))
        vec_field_offsets[i_out + 1] = vec_field_offsets[i_out] + (
            o_cache_1.integration_order + 1
        ) * (o_cache_2.integration_order + 1)
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    return vec_field_offsets, vec_fields


def rhs_2d_element_projection(
    right: KElementProjection,
    corners: npt.NDArray[np.float64],
    basis: Basis2D,
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KElementProjection
        The projection of a function on the element.

    corners : (4, 2) array
        Array with corners of the element.

    basis : Basis2D
        Basis to use for computing the projection.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    fn = right.func

    # If `fn` is `None`, it is equal to just zeros
    if fn is None:
        n_dof: int
        if right.weight.order == 0:
            n_dof = (basis.basis_xi.order + 1) * (basis.basis_eta.order + 1)
        elif right.weight.order == 1:
            n_dof = (
                basis.basis_xi.order + 1
            ) * basis.basis_eta.order + basis.basis_xi.order * (basis.basis_eta.order + 1)
        elif right.weight.order == 2:
            n_dof = basis.basis_xi.order * basis.basis_eta.order
        else:
            raise ValueError(f"Invalid weight order {right.weight.order}.")

        return np.zeros(n_dof)

    return element_dual_dofs(UnknownFormOrder(right.weight.order + 1), corners, basis, fn)


def _extract_rhs_2d(
    proj: Sequence[tuple[float, KExplicit]],
    weight: KWeight,
    corners: npt.NDArray[np.float64],
    basis: Basis2D,
) -> npt.NDArray[np.float64]:
    """Extract the rhs resulting from element projections.

    Combines the sequence of :class:`KExplicit` terms together.

    Parameters
    ----------
    proj : Sequence of (float, KExplicit)
        Sequence of projections to compute.

    weight : KWeight
        Weight form used for these projections.

    corners : (4, 2) array
        Array of corners of the element.

    basis : Basis2D
        Basis to use for computing the projection.

    Returns
    -------
    array
        Array of the resulting projection degrees of freedom.
    """
    n_dof: int
    # Create empty vector into which to accumulate
    if weight.order == 0:
        n_dof = (basis.basis_xi.order + 1) * (basis.basis_eta.order + 1)
    elif weight.order == 1:
        n_dof = (
            basis.basis_xi.order + 1
        ) * basis.basis_eta.order + basis.basis_xi.order * (basis.basis_eta.order + 1)
    elif weight.order == 2:
        n_dof = basis.basis_xi.order * basis.basis_eta.order
    else:
        raise ValueError(f"Invalid weight order {weight.order}.")

    vec = np.zeros(n_dof, np.float64)

    # Loop over all entries that are KElementProjection
    for k, f in filter(lambda v: isinstance(v[1], KElementProjection), proj):
        assert isinstance(f, KElementProjection)
        rhs = rhs_2d_element_projection(f, corners, basis)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def compute_element_rhs(
    ie: int,
    system: KFormSystem,
    basis_cache: FemCache,
    orders: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute rhs for an element.

    This basically means just concatenating the projections of the functions on the
    element for each of the equations in the system.

    Parameters
    ----------
    ie : int
        Index of the element.
        TODO: REPLACE WITH JUST ORDERS AND BASIS

    system : KFormSystem
        System for which to compute the rhs.

    basis_cache : FemCache
        Cache from which to get the basis from.

    orders : FixedElementArray[np.uint32]
        Array with orders of the elements.

    corners : FixedElementArray[np.float64]
        Array with corners of the elements.

    Returns
    -------
    array
        Array with the resulting rhs.
    """
    vecs: list[npt.NDArray[np.float64]] = list()
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    element_corners = corners[ie]
    for equation in system.equations:
        vecs.append(
            _extract_rhs_2d(
                equation.right.explicit_terms, equation.weight, element_corners, basis
            )
        )

    return np.concatenate(vecs, dtype=np.float64)


@dataclass(frozen=True)
class ElementConstraint:
    """Type intended to enforce a constraint on an element.

    Parameters
    ----------
    i_e : int
        Index of the element for which this constraint is applied.

    dofs : (n,) array
        Array with indices of the degrees of freedom of the element involved.

    coeffs : (n,) array
        Array with coefficients of degrees of freedom of the element involved.
    """

    i_e: int
    dofs: npt.NDArray[np.uint32]
    coeffs: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Constraint:
    """Type used to specify constraints on degrees of freedom.

    This type combines the individual :class:`ElementConstraint` together
    with a right-hand side of the constraint.

    Parameters
    ----------
    rhs : float
        The right-hand side of the constraint.

    *element_constraints : ElementConstraint
        Constraints to combine together.
    """

    rhs: float
    element_constraints: tuple[ElementConstraint, ...]

    def __init__(self, rhs: float, *element_constraints: ElementConstraint) -> None:
        object.__setattr__(self, "rhs", float(rhs))
        object.__setattr__(self, "element_constraints", element_constraints)


# TODO: remove the slow version.
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


def _continuity_element_1_forms(
    unknown_ordering: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    e_other: int,
    e_self: int,
    side_other: ElementSide,
    side_self: ElementSide,
) -> tuple[Constraint, ...]:
    """Generate equations for 1-form continuity between elements.

    Returns
    -------
    tuple of Constraint
        List of constraint equations, which enforce continuity of 1 forms.
    """
    equations: list[Constraint] = list()

    base_other_dofs = np.flip(
        elements.get_boundary_dofs(e_other, UnknownFormOrder.FORM_ORDER_1, side_other)
    )
    base_self_dofs = elements.get_boundary_dofs(
        e_self, UnknownFormOrder.FORM_ORDER_1, side_self
    )
    order_self = elements.get_element_order_on_side(e_self, side_self)
    order_other = elements.get_element_order_on_side(e_other, side_other)

    # This here is magic to see if the vector degrees of freedom are aligned or opposed.
    # If they are opposed, then one pair of coefficients needs a negative sign.
    trans_table = (-1, +1, +1, -1)
    orientation_coefficient = (
        trans_table[side_self.value - 1] * trans_table[side_other.value - 1]
    )

    for var_idx in (
        iform
        for iform, form in enumerate(unknown_ordering.form_orders)
        if form == UnknownFormOrder.FORM_ORDER_1
    ):
        self_var_offset = int(dof_offsets[e_self][var_idx])
        other_var_offset = int(dof_offsets[e_other][var_idx])

        dofs_other = base_other_dofs + other_var_offset
        dofs_self = base_self_dofs + self_var_offset

        if order_self == order_other:
            assert base_other_dofs.size == base_self_dofs.size
            for v1, v2 in zip(dofs_self, dofs_other, strict=True):
                equations.append(
                    Constraint(
                        0,
                        ElementConstraint(e_self, np.asarray([v1]), np.array([+1.0])),
                        ElementConstraint(
                            e_other, np.asarray([v2]), np.array([orientation_coefficient])
                        ),
                    )
                )

        else:
            if order_self > order_other:
                order_high = order_self
                order_low = order_other
                dofs_high = dofs_self
                dofs_low = dofs_other
                e_high = e_self
                e_low = e_other
            else:
                order_low = order_self
                order_high = order_other
                dofs_high = dofs_other
                dofs_low = dofs_self
                e_low = e_self
                e_high = e_other

            _, coeffs_1 = continuity_matrices(order_high, order_low)

            for i_h, v_h in zip(range(order_high), dofs_high, strict=True):
                coefficients = coeffs_1[i_h, ...]

                equations.append(
                    Constraint(
                        0,
                        ElementConstraint(
                            e_high, np.array([v_h]), np.array([orientation_coefficient])
                        ),
                        ElementConstraint(e_low, dofs_low, coefficients),
                    )
                )
                # ConstraintEquation(
                #     np.concatenate(((v_h,), dofs_low)),
                #     np.concatenate(((orientation_coefficient,), coefficients)),
                #     0.0,

    return tuple(equations)


def _continuity_element_0_forms_inner(
    unknown_ordering: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    e_other: int,
    e_self: int,
    side_other: ElementSide,
    side_self: ElementSide,
) -> tuple[Constraint, ...]:
    """Generate equations for 0-form continuity between elements.

    Returns
    -------
    tuple of Constraint
        List of constraint equations, which enforce continuity of 0 forms.
    """
    equations: list[Constraint] = list()

    base_other_dofs = np.flip(
        elements.get_boundary_dofs(e_other, UnknownFormOrder.FORM_ORDER_0, side_other)
    )
    base_self_dofs = elements.get_boundary_dofs(
        e_self, UnknownFormOrder.FORM_ORDER_0, side_self
    )
    order_self = elements.get_element_order_on_side(e_self, side_self)
    order_other = elements.get_element_order_on_side(e_other, side_other)

    for var_idx in (
        iform
        for iform, form in enumerate(unknown_ordering.form_orders)
        if form == UnknownFormOrder.FORM_ORDER_0
    ):
        self_var_offset = int(dof_offsets[e_self][var_idx])
        other_var_offset = int(dof_offsets[e_other][var_idx])

        dofs_other = base_other_dofs + other_var_offset
        dofs_self = base_self_dofs + self_var_offset

        if order_self == order_other:
            assert base_other_dofs.size == base_self_dofs.size
            for v1, v2 in zip(dofs_self[1:-1], dofs_other[1:-1], strict=True):
                equations.append(
                    Constraint(
                        0,
                        ElementConstraint(e_self, np.asarray([v1]), np.array([+1.0])),
                        ElementConstraint(e_other, np.asarray([v2]), np.array([-1.0])),
                    )
                )

        else:
            if order_self > order_other:
                order_high = order_self
                order_low = order_other
                dofs_high = dofs_self
                dofs_low = dofs_other
                e_high = e_self
                e_low = e_other
            else:
                order_low = order_self
                order_high = order_other
                dofs_high = dofs_other
                dofs_low = dofs_self
                e_low = e_self
                e_high = e_other
            dofs_high = dofs_high[1:-1]

            coeffs_0, _ = continuity_matrices(order_high, order_low)

            for i_h, v_h in zip(range(order_high - 1), dofs_high, strict=True):
                coefficients = coeffs_0[i_h + 1, ...]

                equations.append(
                    Constraint(
                        0,
                        ElementConstraint(e_high, np.array([v_h]), np.array([-1])),
                        ElementConstraint(e_low, dofs_low, coefficients),
                    )
                )

    return tuple(equations)


def _continuity_element_0_forms_corner(
    unknown_ordering: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    e_other: int,
    e_self: int,
    side_other: ElementSide,
    side_self: ElementSide,
) -> tuple[Constraint, ...]:
    """Generate equations for 0-form continuity between elements.

    Returns
    -------
    tuple of Constraint
        List of constraint equations, which enforce continuity of 0 forms.
    """
    # BUG: Wrong signs somehow
    equations: list[Constraint] = list()

    base_other_dofs = elements.get_boundary_dofs(
        e_other, UnknownFormOrder.FORM_ORDER_0, side_other
    )[-1]
    base_self_dofs = elements.get_boundary_dofs(
        e_self, UnknownFormOrder.FORM_ORDER_0, side_self
    )[0]

    for var_idx in (
        iform
        for iform, form in enumerate(unknown_ordering.form_orders)
        if form == UnknownFormOrder.FORM_ORDER_0
    ):
        self_var_offset = int(dof_offsets[e_self][var_idx])
        other_var_offset = int(dof_offsets[e_other][var_idx])

        dofs_other = base_other_dofs + other_var_offset
        dofs_self = base_self_dofs + self_var_offset

        equations.append(
            Constraint(
                0,
                ElementConstraint(e_self, np.asarray([dofs_self]), np.array([+1.0])),
                ElementConstraint(e_other, np.asarray([dofs_other]), np.array([-1.0])),
            )
        )

    return tuple(equations)


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


def _continuity_parent_child_edges(
    unknown_orders: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    e_parent: int,
    e_child: int,
    side: ElementSide,
    flipped: bool,
) -> tuple[Constraint, ...]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    unknown_orders : UnknownOrderings
        Orders of all unknown differential forms.
    e_parent : int
        Parent element.
    e_child : int
        Child element.
    side : ElementSide
        Index of the boundary which is to be connected.
    flipped : bool
        Determines if the child is on the second half of the boundary instead of the
        first. This in practice means that coefficients in equations must be flipped.

    Returns
    -------
    tuple of Constraint
        Equations which enforce continuity of the 1-forms on the boundary of a parent and
        a child.
    """
    dofs_parent = elements.get_boundary_dofs(
        e_parent, UnknownFormOrder.FORM_ORDER_1, side
    )
    dofs_child = elements.get_boundary_dofs(e_child, UnknownFormOrder.FORM_ORDER_1, side)
    _, coeff_1 = continuity_child_matrices(
        elements.get_element_order_on_side(e_child, side),
        elements.get_element_order_on_side(e_parent, side),
    )
    if flipped:
        coeff_1 = np.flip(coeff_1, axis=0)
        coeff_1 = np.flip(coeff_1, axis=1)

    equations: list[Constraint] = list()
    for var_idx in (
        i_form
        for i_form, form in enumerate(unknown_orders.form_orders)
        if form == UnknownFormOrder.FORM_ORDER_1
    ):
        var_parent_offset = int(dof_offsets[e_parent][var_idx])
        var_child_offset = int(dof_offsets[e_child][var_idx])

        dp = var_parent_offset + dofs_parent
        dc = var_child_offset + dofs_child

        for i_c, v_c in enumerate(dc):
            coeffs = coeff_1[i_c, :]
            equations.append(
                Constraint(
                    0,
                    ElementConstraint(e_child, np.array([v_c]), np.array([-1])),
                    ElementConstraint(e_parent, dp, coeffs),
                )
            )

    return tuple(equations)


def _continuity_parent_child_nodes(
    unknown_orders: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    e_parent: int,
    e_child: int,
    side: ElementSide,
    flipped: bool,
) -> tuple[Constraint, ...]:
    """Connect parent and child element boundaries.

    Parameters
    ----------
    unknown_orders : UnknownOrderings
        Orders of all unknown differential forms.
    e_parent : int
        Parent element.
    e_child : int
        Child element.
    side : ElementSide
        Index of the boundary which is to be connected.
    flipped : bool
        Determines if the child is on the second half of the boundary instead of the
        first. This in practice means that coefficients in equations must be flipped.

    Returns
    -------
    tuple of Constraint
        Equations which enforce continuity of the 0-forms on the boundary of a parent and
        a child.
    """
    dofs_parent = elements.get_boundary_dofs(
        e_parent, UnknownFormOrder.FORM_ORDER_0, side
    )
    dofs_child = elements.get_boundary_dofs(e_child, UnknownFormOrder.FORM_ORDER_0, side)
    coeff_0, _ = continuity_child_matrices(
        elements.get_element_order_on_side(e_child, side),
        elements.get_element_order_on_side(e_parent, side),
    )
    if flipped:
        coeff_0 = np.flip(coeff_0, axis=0)
        coeff_0 = np.flip(coeff_0, axis=1)
        # Only do the corner on non-flipped, so
        # that we do not double constraints for it.
        dofs_child = dofs_child[1:-1]
        coeff_0 = coeff_0[1:-1, :]

    equations: list[Constraint] = list()
    for var_idx in (
        i_form
        for i_form, form in enumerate(unknown_orders.form_orders)
        if form == UnknownFormOrder.FORM_ORDER_0
    ):
        var_parent_offset = int(dof_offsets[e_parent][var_idx])
        var_child_offset = int(dof_offsets[e_child][var_idx])

        dp = var_parent_offset + dofs_parent
        dc = var_child_offset + dofs_child

        for i_c, v_c in enumerate(dc):
            coeffs = coeff_0[i_c, :]
            equations.append(
                Constraint(
                    0,
                    ElementConstraint(e_child, np.array([v_c]), np.array([-1])),
                    ElementConstraint(e_parent, dp, coeffs),
                )
            )

    return tuple(equations)


def _parent_child_equations(
    unknown_ordering: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    ie: int,
    child_bl: int,
    child_br: int,
    child_tr: int,
    child_tl: int,
) -> tuple[Constraint, ...]:
    """Create constraint equations for the parent-child and child-child continuity.

    Parameters
    ----------
    unknown_ordering : UnknownOrderings
        Order of unknown differential forms.


    Returns
    -------
    tuple of Constraint
        Tuple of the constraint equations which ensure continuity between these elements.
    """
    child_child: list[Constraint] = list()
    parent_child: list[Constraint] = list()
    for form in unknown_ordering.form_orders:
        # TODO: change this to be per form instead of all forms of a type at once
        if form == UnknownFormOrder.FORM_ORDER_1:
            # Glue 1-form edges
            child_child += _continuity_element_1_forms(
                unknown_ordering,
                elements,
                dof_offsets,
                child_br,
                child_bl,
                ElementSide.SIDE_LEFT,
                ElementSide.SIDE_RIGHT,
            )
            child_child += _continuity_element_1_forms(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tr,
                child_br,
                ElementSide.SIDE_BOTTOM,
                ElementSide.SIDE_TOP,
            )
            child_child += _continuity_element_1_forms(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tl,
                child_tr,
                ElementSide.SIDE_RIGHT,
                ElementSide.SIDE_LEFT,
            )
            child_child += _continuity_element_1_forms(
                unknown_ordering,
                elements,
                dof_offsets,
                child_bl,
                child_tl,
                ElementSide.SIDE_TOP,
                ElementSide.SIDE_BOTTOM,
            )

            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_bl,
                ElementSide.SIDE_BOTTOM,
                False,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_br,
                ElementSide.SIDE_BOTTOM,
                True,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_br,
                ElementSide.SIDE_RIGHT,
                False,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tr,
                ElementSide.SIDE_RIGHT,
                True,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tr,
                ElementSide.SIDE_TOP,
                False,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tl,
                ElementSide.SIDE_TOP,
                True,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tl,
                ElementSide.SIDE_LEFT,
                False,
            )
            parent_child += _continuity_parent_child_edges(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_bl,
                ElementSide.SIDE_LEFT,
                True,
            )

        elif form == UnknownFormOrder.FORM_ORDER_0:
            # Glue 0-form edges
            child_child += _continuity_element_0_forms_inner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_br,
                child_bl,
                ElementSide.SIDE_LEFT,
                ElementSide.SIDE_RIGHT,
            )
            child_child += _continuity_element_0_forms_inner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tr,
                child_br,
                ElementSide.SIDE_BOTTOM,
                ElementSide.SIDE_TOP,
            )
            child_child += _continuity_element_0_forms_inner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tl,
                child_tr,
                ElementSide.SIDE_RIGHT,
                ElementSide.SIDE_LEFT,
            )
            child_child += _continuity_element_0_forms_inner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_bl,
                child_tl,
                ElementSide.SIDE_TOP,
                ElementSide.SIDE_BOTTOM,
            )
            # Glue the corner they all share
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_bl,
                child_br,
                ElementSide.SIDE_RIGHT,
                ElementSide.SIDE_LEFT,
            )
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_br,
                child_tr,
                ElementSide.SIDE_TOP,
                ElementSide.SIDE_BOTTOM,
            )
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tr,
                child_tl,
                ElementSide.SIDE_LEFT,
                ElementSide.SIDE_RIGHT,
            )

            # Glue the child corners too
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_br,
                child_bl,
                ElementSide.SIDE_LEFT,
                ElementSide.SIDE_RIGHT,
            )
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tr,
                child_br,
                ElementSide.SIDE_BOTTOM,
                ElementSide.SIDE_TOP,
            )
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_tl,
                child_tr,
                ElementSide.SIDE_RIGHT,
                ElementSide.SIDE_LEFT,
            )
            child_child += _continuity_element_0_forms_corner(
                unknown_ordering,
                elements,
                dof_offsets,
                child_bl,
                child_tl,
                ElementSide.SIDE_TOP,
                ElementSide.SIDE_BOTTOM,
            )

            # Don't add the fourth equation!

            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_bl,
                ElementSide.SIDE_BOTTOM,
                False,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_br,
                ElementSide.SIDE_BOTTOM,
                True,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_br,
                ElementSide.SIDE_RIGHT,
                False,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tr,
                ElementSide.SIDE_RIGHT,
                True,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tr,
                ElementSide.SIDE_TOP,
                False,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tl,
                ElementSide.SIDE_TOP,
                True,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_tl,
                ElementSide.SIDE_LEFT,
                False,
            )
            parent_child += _continuity_parent_child_nodes(
                unknown_ordering,
                elements,
                dof_offsets,
                ie,
                child_bl,
                ElementSide.SIDE_LEFT,
                True,
            )

        elif form == UnknownFormOrder.FORM_ORDER_2:
            continue

        else:
            raise ValueError(f"Unknown form order {form=}.")

    return tuple(child_child + parent_child)


def _compute_element_matrix(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    ie: int,
    dof_offsets_array: FixedElementArray[np.uint32],
    leaf_matrices: FlexibleElementArray[np.float64, np.uint32],
    elements: ElementCollection,
) -> sp.csr_array | npt.NDArray[np.float64]:
    """Compute matrix for a top level element."""
    children = collection.child_array[ie]
    if children.size == 0:
        return leaf_matrices[ie]

    # Matrix due to self DoFs (zeros)
    dof_offsets = dof_offsets_array[ie]
    self_matrix = sp.csr_array((dof_offsets[-1], dof_offsets[-1]))

    # Child element matrices
    child_matrices = [
        _compute_element_matrix(
            unknown_ordering,
            collection,
            int(ic),
            dof_offsets_array,
            leaf_matrices,
            collection,
        )
        for ic in children
    ]
    sizes = [mat.shape[0] for mat in child_matrices]
    offsets = {int(ic): int(np.sum(sizes[:i])) for i, ic in enumerate(children)}

    # Lagrange muliplier block
    constraint_equations = _parent_child_equations(
        unknown_ordering, collection, dof_offsets_array, ie, *children
    )
    rows_c: list[npt.NDArray[np.uint32]] = list()
    rows_s: list[npt.NDArray[np.uint32]] = list()
    cols_c: list[npt.NDArray[np.uint32]] = list()
    cols_s: list[npt.NDArray[np.uint32]] = list()
    vals_c: list[npt.NDArray[np.float64]] = list()
    vals_s: list[npt.NDArray[np.float64]] = list()
    for ic, con in enumerate(constraint_equations):
        assert con.rhs == 0
        for ec in con.element_constraints:
            idx = ec.i_e
            if idx == ie:
                cols_s.append(ec.dofs)
                rows_s.append(np.full_like(ec.dofs, ic, np.uint32))
                vals_s.append(ec.coeffs)
            else:
                offset = offsets[idx]
                cols_c.append(ec.dofs + offset)
                rows_c.append(np.full_like(ec.dofs, ic, np.uint32))
                vals_c.append(ec.coeffs)

    lagmat_c = sp.csr_array(
        (np.concatenate(vals_c), (np.concatenate(rows_c), np.concatenate(cols_c))),
        shape=(len(constraint_equations), sum(sizes)),
    )
    lagmat_s = sp.csr_array(
        (np.concatenate(vals_s), (np.concatenate(rows_s), np.concatenate(cols_s))),
        shape=(len(constraint_equations), dof_offsets[-1]),
    )

    assert len(constraint_equations) == _compute_element_lagrange_multipliers(
        ie, elements, unknown_ordering
    )  # TODO: remove

    return cast(
        sp.csr_array,
        sp.block_array(
            (
                (self_matrix, lagmat_s.T, None),
                (lagmat_s, None, lagmat_c),
                (None, lagmat_c.T, sp.block_diag([*child_matrices])),
            ),
            format="csr",
        ),
    )


def assemble_matrix(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    dof_offsets_array: FixedElementArray[np.uint32],
    leaf_matrices: FlexibleElementArray[np.float64, np.uint32],
) -> sp.csr_array:
    """Assemble global matrix."""
    matrices: list[sp.csr_array | npt.NDArray[np.float64]] = list()
    for ie, i_parent in enumerate(collection.parent_array):
        if int(i_parent[0]) != 0:
            continue
        mat = _compute_element_matrix(
            unknown_ordering,
            collection,
            ie,
            dof_offsets_array,
            leaf_matrices,
            collection,
        )
        matrices.append(mat)
    return cast(sp.csr_array, sp.block_diag(matrices, format="csr"))


def _compute_element_vector(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    ie: int,
    dof_offsets_array: FixedElementArray[np.uint32],
    lagrange_dof_count_array: FixedElementArray[np.uint32],
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Compute matrix for a top level element."""
    children = collection.child_array[ie]
    if children.size == 0:
        return leaf_vectors[ie]

    # Matrix due to self DoFs (zeros)
    dof_offsets = dof_offsets_array[ie]
    self_vec = np.zeros(dof_offsets[-1])

    # Child element matrices
    child_vectors = [
        _compute_element_vector(
            unknown_ordering,
            collection,
            int(ic),
            dof_offsets_array,
            lagrange_dof_count_array,
            leaf_vectors,
        )
        for ic in children
    ]
    lagvals = np.zeros(lagrange_dof_count_array[ie])

    return np.concatenate((self_vec, lagvals, *child_vectors), dtype=np.float64)


def assemble_vector(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    dof_offsets_array: FixedElementArray[np.uint32],
    lagrange_dof_count_array: FixedElementArray[np.uint32],
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Assemble global vector."""
    vectors: list[npt.NDArray[np.float64]] = list()
    for ie, i_parent in enumerate(collection.parent_array):
        if int(i_parent[0]) != 0:
            continue
        vectors.append(
            _compute_element_vector(
                unknown_ordering,
                collection,
                ie,
                dof_offsets_array,
                lagrange_dof_count_array,
                leaf_vectors,
            )
        )

    return np.concatenate(vectors, dtype=np.float64)


def _compute_element_forcing(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    ie: int,
    dof_offsets_array: FixedElementArray[np.uint32],
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Compute forcing for an element."""
    children = collection.child_array[ie]
    if children.size == 0:
        return leaf_vectors[ie]

    # Forcing due to self DoFs (zeros)
    dof_offsets = dof_offsets_array[ie]

    # Child element forcing
    forcing_arrays = {
        **{
            ic: _compute_element_forcing(
                unknown_ordering,
                collection,
                int(ic),
                dof_offsets_array,
                leaf_vectors,
                solution,
            )
            for ic in children
        },
        ie: np.zeros(dof_offsets[-1], np.float64),
    }

    # Lagrange muliplier block
    constraint_equations = _parent_child_equations(
        unknown_ordering, collection, dof_offsets_array, ie, *children
    )
    vals: list[float] = list()
    e_sol = solution[ie]
    lagmul = e_sol[dof_offsets[-1] :]
    for ic, con in enumerate(constraint_equations):
        assert con.rhs == 0
        v = 0.0
        for ec in con.element_constraints:
            sol = solution[ec.i_e]
            v += np.dot(ec.coeffs, sol[ec.dofs])
            forcing = forcing_arrays[ec.i_e]
            forcing[ec.dofs] += ec.coeffs * lagmul[ic]
        vals.append(v)

    return np.concatenate(
        [forcing_arrays[ie], vals, *(forcing_arrays[ic] for ic in children)],
        dtype=np.float64,
    )


def assemble_forcing(
    unknown_ordering: UnknownOrderings,
    collection: ElementCollection,
    dof_offsets_array: FixedElementArray[np.uint32],
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Assemble global vector."""
    vectors: list[npt.NDArray[np.float64]] = list()
    for ie, i_parent in enumerate(collection.parent_array):
        if int(i_parent[0]) != 0:
            continue
        vectors.append(
            _compute_element_forcing(
                unknown_ordering,
                collection,
                ie,
                dof_offsets_array,
                leaf_vectors,
                solution,
            )
        )

    return np.concatenate(vectors, dtype=np.float64)


def _find_boundary_id(s: Surface, i: int) -> ElementSide:
    """Find what boundary the line with a given index is in the surface."""
    if s[0].index == i:
        return ElementSide.SIDE_BOTTOM
    if s[1].index == i:
        return ElementSide.SIDE_RIGHT
    if s[2].index == i:
        return ElementSide.SIDE_TOP
    if s[3].index == i:
        return ElementSide.SIDE_LEFT
    raise ValueError(f"Line with index {i} is not in the surface {s}.")


def _top_level_continuity_1(
    mesh: Mesh2D,
    top_indices: npt.NDArray[np.uint32],
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    dof_offsets: FixedElementArray[np.uint32],
) -> list[Constraint]:
    """Create equations enforcing 1-form continuity between top level elements."""
    # Continuity of 1-forms
    continuity_equations: list[Constraint] = list()
    for il in range(mesh.dual.n_lines):
        dual_line = mesh.dual.get_line(il + 1)
        idx_neighbour = dual_line.begin
        idx_self = dual_line.end
        if not idx_self or not idx_neighbour:
            continue

        # For each variable which must be continuous, get locations in left and right
        s_other = mesh.primal.get_surface(idx_neighbour)
        s_self = mesh.primal.get_surface(idx_self)

        i_other = int(top_indices[idx_neighbour.index])
        i_self = int(top_indices[idx_self.index])
        continuity_equations.extend(
            _continuity_element_1_forms(
                unknown_orders,
                elements,
                dof_offsets,
                i_other,
                i_self,
                _find_boundary_id(s_other, il),
                _find_boundary_id(s_self, il),
            )
        )
    return continuity_equations


def _top_level_continuity_0(
    mesh: Mesh2D,
    top_indices: npt.NDArray[np.uint32],
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    only_first_order: bool,
    dof_offsets: FixedElementArray[np.uint32],
) -> list[Constraint]:
    """Create equations enforcing 0-form continuity between top level elements."""
    # Continuity of 0-forms on the non-corner DoFs
    continuity_equations: list[Constraint] = list()
    if only_first_order:
        for il in range(mesh.dual.n_lines):
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end

            if not idx_neighbour or not idx_self:
                continue

            s_other = mesh.primal.get_surface(idx_neighbour)
            s_self = mesh.primal.get_surface(idx_self)

            i_other = int(top_indices[idx_neighbour.index])
            i_self = int(top_indices[idx_self.index])
            continuity_equations.extend(
                _continuity_element_0_forms_inner(
                    unknown_orders,
                    elements,
                    dof_offsets,
                    i_other,
                    i_self,
                    _find_boundary_id(s_other, il),
                    _find_boundary_id(s_self, il),
                )
            )

    # Continuity of 0-forms on the corner DoFs
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

            i_other = int(top_indices[idx_neighbour.index])
            i_self = int(top_indices[idx_self.index])
            continuity_equations.extend(
                _continuity_element_0_forms_corner(
                    unknown_orders,
                    elements,
                    dof_offsets,
                    i_other,
                    i_self,
                    _find_boundary_id(s_other, id_line.index),
                    _find_boundary_id(s_self, id_line.index),
                )
            )

    return continuity_equations


def mesh_continuity_constraints(
    system: KFormSystem,
    mesh: Mesh2D,
    top_indices: npt.NDArray[np.uint32],
    unknown_orders: UnknownOrderings,
    elements: ElementCollection,
    only_first_order: bool,
    dof_offsets: FixedElementArray[np.uint32],
) -> tuple[Constraint, ...]:
    """Return the boundary conditions system."""
    continuity_equations: list[Constraint] = list()

    # Continuity of 1-forms on top level
    if system.get_form_indices_by_order(1):
        continuity_equations.extend(
            _top_level_continuity_1(
                mesh, top_indices, elements, unknown_orders, dof_offsets
            )
        )

    if system.get_form_indices_by_order(0):
        continuity_equations.extend(
            _top_level_continuity_0(
                mesh, top_indices, elements, unknown_orders, only_first_order, dof_offsets
            )
        )

    return tuple(continuity_equations)


def _element_weak_boundary_condition(
    elements: ElementCollection,
    ie: int,
    side: ElementSide,
    unknown_orders: UnknownOrderings,
    unknown_index: int,
    dof_offsets: FixedElementArray[np.uint32],
    weak_terms: Sequence[tuple[float, KBoundaryProjection]],
    cache_1d: FemCache,
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
                cache_1d,
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
                cache_1d,
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
                cache_1d,
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
                cache_1d,
            )

        return current

    side_order = elements.get_element_order_on_side(ie, side)

    basis_1d = cache_1d.get_basis1d(side_order)
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
    cache_1d: FemCache,
    skip_first: bool,
    skip_last: bool,
) -> tuple[ElementConstraint, ...]:
    """Determine boundary conditions given an element and a side."""
    # children = elements.child_array[ie]
    # if children.size != 0:
    #     # Node, has children
    #     current: tuple[ElementConstraint, ...] = tuple()
    #     # For children skipping:
    #     # Skip first if: (on start of the side AND skip_first is True) OR (end of side)
    #     # Skip last if: on end of the side AND skip_last is True
    #     # So only time first not skipped if: skip_first is True AND beginning of side
    #     if side == ElementSide.SIDE_LEFT or side == ElementSide.SIDE_BOTTOM:
    #         current += _element_strong_boundary_condition(
    #             elements,
    #             int(children[0]),
    #             side,
    #             unknown_orders,
    #             unknown_index,
    #             dof_offsets,
    #             strong_bc,
    #             cache_1d,
    #             (skip_first and side == ElementSide.SIDE_BOTTOM)
    #             or side == ElementSide.SIDE_LEFT,
    #             skip_last and side == ElementSide.SIDE_LEFT,
    #         )
    #     if side == ElementSide.SIDE_BOTTOM or side == ElementSide.SIDE_RIGHT:
    #         current += _element_strong_boundary_condition(
    #             elements,
    #             int(children[1]),
    #             side,
    #             unknown_orders,
    #             unknown_index,
    #             dof_offsets,
    #             strong_bc,
    #             cache_1d,
    #             (skip_first and side == ElementSide.SIDE_RIGHT)
    #             or side == ElementSide.SIDE_BOTTOM,
    #             skip_last and side == ElementSide.SIDE_BOTTOM,
    #         )
    #     if side == ElementSide.SIDE_RIGHT or side == ElementSide.SIDE_TOP:
    #         current += _element_strong_boundary_condition(
    #             elements,
    #             int(children[2]),
    #             side,
    #             unknown_orders,
    #             unknown_index,
    #             dof_offsets,
    #             strong_bc,
    #             cache_1d,
    #             (skip_first and side == ElementSide.SIDE_TOP)
    #             or side == ElementSide.SIDE_RIGHT,
    #             skip_last and side == ElementSide.SIDE_RIGHT,
    #         )
    #     if side == ElementSide.SIDE_TOP or side == ElementSide.SIDE_LEFT:
    #         current += _element_strong_boundary_condition(
    #             elements,
    #             int(children[3]),
    #             side,
    #             unknown_orders,
    #             unknown_index,
    #             dof_offsets,
    #             strong_bc,
    #             cache_1d,
    #             (skip_first and side == ElementSide.SIDE_LEFT)
    #             or side == ElementSide.SIDE_TOP,
    #             skip_last and side == ElementSide.SIDE_TOP,
    #         )

    #     return current

    side_order = elements.get_element_order_on_side(ie, side)

    basis_1d = cache_1d.get_basis1d(side_order)
    ndir = 2 * ((side.value & 2) >> 1) - 1
    i0 = side.value - 1
    i1 = side.value & 3
    corners = elements.corners_array[ie]
    p0: tuple[float, float] = corners[i0]
    p1: tuple[float, float] = corners[i1]
    # ndir, p0, p1 = _endpoints_from_line(e, i_side)
    dx = (p1[0] - p0[0]) / 2
    xv = np.astype((p1[0] + p0[0]) / 2 + dx * basis_1d.roots, np.float64, copy=False)
    dy = (p1[1] - p0[1]) / 2
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

    assert vals.size == dofs.size
    return (ElementConstraint(ie, dofs, vals),)


def mesh_boundary_conditions(
    evaluatable_terms: Sequence[KSum],
    mesh: Mesh2D,
    unknown_order: UnknownOrderings,
    elements: ElementCollection,
    dof_offsets: FixedElementArray[np.uint32],
    top_indices: npt.NDArray[np.uint32],
    strong_bcs: Sequence[Sequence[BoundaryCondition2DSteady]],
    cache_1d: FemCache,
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
        i_side = _find_boundary_id(primal_surface, i_boundary)
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
                        cache_1d,
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
                        cache_1d,
                    )
                )

            else:
                # Strong not given, but also no weak ones.
                pass

    return tuple(s_bcs), tuple(w_bcs)


def reconstruct_mesh_from_solution(
    system: KFormSystem,
    recon_order: int | None,
    element_collection: ElementCollection,
    caches: FemCache,
    dof_offsets: FixedElementArray[np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> pv.UnstructuredGrid:
    """Reconstruct the unknown differential forms."""
    build: dict[KFormUnknown, list[npt.NDArray[np.float64]]] = {
        form: list() for form in system.unknown_forms
    }
    xvals: list[npt.NDArray[np.float64]] = list()
    yvals: list[npt.NDArray[np.float64]] = list()

    node_array: list[npt.NDArray[np.int32]] = list()
    orders_1: list[int] = list()
    orders_2: list[int] = list()
    node_cnt = 0
    for ie, cnt in enumerate(element_collection.child_count_array):
        if int(cnt) != 0:
            continue

        # Extract element DoFs
        element_dofs = solution[ie]
        real_orders = element_collection.orders_array[ie]
        element_order = int(max(real_orders)) if recon_order is None else int(recon_order)
        order_1 = int(real_orders[0])
        order_2 = int(real_orders[1])
        orders_1.append(order_1)
        orders_2.append(order_2)
        element_basis = caches.get_basis2d(element_order, element_order)
        ordering = vtk_lagrange_ordering(element_order) + node_cnt
        node_array.append(np.concatenate(((ordering.size,), ordering)))
        node_cnt += ordering.size
        corners = element_collection.corners_array[ie]
        ex = poly_x(
            corners[:, 0],
            element_basis.basis_xi.roots[None, :],
            element_basis.basis_eta.roots[:, None],
        )
        ey = poly_y(
            corners[:, 1],
            element_basis.basis_xi.roots[None, :],
            element_basis.basis_eta.roots[:, None],
        )

        xvals.append(ex.flatten())
        yvals.append(ey.flatten())
        offsets = dof_offsets[ie]
        # Loop over each of the primal forms
        for idx, form in enumerate(system.unknown_forms):
            form_offset = int(offsets[idx])
            form_offset_end = int(offsets[idx + 1])
            form_dofs = element_dofs[form_offset:form_offset_end]
            if not form.is_primal:
                raise ValueError("Can not reconstruct a non-primal form.")
            # Reconstruct unknown
            recon_v = reconstruct(
                corners,
                form.order,
                form_dofs,
                element_basis.basis_xi.roots[None, :],
                element_basis.basis_eta.roots[:, None],
                caches.get_basis2d(order_1, order_2),
            )
            shape = (-1, 2) if form.order == 1 else (-1,)
            build[form].append(np.reshape(recon_v, shape))

    grid = pv.UnstructuredGrid(
        np.concatenate(node_array),
        np.full(len(node_array), pv.CellType.LAGRANGE_QUADRILATERAL),
        np.pad(
            np.stack((np.concatenate(xvals), np.concatenate(yvals)), axis=1),
            ((0, 0), (0, 1)),
        ),
    )

    # Build the outputs
    for form in build:
        vf = np.concatenate(build[form], axis=0, dtype=np.float64)
        grid.point_data[form.label] = vf

    return grid


def _extract_time_carry(
    ie: int,
    time_carry_index_array: FlexibleElementArray[np.uint32, np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Element function for extraction."""
    """Extract time carry from solution."""
    idx = time_carry_index_array[ie]
    sol = solution[ie]
    return sol[idx]


def extract_carry(
    element_collection: ElementCollection,
    time_carry_index_array: FlexibleElementArray[np.uint32, np.uint32],
    initial_solution: FlexibleElementArray[np.float64, np.uint32],
):
    """Extract carry terms from solution."""
    return call_per_element_flex(
        element_collection.com,
        1,
        np.float64,
        _extract_time_carry,
        time_carry_index_array,
        initial_solution,
    )


def compute_leaf_matrix(
    ie: int,
    expressions: _CompiledCodeMatrix,
    unknowns: UnknownOrderings,
    orders: FixedElementArray[np.uint32],
    basis_cache: FemCache,
    corners: FixedElementArray[np.float64],
    fields: FixedElementArray[np.object_],
) -> npt.NDArray[np.float64]:
    """Compute the element matrix."""
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    vec_fields = tuple(fields[ie])

    mat = compute_element_matrix(
        [form.value - 1 for form in unknowns.form_orders],
        expressions,
        corners[ie],
        vec_fields,
        basis,
    )

    return mat


def compute_leaf_vector(
    ie: int,
    expressions: _CompiledCodeMatrix,
    unknowns: UnknownOrderings,
    orders: FixedElementArray[np.uint32],
    basis_cache: FemCache,
    corners: FixedElementArray[np.float64],
    fields: FixedElementArray[np.object_],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Compute the element vector."""
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    vec_fields = tuple(fields[ie])

    vec = compute_element_vector(
        [form.value - 1 for form in unknowns.form_orders],
        expressions,
        corners[ie],
        vec_fields,
        basis,
        solution[ie],
    )

    return vec


def compute_element_dual(
    ie: int,
    ordering: UnknownOrderings,
    functions: Sequence[
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike] | None
    ],
    orders: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
    basis_cache: FemCache,
) -> npt.NDArray[np.float64]:
    """Compute element L2 projection."""
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    vecs: list[npt.NDArray[np.float64]] = list()
    for order, func in zip(ordering.form_orders, functions, strict=True):
        if func is None:
            vecs.append(np.zeros(order.full_unknown_count(order_1, order_2), np.float64))
        else:
            vecs.append(
                np.asarray(element_dual_dofs(order, corners[ie], basis, func), np.float64)
            )

    return np.concatenate(vecs)


def compute_element_primal(
    ie: int,
    ordering: UnknownOrderings,
    dual_dofs: FlexibleElementArray[np.float64, np.uint32],
    orders: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
    basis_cache: FemCache,
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    dual = np.array(dual_dofs[ie], np.float64, copy=True)
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    element_corners = corners[ie]
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    for form in ordering.form_orders:
        cnt = form.full_unknown_count(order_1, order_2)
        v = dual[offset : offset + cnt]
        if form in mats:
            m = mats[form]
        else:
            m = compute_element_mass_matrix(form, element_corners, basis, True)
            mats[form] = m

        dual[offset : offset + cnt] = m @ v

        offset += cnt

    return dual


def compute_element_primal_to_dual(
    ie: int,
    ordering: UnknownOrderings,
    primal_dofs: FlexibleElementArray[np.float64, np.uint32],
    orders: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
    basis_cache: FemCache,
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    primal = np.array(primal_dofs[ie], np.float64, copy=True)
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    element_corners = corners[ie]
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    for form in ordering.form_orders:
        cnt = form.full_unknown_count(order_1, order_2)
        v = primal[offset : offset + cnt]
        if form in mats:
            m = mats[form]
        else:
            m = compute_element_mass_matrix(form, element_corners, basis, False)
            mats[form] = m

        primal[offset : offset + cnt] = m @ v

        offset += cnt

    return primal


def compute_leaf_full_mass_matrix(
    ie: int,
    ordering: UnknownOrderings,
    orders: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
    basis_cache: FemCache,
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    order_1, order_2 = orders[ie]
    basis = basis_cache.get_basis2d(order_1, order_2)
    element_corners = corners[ie]
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    diags: list[npt.NDArray[np.float64]] = list()
    for form in ordering.form_orders:
        if form in mats:
            m = mats[form]
        else:
            m = compute_element_mass_matrix(form, element_corners, basis, False)
            mats[form] = m

        diags.append(m)

    mat = sp.block_diag(diags).toarray()

    return np.astype(mat, np.float64, copy=False)


def compute_element_vector_fields(
    ie: int,
    system: KFormSystem,
    child_count_array: FixedElementArray[np.uint32],
    orders_in: FixedElementArray[np.uint32],
    orders_out: FixedElementArray[np.uint32],
    basis_cache: FemCache,
    vector_fields: Sequence[Function2D | KFormUnknown],
    corners: FixedElementArray[np.float64],
    dof_offsets: FixedElementArray[np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> tuple:
    """Wrap the vector fields func."""
    if int(child_count_array[ie][0]) != 0:
        return (np.zeros((0, 0, 2)),) * len(vector_fields)

    order_1, order_2 = orders_in[ie]
    basis_element = basis_cache.get_basis2d(order_1, order_2)
    order_1, order_2 = orders_out[ie]
    basis_out = basis_cache.get_basis2d(order_1, order_2)

    res = compute_element_vector_fields_nonlin(
        system,
        basis_element,
        basis_out,
        vector_fields,
        corners[ie],
        dof_offsets[ie],
        solution[ie],
    )

    return res


def non_linear_solve_run(
    system: KFormSystem,
    max_iterations: int,
    relax: float,
    atol: float,
    rtol: float,
    print_residual: bool,
    unknown_ordering: UnknownOrderings,
    element_collection: ElementCollection,
    leaf_elements: npt.NDArray[np.integer],
    cache_2d: FemCache,
    compiled_system: CompiledSystem,
    explicit_vec: npt.NDArray[np.float64],
    dof_offsets: FixedElementArray[np.uint32],
    element_offsets: npt.NDArray[np.uint32],
    linear_element_matrices: FlexibleElementArray[np.float64, np.uint32],
    time_carry_index_array: FlexibleElementArray[np.uint32, np.uint32] | None,
    time_carry_term: FlexibleElementArray[np.float64, np.uint32] | None,
    solution: FlexibleElementArray[np.float64, np.uint32],
    global_lagrange: npt.NDArray[np.float64],
    max_mag: float,
    vector_fields: Sequence[Function2D | KFormUnknown],
    system_decomp: sla.SuperLU,
    lagrange_mat: sp.csr_array | None,
    return_all_residuals: bool = False,
):
    """Run the iterative non-linear solver.

    Based on how the compiled system looks, this may only take a single iteration,
    otherwise, it may run for as long as it needs to converge.
    """
    progress_tracker: None | ProgressTracker = None
    iter_cnt = 0
    base_vec = np.array(explicit_vec, copy=True)  # Make a copy
    if time_carry_term is not None:
        assert time_carry_index_array is not None
        for ie, (off, idx, vals) in enumerate(
            zip(element_offsets, time_carry_index_array, time_carry_term)
        ):
            base_vec[idx + int(off)] += vals
    else:
        assert time_carry_index_array is None
    residuals = np.zeros(max_iterations, np.float64)
    max_residual = 0.0
    while iter_cnt < max_iterations:
        # Recompute vector fields
        # Compute vector fields at integration points for leaf elements

        vec_fields_array = call_per_element_fix(
            element_collection.com,
            np.object_,
            len(vector_fields),
            compute_element_vector_fields,
            system,
            element_collection.child_count_array,
            element_collection.orders_array,
            element_collection.orders_array,
            cache_2d,
            vector_fields,
            element_collection.corners_array,
            dof_offsets,
            solution,
        )

        combined_solution = np.concatenate(solution, dtype=np.float64)

        equation_values = call_per_leaf_flex(
            element_collection,
            1,
            np.float64,
            compute_leaf_vector,
            compiled_system.lhs_full,
            unknown_ordering,
            element_collection.orders_array,
            cache_2d,
            element_collection.corners_array,
            vec_fields_array,
            solution,
        )

        main_value = assemble_forcing(
            unknown_ordering,
            element_collection,
            dof_offsets,
            equation_values,
            solution,
        )

        if compiled_system.rhs_codes is not None:
            main_vec = np.array(base_vec, copy=True)  # Make a copy
            # Update RHS implicit terms

            explicit_values = call_per_leaf_flex(
                element_collection,
                1,
                np.float64,
                compute_leaf_vector,
                compiled_system.rhs_codes,
                unknown_ordering,
                element_collection.orders_array,
                cache_2d,
                element_collection.corners_array,
                vec_fields_array,
                solution,
            )

            for off, cnt, vals in (
                (element_offsets[i], dof_offsets[i], explicit_values[i])
                for i in leaf_elements
            ):
                main_vec[off : off + cnt[-1]] += vals
        else:
            main_vec = base_vec

        if lagrange_mat is not None:
            main_value += lagrange_mat.T @ global_lagrange
            main_value = np.concatenate(
                (main_value, lagrange_mat @ combined_solution),
                dtype=np.float64,
            )

        residual = main_vec - main_value
        max_residual = np.abs(residual).max()
        residuals[iter_cnt] = max_residual
        if print_residual:
            if progress_tracker is None:
                progress_tracker = ProgressTracker(
                    atol, max_residual, max_residual, max_iterations, err_width=20
                )
            else:
                progress_tracker.update_iteration(max_residual)
            print(progress_tracker.state_str("{} - {} | {}"), end="\r")
            # print(f"Iteration {iter_cnt} has residual of {max_residual:.4e}", end="\r")

        if not (max_residual > atol and max_residual > max_mag * rtol):
            break

        if compiled_system.nonlin_codes is not None:
            new_matrices = call_per_leaf_flex(
                element_collection,
                2,
                np.float64,
                compute_leaf_matrix,
                compiled_system.nonlin_codes,
                unknown_ordering,
                element_collection.orders_array,
                cache_2d,
                element_collection.corners_array,
                vec_fields_array,
            )
            element_matrices = call_per_leaf_flex(
                element_collection,
                2,
                np.float64,
                lambda ie, m1, m2: m1[ie] + m2[ie],
                new_matrices,
                linear_element_matrices,
            )

            main_mat = assemble_matrix(
                unknown_ordering,
                element_collection,
                dof_offsets,
                element_matrices,
            )
            if lagrange_mat is not None:
                main_mat = sp.block_array(
                    [[main_mat, lagrange_mat.T], [lagrange_mat, None]], format="csc"
                )
            else:
                main_mat = main_mat.tocsc()

            system_decomp = sla.splu(main_mat)
            del main_mat

        d_solution = np.asarray(
            system_decomp.solve(residual),
            dtype=np.float64,
            copy=None,
        )
        sol_updates = extract_from_flat(
            element_collection, element_offsets, relax * d_solution
        )
        # update lagrange multipliers (haha pliers)
        if len(global_lagrange):
            d_lag = d_solution[-global_lagrange.size :]
            global_lagrange += relax * d_lag

        def _update_solution(
            ie: int,
            sol: FlexibleElementArray[np.float64, np.uint32],
            up: FlexibleElementArray[np.float64, np.uint32],
        ) -> npt.NDArray[np.float64]:
            """Compute solution given update value."""
            return sol[ie] + up[ie]

        solution = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            _update_solution,
            solution,
            sol_updates,
        )

        iter_cnt += 1

        del main_vec

    if not return_all_residuals:
        return solution, global_lagrange, iter_cnt, max_residual

    return solution, global_lagrange, iter_cnt, residuals


_T = TypeVar("_T", bound=np.generic)


def extract_from_flat(
    elements: ElementCollection,
    element_offsets: npt.NDArray[np.integer],
    v: npt.NDArray[_T],
) -> FlexibleElementArray[_T, np.uint32]:
    """Extract from flat vector to per-element values."""

    def _extract_single(
        ie: int,
        element_offsets: npt.NDArray[np.integer],
        v: npt.NDArray[_T],
    ) -> npt.NDArray[_T]:
        """Extract DoFs from the vector."""
        begin = int(element_offsets[ie])
        end = int(element_offsets[ie + 1])
        return v[begin:end]

    return call_per_element_flex(
        elements.com,
        1,
        cast(type[_T], v.dtype),
        _extract_single,
        element_offsets,
        v,
    )


@dataclass(frozen=True)
class TimeSettings:
    """Type for defining time settings of the solver.

    Parameters
    ----------
    dt : float
        Time step to take.

    nt : int
        Number of time steps to simulate.

    time_march_relations : dict of (KWeight, KFormUnknown)
        Pairs of weights and unknowns, which determine what equations are treated as time
        marching equations for which unknowns. At least one should be present.

    sample_rate : int, optional
        How often the output is saved. If not specified, every time step is saved. First
        and last steps are always saved.
    """

    dt: float
    nt: int
    time_march_relations: Mapping[KWeight, KFormUnknown]
    sample_rate: int = 1


@dataclass(frozen=True)
class SystemSettings:
    """Type used to hold system information for solving.

    Parameters
    ----------
    system : KFormSystem
        System of equations to solve.

    boundaray_conditions: Sequence of BoundaryCondition2DSteady, optional
        Sequence of boundary conditions to be applied to the system.

    constrained_forms : Sequence of (float, KFormUnknown), optional
        Sequence of 2-form unknowns which must be constrained. These can arrise form
        cases where a continuous variable acts as a Lagrange multiplier on the continuous
        level and only appears in the PDE as a gradient. In that case it will result
        in a singular system if not constrained manually to a fixed value.

        An example of such a case is pressure in Stokes flow or incompressible
        Navier-Stokes equations.

    intial_conditions : Mapping of (KFormUnknown, Callable), optional
        Functions which give initial conditions for different forms.
    """

    system: KFormSystem
    boundary_conditions: Sequence[BoundaryCondition2DSteady] = field(
        default_factory=tuple
    )
    constrained_forms: Sequence[tuple[float, KFormUnknown]] = field(default_factory=tuple)
    initial_conditions: Mapping[
        KFormUnknown,
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike],
    ] = field(default_factory=dict)


def divide_old(
    order: int, level: int, max_level: int
) -> tuple[int | None, tuple[int, int, int, int]]:
    """Keep child order equal to parent and set parent to double the child."""
    del level, max_level
    v = order
    return 2 * order, (v, v, v, v)


@dataclass(frozen=True)
class RefinementSettings:
    """Type used to hold settings related to refinement information.

    Parameters
    ----------
    refinement_levels : int
        Number of mesh refinement levels which can be done. When zero,
        no refinement is done.

    division_predicate : Callable (Element2D, int) -> bool, optional
        Callable used to determine if an element should be divided further. If
        not specified, no refinement is done.

    division_function : OrderDivisionFunction, optional
        Function which determines order of the parent and child elements resulting from
        the division of the element. When not specified, the "old" method is used.
    """

    refinement_levels: int
    division_predicate: Callable[[ElementLeaf2D, int], bool] | None = None
    division_function: OrderDivisionFunction = divide_old


@dataclass(frozen=True)
class SolverSettings:
    r"""Settings used by the non-linear solver.

    Solver finds a solution to the system :eq:`solve_system_equation`, where
    :math:`\mathbb{I}` is the implicit part of the system, :math:`\vec{E}` is the explicit
    part of the system, and :math:`\vec{F}` is the constant part of the system.

    This is done by computing updates to the state :math:`\Delta u` as per
    :eq:`solve_system_iteration`. The iterations stop once the residual
    :math:`\vec{E}\left({\vec{u}}^i\right) + \vec{F} + \vec{I}\left({\vec{u}}^i\right)`
    falls under the specified criterion.

    The stopping criterion has two tolerances - absolute and relative - which both need
    to be met in order for the solver to stop. The absolute criterion checks if the
    highest absolute value of the residual elements is bellow the value specified. The
    relative first scales it by the maximum absolute value of the constant forcing.
    Depending on the system and the solution, these may need to be adjusted. If the system
    is linear, meaning that :math:`\vec{E} = 0` and :math:`\mathbb{I}` is
    not dependant on math:`\vec{u}`, then the solver will terminate in a single iteration.

    Last thing to consider is the relaxation. Sometimes, the system is very stiff and does
    not converge nicely. This can be the case for steady-state calculations of non-linear
    systems with bad initial guesses. In such cases, the correction can be too large and
    overshoot the solution. In such cases, convergence may still be achieved if the update
    is scaled by a "relaxation factor". Conversely, convergence may be slightly sped up
    for some very stable problems if the increment is amplified, meaning the relaxation
    factor is greater than 1.

    .. math::
        :label: solve_system_equation

        \mathbb{I}\left({\vec{u}}\right) {\vec{u}} = \vec{E}\left({\vec{u}}^i
        \right) + \vec{F}


    .. math::
        :label: solve_system_iteration

        \Delta {\vec{u}}^i = \left(\mathbb{I}\left({\vec{u}}^i\right)\right)^{-1} \left(
        \vec{E}\left({\vec{u}}^i\right) + \vec{F} + \vec{I}\left({\vec{u}}^i\right)\right)

    Parameters
    ----------
    maximum_iterations : int, default: 100
        Maximum number of iterations the solver performs.

    relaxation : float, default: 1.0
        Fraction of solution increment to use.

    absolute_tolerance : float, default: 1e-6
        Maximum value of the residual must meet in order for the solution
        to be considered converged.

    relative_tolerance : float, default: 1e-5
        Maximum fraction of the maximum of the right side of the equation the residual
        must meet in order for the solution to be considered converged.
    """

    maximum_iterations: int = 100
    relaxation: float = 1.0
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5


def find_time_carry_indices(
    ie: int,
    unknowns: Sequence[int],
    dof_offsets: FixedElementArray[np.uint32],
    child_count_array: FixedElementArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    """Find what are the indices of DoFs that should be carried for the time march."""
    if int(child_count_array[ie][0]) != 0:
        return np.zeros(0, np.uint32)

    output: list[npt.NDArray[np.uint32]] = list()
    offsets = dof_offsets[ie]
    for iu, u in enumerate(unknowns):
        assert iu == 0 or unknowns[iu] < u, "Unknowns must be sorted."
        output.append(np.arange(offsets[u], offsets[u + 1], dtype=np.uint32))
    return np.concatenate(output, dtype=np.uint32)


@dataclass(frozen=True)
class SolutionStatistics:
    """Information about the solution."""

    element_orders: dict[int, int]
    n_total_dofs: int
    n_leaf_dofs: int
    n_lagrange: int
    n_elems: int
    n_leaves: int
    iter_history: npt.NDArray[np.uint32]
    residual_history: npt.NDArray[np.float64]


@dataclass(frozen=True)
class VmsSettings:
    """Type used for VMS related information."""

    full_system: KFormSystem
    symmetric_part: KFormSystem
    advection_part: KFormSystem
