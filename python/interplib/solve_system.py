"""New implementation of solve functions."""

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms
from interplib._interp import dlagrange1d, lagrange1d
from interplib._mimetic import Surface, compute_element_explicit, compute_element_matrices
from interplib.element import (
    ElementCollection,
    ElementSide,
    FixedElementArray,
    FlexibleElementArray,
    UnknownFormOrder,
    UnknownOrderings,
    call_per_element_fix,
    call_per_element_flex,
    compute_dof_sizes,
    compute_lagrange_sizes,
    jacobian,
    poly_x,
    poly_y,
)
from interplib.kforms.eval import CompiledSystem
from interplib.kforms.kform import (
    KBoundaryProjection,
    KElementProjection,
    KExplicit,
    KSum,
    KWeight,
)
from interplib.mimetic.mimetic2d import BasisCache, Element2D, ElementLeaf2D, Mesh2D
from interplib.system2d import (
    OrderDivisionFunction,
    SolutionStatisticsUnsteady,
    VectorFieldFunction,
    continuity_child_matrices,
    continuity_matrices,
    divide_old,
    vtk_lagrange_ordering,
    # find_strong_bc_edge_indices,
)


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
        out = cbl + cbr + ctl + ctr + [new_e]

    else:
        out = [e]

    return out


def reconstruct(
    corners: npt.NDArray[np.float64],
    k: int,
    coeffs: npt.ArrayLike,
    xi: npt.ArrayLike,
    eta: npt.ArrayLike,
    order_1: int,
    order_2: int,
    cache_1: BasisCache,
    cache_2: BasisCache,
    /,
) -> npt.NDArray[np.float64]:
    """Reconstruct a k-form on the element."""
    assert k >= 0 and k < 3
    assert cache_1.basis_order == order_1
    assert cache_2.basis_order == order_2
    out: float | npt.NDArray[np.floating] = 0.0
    c = np.asarray(coeffs, dtype=np.float64, copy=None)
    if c.ndim != 1:
        raise ValueError("Coefficient array must be one dimensional.")

    if k == 0:
        vals_xi = lagrange1d(cache_1.nodes_1d, xi)
        vals_eta = lagrange1d(cache_2.nodes_1d, eta)
        for i in range(order_2 + 1):
            v = vals_eta[..., i]
            for j in range(order_1 + 1):
                u = vals_xi[..., j]
                out += c[i * (order_1 + 1) + j] * (u * v)

    elif k == 1:
        # TODO: check if reconstruction is done correctly on non-unit domain.
        values_xi = lagrange1d(cache_1.nodes_1d, xi)
        values_eta = lagrange1d(cache_2.nodes_1d, eta)
        in_dvalues_xi = dlagrange1d(cache_1.nodes_1d, xi)
        in_dvalues_eta = dlagrange1d(cache_2.nodes_1d, eta)
        dvalues_xi = tuple(accumulate(-in_dvalues_xi[..., i] for i in range(order_1)))
        dvalues_eta = tuple(accumulate(-in_dvalues_eta[..., i] for i in range(order_2)))
        (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
        det = j00 * j11 - j10 * j01
        out_xi: float | npt.NDArray[np.floating] = 0.0
        out_eta: float | npt.NDArray[np.floating] = 0.0
        for i1 in range(order_2 + 1):
            v1 = values_eta[..., i1]
            for j1 in range(order_1):
                u1 = dvalues_xi[j1]
                out_eta += c[i1 * order_1 + j1] * u1 * v1

        for i1 in range(order_2):
            v1 = dvalues_eta[i1]
            for j1 in range(order_1 + 1):
                u1 = values_xi[..., j1]

                out_xi += c[(order_2 + 1) * order_1 + i1 * (order_1 + 1) + j1] * u1 * v1
        out = np.stack(
            (out_xi * j00 + out_eta * j10, out_xi * j01 + out_eta * j11), axis=-1
        )
        out /= det[..., None]

    elif k == 2:
        in_dvalues_xi = dlagrange1d(cache_1.nodes_1d, xi)
        in_dvalues_eta = dlagrange1d(cache_2.nodes_1d, eta)
        dvalues_xi = tuple(accumulate(-in_dvalues_xi[..., i] for i in range(order_1)))
        dvalues_eta = tuple(accumulate(-in_dvalues_eta[..., i] for i in range(order_2)))
        (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
        det = j00 * j11 - j10 * j01
        for i1 in range(order_2):
            v1 = dvalues_eta[i1]
            for j1 in range(order_1):
                u1 = dvalues_xi[j1]
                out += c[i1 * order_1 + j1] * u1 * v1

        out /= det
    else:
        raise ValueError(f"Order of the differential form {k} is not valid.")

    return np.array(out, np.float64, copy=None)


def compute_vector_fields_nonlin(
    system: kforms.KFormSystem,
    leaf_elements: Sequence[int] | npt.NDArray[np.integer],
    cache: Mapping[int, BasisCache],
    vector_fields: Sequence[VectorFieldFunction | kforms.KFormUnknown],
    corners: FixedElementArray[np.float64],
    orders: FixedElementArray[np.uint32],
    unknown_offsets: FixedElementArray[np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32] | None,
) -> tuple[npt.NDArray[np.uint64], tuple[npt.NDArray[np.float64], ...]]:
    """Evaluate vector fields which may be non-linear."""
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(len(leaf_elements) + 1, np.uint64)

    for i_out, e in enumerate(leaf_elements):
        order_1, order_2 = orders[e]
        e_cache_1 = cache[order_1]
        e_cache_2 = cache[order_2]
        e_corners = corners[e]
        # Extract element DoFs
        x = poly_x(
            e_corners[:, 0],
            e_cache_1.int_nodes_1d[None, :],
            e_cache_2.int_nodes_1d[:, None],
        )
        y = poly_y(
            e_corners[:, 1],
            e_cache_1.int_nodes_1d[None, :],
            e_cache_2.int_nodes_1d[:, None],
        )
        for i, vec_fld in enumerate(vector_fields):
            if isinstance(vec_fld, kforms.KFormUnknown):
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
                        e_cache_1.int_nodes_1d[None, :],
                        e_cache_2.int_nodes_1d[:, None],
                        order_1,
                        order_2,
                        e_cache_1,
                        e_cache_2,
                    )
                    if vec_fld.order != 1:
                        vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
                else:
                    vf = np.zeros(
                        (
                            e_cache_2.integration_order + 1,
                            e_cache_1.integration_order + 1,
                            2,
                        ),
                        np.float64,
                    )
            else:
                vf = vec_fld(x, y)
            vec_field_lists[i].append(np.reshape(vf, (-1, 2)))
        vec_field_offsets[i_out + 1] = vec_field_offsets[i_out] + (
            e_cache_1.integration_order + 1
        ) * (e_cache_2.integration_order + 1)
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    return vec_field_offsets, vec_fields


def rhs_2d_element_projection(
    right: KElementProjection,
    corners: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    cache_1: BasisCache,
    cache_2: BasisCache,
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element2D
        The element on which the projection is evaluated on.
    cache : BasisCache
        Cache for the correct element order.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    # TODO: don't recompute basis, just reuse the cached values.
    assert cache_1.basis_order == order_1
    assert cache_2.basis_order == order_1
    fn = right.func

    n_dof: int
    if right.weight.order == 0:
        n_dof = (order_1 + 1) * (order_2 + 1)
    elif right.weight.order == 1:
        n_dof = (order_1 + 1) * order_2 + order_1 * (order_2 + 1)
    elif right.weight.order == 2:
        n_dof = order_1 * order_2
    else:
        raise ValueError(f"Invalid weight order {right.weight.order}.")

    if fn is None:
        return np.zeros(n_dof)

    out_vec = np.empty(n_dof)

    basis_vals: list[npt.NDArray[np.floating]] = list()

    nodes_1 = cache_1.int_nodes_1d
    weights_1 = cache_1.int_weights_1d
    nodes_2 = cache_2.int_nodes_1d
    weights_2 = cache_2.int_weights_1d

    (j00, j01), (j10, j11) = jacobian(corners, nodes_1[None, :], nodes_2[:, None])
    det = j00 * j11 - j10 * j01

    real_x = poly_x(corners[:, 0], nodes_1[None, :], nodes_2[:, None])
    real_y = poly_y(corners[:, 1], nodes_1[None, :], nodes_2[:, None])
    f_vals = fn(real_x, real_y)
    weights_2d = weights_1[None, :] * weights_2[:, None]

    # Deal with vectors first. These need special care.
    if right.weight.order == 1:
        values1 = lagrange1d(cache_1.nodes_1d, nodes_1)
        d_vals1 = dlagrange1d(cache_1.nodes_1d, nodes_1)
        values2 = lagrange1d(cache_2.nodes_1d, nodes_2)
        d_vals2 = dlagrange1d(cache_2.nodes_1d, nodes_2)
        d_values1 = tuple(accumulate(-d_vals1[..., i] for i in range(order_1)))
        d_values2 = tuple(accumulate(-d_vals2[..., i] for i in range(order_2)))

        new_f0 = j00 * f_vals[..., 0] + j01 * f_vals[..., 1]
        new_f1 = j10 * f_vals[..., 0] + j11 * f_vals[..., 1]

        for i1 in range(order_2 + 1):
            v1 = values2[..., i1]
            for j1 in range(order_1):
                u1 = d_values1[j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[i1 * order_1 + j1] = np.sum(basis1 * weights_2d * new_f1)

        for i1 in range(order_2):
            v1 = d_values2[i1]
            for j1 in range(order_1 + 1):
                u1 = values1[..., j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[order_1 * (order_2 + 1) + i1 * (order_1 + 1) + j1] = np.sum(
                    basis1 * weights_2d * new_f0
                )
        return out_vec

    if right.weight.order == 2:
        d_vals1 = dlagrange1d(cache_1.nodes_1d, nodes_1)
        d_vals2 = dlagrange1d(cache_2.nodes_1d, nodes_2)
        d_values1 = tuple(accumulate(-d_vals1[..., i] for i in range(order_1)))
        d_values2 = tuple(accumulate(-d_vals2[..., i] for i in range(order_2)))
        for i1 in range(order_2):
            v1 = d_values2[i1]
            for j1 in range(order_1):
                u1 = d_values1[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof

    elif right.weight.order == 0:
        values1 = lagrange1d(cache_1.nodes_1d, nodes_1)
        values2 = lagrange1d(cache_2.nodes_1d, nodes_2)
        for i1 in range(order_2 + 1):
            v1 = values2[..., i1]
            for j1 in range(order_1 + 1):
                u1 = values1[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof
        weights_2d *= det

    else:
        raise ValueError(f"Invalid weight order {right.weight.order}.")

    # Compute rhs integrals
    for i, bv in enumerate(basis_vals):
        out_vec[i] = np.sum(bv * f_vals * weights_2d)

    return out_vec


def _extract_rhs_2d(
    proj: Sequence[tuple[float, KExplicit]],
    weight: KWeight,
    corners: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    cache_1: BasisCache,
    cache_2: BasisCache,
) -> npt.NDArray[np.float64]:
    """Extract the rhs resulting from element projections."""
    if weight.order == 0:
        n_out = (order_1 + 1) * (order_2 + 1)
    elif weight.order == 1:
        n_out = (order_1 + 1) * order_2 + order_1 * (order_2 + 1)
    elif weight.order == 2:
        n_out = order_1 * order_2
    else:
        raise ValueError(f"Invalid weight order {weight.order}.")

    vec = np.zeros(n_out)

    for k, f in filter(lambda v: isinstance(v[1], KElementProjection), proj):
        assert isinstance(f, KElementProjection)
        rhs = rhs_2d_element_projection(f, corners, order_1, order_2, cache_1, cache_2)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def compute_element_rhs(
    ie: int,
    system: kforms.KFormSystem,
    cache: Mapping[int, BasisCache],
    corners: FixedElementArray[np.float64],
    orders: FixedElementArray[np.uint32],
    child_count_array: FixedElementArray[np.uint32],
) -> npt.NDArray[np.float64]:
    """Compute rhs for an element."""
    if int(child_count_array[ie][0]) != 0:
        return np.zeros(0, np.float64)

    vecs: list[npt.NDArray[np.float64]] = list()
    order_1: int
    order_2: int
    element_corners = corners[ie]
    order_1, order_2 = orders[ie]
    for equation in system.equations:
        vecs.append(
            _extract_rhs_2d(
                equation.right.explicit_terms,
                equation.weight,
                element_corners,
                order_1,
                order_2,
                cache[order_1],
                cache[order_2],
            )
        )

    return np.concatenate(vecs, dtype=np.float64)


@dataclass(frozen=True)
class ElementConstraint:
    """Type intended to enforce a constraint on an element."""

    i_e: int
    dofs: npt.NDArray[np.uint32]
    coeffs: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Constraint:
    """Type used to specify constraints on degrees of freedom."""

    rhs: float
    element_constraints: tuple[ElementConstraint, ...]

    def __init__(self, rhs: float, *element_constraints: ElementConstraint) -> None:
        object.__setattr__(self, "rhs", float(rhs))
        object.__setattr__(self, "element_constraints", element_constraints)


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
    child_tl: int,
    child_tr: int,
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
        )
        for ic in children
    ]
    sizes = [mat.shape[0] for mat in child_matrices]
    offsets = {int(ic): int(np.sum(sizes[:i])) for i, ic in enumerate(children)}
    offsets[ie] = sum(sizes)

    # Lagrange muliplier block
    constraint_equations = _parent_child_equations(
        unknown_ordering, collection, dof_offsets_array, ie, *children
    )
    rows: list[npt.NDArray[np.uint32]] = list()
    cols: list[npt.NDArray[np.uint32]] = list()
    vals: list[npt.NDArray[np.float64]] = list()
    for ic, con in enumerate(constraint_equations):
        assert con.rhs == 0
        for ec in con.element_constraints:
            offset = offsets[ec.i_e]
            cols.append(ec.dofs + offset)
            rows.append(np.full_like(ec.dofs, ic, np.uint32))
            vals.append(ec.coeffs)

    lagmat = sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(constraint_equations), sum(sizes) + dof_offsets[-1]),
    )

    return cast(
        sp.csr_array,
        sp.block_array(
            ((sp.block_diag((*child_matrices, self_matrix)), lagmat.T), (lagmat, None)),
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

    return np.concatenate((*child_vectors, self_vec, lagvals), dtype=np.float64)


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
        [*(forcing_arrays[ic] for ic in children), forcing_arrays[ie], vals],
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
    unique_orders: npt.NDArray[np.uint32],
    dof_offsets: FixedElementArray[np.uint32],
) -> list[Constraint]:
    """Create equations enforcing 0-form continuity between top level elements."""
    # Continuity of 0-forms on the non-corner DoFs
    continuity_equations: list[Constraint] = list()
    if (unique_orders.size > 1) or (1 not in unique_orders):
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
    system: kforms.KFormSystem,
    mesh: Mesh2D,
    top_indices: npt.NDArray[np.uint32],
    unknown_orders: UnknownOrderings,
    elements: ElementCollection,
    unique_orders: npt.NDArray[np.uint32],
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
                mesh, top_indices, elements, unknown_orders, unique_orders, dof_offsets
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
    caches: MutableMapping[int, BasisCache],
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
                caches,
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
                caches,
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
                caches,
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
                caches,
            )

        return current

    side_order = elements.get_element_order_on_side(ie, side)
    if side_order not in caches:
        caches[side_order] = BasisCache(side_order, 2 * side_order)

    basis_cache = caches[side_order]
    ndir = 2 * ((side.value & 2) >> 1) - 1
    i0 = side.value - 1
    i1 = side.value & 3
    corners = elements.corners_array[ie]
    p0: tuple[float, float] = corners[i0]
    p1: tuple[float, float] = corners[i1]
    # ndir, p0, p1 = _endpoints_from_line(e, i_side)
    dx = (p1[0] - p0[0]) / 2
    xv = (p1[0] + p0[0]) / 2 + dx * basis_cache.int_nodes_1d
    dy = (p1[1] - p0[1]) / 2
    yv = (p1[1] + p0[1]) / 2 + dy * basis_cache.int_nodes_1d
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
            basis = basis_cache.nodal_1d
            f_vals = (
                f_vals[..., 0] * dx + f_vals[..., 1] * dy
            ) * basis_cache.int_weights_1d

        elif form_order == UnknownFormOrder.FORM_ORDER_1:
            # Integral with the normal basis
            basis = basis_cache.edge_1d
            f_vals *= basis_cache.int_weights_1d * ndir

        else:
            raise ValueError(f"Unknown/Invalid weak form order {form_order=}.")

        vals[:] += np.sum(f_vals[..., None] * basis, axis=0)

    return (ElementConstraint(ie, dofs, vals),)


def _element_strong_boundary_condition(
    elements: ElementCollection,
    ie: int,
    side: ElementSide,
    unknown_orders: UnknownOrderings,
    unknown_index: int,
    dof_offsets: FixedElementArray[np.uint32],
    strong_bc: kforms.BoundaryCondition2DSteady,
    caches: MutableMapping[int, BasisCache],
    skip_first: bool,
    skip_last: bool,
) -> tuple[ElementConstraint, ...]:
    """Determine boundary conditions given an element and a side."""
    children = elements.child_array[ie]

    if children.size != 0:
        # Node, has children
        current: tuple[ElementConstraint, ...] = tuple()
        # For children skipping:
        # Skip first if: (on start of the side AND skip_first is True) OR (end of side)
        # Skip last if: on end of the side AND skip_last is True
        # So only time first not skipped if: skip_first is True AND beginning of side
        if side == ElementSide.SIDE_LEFT or side == ElementSide.SIDE_BOTTOM:
            current += _element_strong_boundary_condition(
                elements,
                int(children[0]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                strong_bc,
                caches,
                (skip_first and side == ElementSide.SIDE_BOTTOM)
                or side == ElementSide.SIDE_LEFT,
                skip_last and side == ElementSide.SIDE_LEFT,
            )
        if side == ElementSide.SIDE_BOTTOM or side == ElementSide.SIDE_RIGHT:
            current += _element_strong_boundary_condition(
                elements,
                int(children[1]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                strong_bc,
                caches,
                (skip_first and side == ElementSide.SIDE_RIGHT)
                or side == ElementSide.SIDE_BOTTOM,
                skip_last and side == ElementSide.SIDE_BOTTOM,
            )
        if side == ElementSide.SIDE_RIGHT or side == ElementSide.SIDE_TOP:
            current += _element_strong_boundary_condition(
                elements,
                int(children[2]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                strong_bc,
                caches,
                (skip_first and side == ElementSide.SIDE_TOP)
                or side == ElementSide.SIDE_RIGHT,
                skip_last and side == ElementSide.SIDE_RIGHT,
            )
        if side == ElementSide.SIDE_TOP or side == ElementSide.SIDE_LEFT:
            current += _element_strong_boundary_condition(
                elements,
                int(children[3]),
                side,
                unknown_orders,
                unknown_index,
                dof_offsets,
                strong_bc,
                caches,
                (skip_first and side == ElementSide.SIDE_LEFT)
                or side == ElementSide.SIDE_TOP,
                skip_last and side == ElementSide.SIDE_TOP,
            )

        return current

    side_order = elements.get_element_order_on_side(ie, side)
    if side_order not in caches:
        caches[side_order] = BasisCache(side_order, 2 * side_order)

    basis_cache = caches[side_order]
    ndir = 2 * ((side.value & 2) >> 1) - 1
    i0 = side.value - 1
    i1 = side.value & 3
    corners = elements.corners_array[ie]
    p0: tuple[float, float] = corners[i0]
    p1: tuple[float, float] = corners[i1]
    # ndir, p0, p1 = _endpoints_from_line(e, i_side)
    dx = (p1[0] - p0[0]) / 2
    xv = (p1[0] + p0[0]) / 2 + dx * basis_cache.nodes_1d
    dy = (p1[1] - p0[1]) / 2
    yv = (p1[1] + p0[1]) / 2 + dy * basis_cache.nodes_1d
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
        lnds = basis_cache.int_nodes_1d
        wnds = basis_cache.int_weights_1d
        for i in range(side_order):
            xc = (xv[i + 1] + xv[i]) / 2 + (xv[i + 1] - xv[i]) / 2 * lnds
            yc = (yv[i + 1] + yv[i]) / 2 + (yv[i + 1] - yv[i]) / 2 * lnds
            dx = (xv[i + 1] - xv[i]) / 2
            dy = (yv[i + 1] - yv[i]) / 2
            normal = ndir * np.array((dy, -dx))
            fvals = strong_bc.func(xc, yc)
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
    strong_bcs: Sequence[Sequence[kforms.BoundaryCondition2DSteady]],
    caches: MutableMapping[int, BasisCache],
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

    strong_bcs : Sequence of Sequence of kforms.BoundaryCondition2DSteady
        Boundary conditions grouped per weight functions and correctly ordered to match
        the order of weight functions in the system.

    caches : MutableMapping of (int, BasisCache)
        Caches which can be used and appended to.

    Returns
    -------
    tuple of ElementConstraint
        Weak boundary conditions in a specific notation. Each of these means
        that for element given by ``ElementConstraint.i_e``, all equations with
        indices ``ElementConstraint.dofs`` should have the value
        ``ElementConstraint.coeffs`` added to them.

    tuple of ElementConstraint
        Strong boundary conditions in a specific notation. Each of these means
        that for element given by ``ElementConstraint.i_e``, all dofs with
        indices ``ElementConstraint.dofs`` should be constrained to value
        ``ElementConstraint.coeffs``.
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
                        caches,
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
                        caches,
                    )
                )

            else:
                # Strong not given, but also no weak ones.
                pass

    return tuple(s_bcs), tuple(w_bcs)


def reconstruct_mesh_from_solution(
    system: kforms.KFormSystem,
    recon_order: int | None,
    element_collection: ElementCollection,
    caches: Mapping[int, BasisCache],
    dof_offsets: FixedElementArray[np.uint32],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> pv.UnstructuredGrid:
    """Reconstruct the unknown differential forms."""
    build: dict[kforms.KFormUnknown, list[npt.NDArray[np.float64]]] = {
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
        recon_nodes_1d = caches[element_order].nodes_1d
        ordering = vtk_lagrange_ordering(element_order) + node_cnt
        node_array.append(np.concatenate(((ordering.size,), ordering)))
        node_cnt += ordering.size
        corners = element_collection.corners_array[ie]
        ex = poly_x(corners[:, 0], recon_nodes_1d[None, :], recon_nodes_1d[:, None])
        ey = poly_y(corners[:, 1], recon_nodes_1d[None, :], recon_nodes_1d[:, None])

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
                recon_nodes_1d[None, :],
                recon_nodes_1d[:, None],
                order_1,
                order_2,
                caches[order_1],
                caches[order_2],
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


def non_linear_solve_run(
    system: kforms.KFormSystem,
    max_iterations: int,
    relax: float,
    atol: float,
    rtol: float,
    print_residual: bool,
    unknown_ordering: UnknownOrderings,
    element_collection: ElementCollection,
    leaf_elements: npt.NDArray[np.integer],
    cache: Mapping[int, BasisCache],
    compiled_system: CompiledSystem,
    explicit_vec: npt.NDArray[np.float64],
    bl: npt.NDArray[np.float64],
    br: npt.NDArray[np.float64],
    tr: npt.NDArray[np.float64],
    tl: npt.NDArray[np.float64],
    orde: npt.NDArray[np.uint32],
    c_ser,
    dof_offsets: FixedElementArray[np.uint32],
    element_offsets: npt.NDArray[np.uint32],
    linear_element_matrices: FlexibleElementArray[np.float64, np.uint32],
    time_carry_index_array: FlexibleElementArray[np.uint32, np.uint32] | None,
    time_carry_term: FlexibleElementArray[np.float64, np.uint32] | None,
    solution: FlexibleElementArray[np.float64, np.uint32],
    global_lagrange: npt.NDArray[np.float64],
    max_mag: float,
    vector_fields: Sequence[VectorFieldFunction | kforms.KFormUnknown],
    system_decomp: sla.SuperLU,
    lagrange_mat: sp.csr_array | None,
    return_all_residuals: bool = False,
):
    """Run the iterative non-linear solver.

    Based on how the compiled system looks, this may only take a single iteration,
    otherwise, it may run for as long as it needs to converge.
    """
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
    while iter_cnt < max_iterations:
        # Recompute vector fields
        # Compute vector fields at integration points for leaf elements
        vec_field_offsets, vec_fields = compute_vector_fields_nonlin(
            system,
            leaf_elements,
            cache,
            vector_fields,
            element_collection.corners_array,
            element_collection.orders_array,
            dof_offsets,
            solution,
        )

        combined_solution = np.concatenate(solution, dtype=np.float64)
        eq_vals = compute_element_explicit(
            combined_solution,
            element_offsets[leaf_elements],
            [f.order for f in system.unknown_forms],
            compiled_system.lhs_full,
            bl,
            br,
            tr,
            tl,
            orde,
            vec_fields,
            vec_field_offsets,
            c_ser,
        )

        equation_values = assign_leaves(element_collection, 1, np.float64, eq_vals)

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
            computed_explicit = compute_element_explicit(
                combined_solution,
                element_offsets[leaf_elements],
                [f.order for f in system.unknown_forms],
                compiled_system.rhs_codes,
                bl,
                br,
                tr,
                tl,
                orde,
                vec_fields,
                vec_field_offsets,
                c_ser,
            )
            explicit_values = assign_leaves(
                element_collection, 1, np.float64, computed_explicit
            )
            for ie, (off, cnt, vals) in enumerate(
                zip(element_offsets[leaf_elements], dof_offsets, explicit_values)
            ):
                main_vec[off : off + cnt[-2]] += vals
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
            print(f"Iteration {iter_cnt} has residual of {max_residual:.4e}", end="\r")

        if not (max_residual > atol and max_residual > max_mag * rtol):
            break

        if compiled_system.nonlin_codes is not None:
            new_element_matrices = [
                m + linear_element_matrices[i]
                for i, m in enumerate(
                    compute_element_matrices(
                        [f.order for f in system.unknown_forms],
                        compiled_system.nonlin_codes,
                        bl,
                        br,
                        tr,
                        tl,
                        orde,
                        vec_fields,
                        vec_field_offsets,
                        c_ser,
                    )
                )
            ]
            element_matrices = assign_leaves(
                element_collection, 1, np.float64, new_element_matrices
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


def assign_leaves(
    element_collection: ElementCollection,
    ndim: int,
    dtype: type[_T],
    iterable: Sequence[npt.NDArray[_T]],
) -> FlexibleElementArray[_T, np.uint32]:
    """Assign elements of the sequence object to the leaves."""

    def _assign_element_matrix(
        ie: int,
        values: Iterator[npt.NDArray[_T]],
        child_count: FixedElementArray[np.uint32],
    ) -> npt.NDArray[_T]:
        """Extract element matrices."""
        cnt = int(child_count[ie][0])
        if cnt != 0:
            return np.zeros([0] * ndim, dtype=dtype)
        return next(values)

    linear_element_matrices = call_per_element_flex(
        element_collection.com,
        ndim,
        dtype,
        _assign_element_matrix,
        iter(iterable),
        element_collection.child_count_array,
    )

    return linear_element_matrices


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
    """Type for defining time settings of the solver."""

    dt: float
    nt: int
    time_march_relations: Mapping[KWeight, kforms.KFormUnknown]
    sample_rate: int = 1


@dataclass(frozen=True)
class SystemSettings:
    """Type used to hold system information for solving."""

    system: kforms.KFormSystem
    boundary_conditions: Sequence[kforms.BoundaryCondition2DSteady] = field(
        default_factory=tuple
    )
    constrained_forms: Sequence[tuple[float, kforms.KFormUnknown]] = field(
        default_factory=tuple
    )
    initial_conditions: Mapping[
        kforms.KFormUnknown,
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike],
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementSettings:
    """Type used to hold settings related to refinement information."""

    refinement_levels: int
    division_predicate: Callable[[ElementLeaf2D, int], bool] | None = None
    division_function: OrderDivisionFunction = divide_old


@dataclass(frozen=True)
class NonlinearSolverSettings:
    """Settings used by the non-linear solver."""

    maximum_iterations: int = 100
    relaxation: float = 1.0
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5


def _find_time_carry_indices(
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
        output.append(np.arange(offsets[iu], offsets[iu + 1], dtype=np.uint32))
    return np.concatenate(output, dtype=np.uint32)


def solve_system_2d_unsteady(
    mesh: Mesh2D,
    system_settings: SystemSettings,
    refinement_settings: RefinementSettings = RefinementSettings(
        refinement_levels=0,
        division_predicate=None,
        division_function=divide_old,
    ),
    solver_settings: NonlinearSolverSettings = NonlinearSolverSettings(
        maximum_iterations=100,
        relaxation=1,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    ),
    time_settings: TimeSettings | None = None,
    *,
    recon_order: int | None = None,
    print_residual: bool = False,
) -> tuple[Sequence[pv.UnstructuredGrid], SolutionStatisticsUnsteady]:
    """Solve the unsteady system on the specified mesh.

    Parameters
    ----------
    system : kforms.KFormSystem
        System of equations to solve.

    mesh : Mesh2D
        Mesh on which to solve the system on.

    dt : float
        Time step to take.

    nt : int
        Number of time steps to simulate.

    time_march_relations : dict of (KWeight, KFormUnknown)
        Pairs of weights and unknowns, which determine what equations are treated as time
        marching equations for which unknowns. At least one should be present.

    intial_conditions : Mapping of (KFormUnknown, Callable), optional
        Functions which give initial conditions for different forms.

    boundaray_conditions: Sequence of kforms.BoundaryCondition2DStrong, optional
        Sequence of boundary conditions to be applied to the system.

    refinement_levels : int, default: 0
        Number of mesh refinement levels which can be done. When zero
        (default value) no refinement is done.

    div_predicate : Callable (Element2D) -> bool, optional
        Callable used to determine if an element should be divided further.

    max_iterations : int, default: 100
        Maximum number of Newton-Raphson iterations solver performs.

    relax : float, default: 1.0
        Fraction of Newton-Raphson increment to use.

    atol : float, default: 1e-6
        Maximum value of the residual must meet in order for the solution
        to be considered converged.

    rtol : float, default: 1e-5
        Maximum fraction of the maximum of the right side of the equation the residual
        must meet in order for the solution to be considered converged.

    sample_rate : int, optional
        How often the output is saved. If not specified, every time step is saved. First
        and last steps are always saved.

    division_function : OrderDivisionFunction, optional
        Function which determines order of the parent and child elements resulting from
        the division of the element. When not specified, the "old" method is used.

    recon_order : int, optional
        When specified, all elements will be reconstructed using this polynomial order.
        Otherwise, they are reconstructed with their own order.

    print_residual : bool, default: False
        Print the maximum of the absolute value of the residual for each iteration of the
        Newton-Raphson method.

    constrained_forms : Sequence of (float, KFormUnknown), optional
        Sequence of 2-form unknowns which must be constrained. These can arrise form
        cases where a continuous variable acts as a Lagrange multiplier on the continuous
        level and only appears in the PDE as a gradient. In that case it will result
        in a singular system if not constrained manually to a fixed value.

        An example of such a case is pressure in Stokes flow or incompressible
        Navier-Stokes equations.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Reconstructed solution as an unstructured grid of VTK's "lagrange quadrilateral"
        cells. This reconstruction is done on the nodal basis for all unknowns.
    stats : SolutionStatisticsNonLin
        Statistics about the solution. This can be used for convergence tests or timing.
    """
    system = system_settings.system

    constrained_forms = system_settings.constrained_forms
    boundary_conditions = system_settings.boundary_conditions

    for _, form in constrained_forms:
        if form not in system.unknown_forms:
            raise ValueError(
                f"Form {form} which is to be zeroed is not involved in the system."
            )

        if boundary_conditions and form in (bc.form for bc in boundary_conditions):
            raise ValueError(
                f"Form {form} can not be zeroed because it is involved in a strong "
                "boundary condition."
            )

    # Make elements into a rectree
    lists = [
        check_and_refine(
            refinement_settings.division_predicate,
            refinement_settings.division_function,
            mesh.get_element(ie),
            0,
            refinement_settings.refinement_levels,
        )
        for ie in range(mesh.n_elements)
    ]
    element_list = sum(lists, start=[])
    element_collection = ElementCollection(element_list)

    # Make element matrices and vectors
    cache: dict[int, BasisCache] = dict()
    unique_orders = element_collection.orders_array.unique()
    for order in (int(order) for order in unique_orders):
        cache[order] = BasisCache(order, order + 2)

    if recon_order is not None and recon_order not in cache:
        cache[int(recon_order)] = BasisCache(int(recon_order), int(recon_order))

    vector_fields = system.vector_fields

    # Create modified system to make it work with time marching.
    if time_settings is not None:
        if time_settings.sample_rate < 1:
            raise ValueError("Sample rate can not be less than 1.")

        if len(time_settings.time_march_relations) < 1:
            raise ValueError("Problem has no time march relations.")

        for w, u in time_settings.time_march_relations.items():
            if u not in system.unknown_forms:
                raise ValueError(f"Unknown form {u} is not in the system.")
            if w not in system.weight_forms:
                raise ValueError(f"Weight form {w} is not in the system.")
            if u.primal_order != w.primal_order:
                raise ValueError(
                    f"Forms {u} and {w} in the time march relation can not be used, as "
                    f"they have differing primal orders ({u.primal_order} vs "
                    f"{w.primal_order})."
                )

        time_march_indices = tuple(
            (
                system.unknown_forms.index(time_settings.time_march_relations[eq.weight])
                if eq.weight in time_settings.time_march_relations
                else None
            )
            for eq in system.equations
        )

        new_equations: list[kforms.KEquation] = list()
        for ie, (eq, m_idx) in enumerate(zip(system.equations, time_march_indices)):
            if m_idx is None:
                new_equations.append(eq)
            else:
                new_equations.append(
                    eq.left
                    + 2
                    / time_settings.dt
                    * (system.weight_forms[m_idx] * system.unknown_forms[m_idx])
                    == eq.right
                )

        system = kforms.KFormSystem(*new_equations)
        del new_equations

    compiled_system = CompiledSystem(system)

    # Make a system that can be used to perform an L2 projection for the initial
    # conditions.
    project_equations: list[kforms.KEquation] = list()
    for ie, eq in enumerate(system.equations):
        base_form = eq.weight.base_form
        proj_rhs = (
            0
            if base_form not in system_settings.initial_conditions
            else eq.weight @ system_settings.initial_conditions[base_form]
        )
        proj_lhs = eq.weight * eq.weight.base_form
        project_equations.append(proj_lhs == proj_rhs)  # type: ignore

    projection_system = kforms.KFormSystem(*project_equations)
    projection_compiled = CompiledSystem(projection_system)
    projection_codes = projection_compiled.linear_codes
    del project_equations

    # Explicit right side
    explicit_vec: npt.NDArray[np.float64]

    # Prepare for evaluation of matrices/vectors
    corners = np.stack([v for v in element_collection.corners_array], axis=0)
    leaf_elements = np.flatnonzero(
        np.concatenate(element_collection.child_count_array.values) == 0
    )
    bl = corners[leaf_elements, 0]
    br = corners[leaf_elements, 1]
    tr = corners[leaf_elements, 2]
    tl = corners[leaf_elements, 3]
    # NOTE: does not work with differing orders yet
    orde = np.array(element_collection.orders_array)[leaf_elements, 0]
    c_ser = tuple(cache[o].c_serialization() for o in cache if o in orde)

    # Release cache element memory. If they will be needed in the future,
    # they will be recomputed, but they consume LOTS of memory

    linear_vectors = call_per_element_flex(
        element_collection.com,
        1,
        np.float64,
        compute_element_rhs,
        system,
        cache,
        element_collection.corners_array,
        element_collection.orders_array,
        element_collection.child_count_array,
    )

    unknown_ordering = UnknownOrderings(*(form.order for form in system.unknown_forms))
    dof_sizes = compute_dof_sizes(element_collection, unknown_ordering)
    lagrange_counts = compute_lagrange_sizes(element_collection, unknown_ordering)
    dof_offsets = call_per_element_fix(
        element_collection.com,
        np.uint32,
        dof_sizes.shape[0] + 1,
        lambda i, x: np.pad(np.cumsum(x[i]), (1, 0)),
        dof_sizes,
    )
    total_dof_counts = call_per_element_fix(
        element_collection.com,
        np.uint32,
        1,
        lambda i, x, y: x[i][-1] + y[i],
        dof_offsets,
        lagrange_counts,
    )

    solution = FlexibleElementArray(element_collection.com, np.float64, total_dof_counts)

    if system_settings.initial_conditions:
        initial_vectors = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            compute_element_rhs,
            projection_system,
            cache,
            element_collection.corners_array,
            element_collection.orders_array,
            element_collection.child_count_array,
        )

        projection_matrices = compute_element_matrices(
            [f.order for f in system.unknown_forms],
            projection_compiled.linear_codes,
            bl,
            br,
            tr,
            tl,
            orde,
            tuple(),
            np.zeros((orde.size + 1), np.uint64),
            c_ser,
        )
        distributed_projections = assign_leaves(
            element_collection, 2, np.float64, projection_matrices
        )
        del projection_matrices

        def _inverse_for_leaves(
            ie: int,
            child_counts: FixedElementArray[np.uint32],
            mat: FlexibleElementArray[np.float64, np.uint32],
            vec: FlexibleElementArray[np.float64, np.uint32],
            total_dofs: FixedElementArray[np.uint32],
        ) -> npt.NDArray[np.float64]:
            """Compute inverse for each leaf element."""
            n_dofs = int(total_dofs[ie][0])
            if int(child_counts[ie][0]) != 0:
                return np.zeros(n_dofs, np.float64)

            res = np.astype(np.linalg.solve(mat[ie], vec[ie]), np.float64, copy=False)
            assert res.size == n_dofs
            return res

        initial_solution = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            _inverse_for_leaves,
            element_collection.child_count_array,
            distributed_projections,
            initial_vectors,
            total_dof_counts,
        )
        del distributed_projections
    else:
        initial_vectors = None
        initial_solution = None

    if time_settings is not None:
        time_carry_index_array = call_per_element_flex(
            element_collection.com,
            1,
            np.uint32,
            _find_time_carry_indices,
            tuple(
                sorted(
                    system.weight_forms.index(form)
                    for form in time_settings.time_march_relations
                )
            ),
            dof_offsets,
            element_collection.child_count_array,
        )
        if initial_vectors and initial_solution:
            # compute carry
            old_solution_carry = extract_carry(
                element_collection, time_carry_index_array, initial_vectors
            )
            solution = initial_solution
        else:
            old_solution_carry = FlexibleElementArray(
                element_collection.com, np.float64, time_carry_index_array.shapes
            )
    else:
        time_carry_index_array = None
        old_solution_carry = None

    del initial_solution, initial_vectors

    assert compiled_system.linear_codes

    # Compute vector fields at integration points for leaf elements
    vec_field_offsets, vec_fields = compute_vector_fields_nonlin(
        system,
        leaf_elements,
        cache,
        vector_fields,
        element_collection.corners_array,
        element_collection.orders_array,
        dof_offsets,
        solution,
    )

    element_matrices = compute_element_matrices(
        [f.order for f in system.unknown_forms],
        compiled_system.linear_codes,
        bl,
        br,
        tr,
        tl,
        orde,
        vec_fields,
        vec_field_offsets,
        c_ser,
    )
    linear_element_matrices = assign_leaves(
        element_collection, 2, np.float64, element_matrices
    )

    main_mat = assemble_matrix(
        unknown_ordering,
        element_collection,
        dof_offsets,
        linear_element_matrices,
    )
    main_vec = assemble_vector(
        unknown_ordering,
        element_collection,
        dof_offsets,
        lagrange_counts,
        linear_vectors,
    )

    def _find_constrained_indices(
        ie: int,
        i_unknown: int,
        child_count: FixedElementArray[np.uint32],
        dof_offsets: FixedElementArray[np.uint32],
    ) -> npt.NDArray[np.uint32]:
        """Find indices of DoFs that should be constrained for an element."""
        if int(child_count[ie][0]) != 0:
            return np.zeros(0, np.uint32)
        offsets = dof_offsets[ie]
        return np.arange(offsets[i_unknown], offsets[i_unknown + 1], dtype=np.uint32)

    # Generate constraints that force the specified for to have the (child element) sum
    # equal to a prescribed value.
    constrained_form_constaints = {
        form: Constraint(
            k,
            *(
                ElementConstraint(ie, dofs, np.ones_like(dofs, dtype=np.float64))
                for ie, dofs in enumerate(
                    call_per_element_flex(
                        element_collection.com,
                        1,
                        np.uint32,
                        _find_constrained_indices,
                        system.unknown_forms.index(form),
                        element_collection.child_count_array,
                        dof_offsets,
                    )
                )
            ),
        )
        for k, form in constrained_forms
    }

    if boundary_conditions is None:
        boundary_conditions = list()

    top_indices = np.astype(
        np.flatnonzero(np.array(element_collection.parent_array) == 0),
        np.uint32,
        copy=False,
    )

    strong_bc_constraints, weak_bc_constraints = mesh_boundary_conditions(
        [eq.right for eq in system.equations],
        mesh,
        unknown_ordering,
        element_collection,
        dof_offsets,
        top_indices,
        [
            [bc for bc in boundary_conditions if bc.form == eq.weight.base_form]
            for eq in system.equations
        ],
        cache,
    )

    continuity_constraints = mesh_continuity_constraints(
        system,
        mesh,
        top_indices,
        unknown_ordering,
        element_collection,
        unique_orders,
        dof_offsets,
    )

    element_offset = np.astype(
        np.pad(np.array(total_dof_counts, np.uint32).flatten().cumsum(), (1, 0)),
        np.uint32,
        copy=False,
    )

    constraint_rows: list[npt.NDArray[np.uint32]] = list()
    constraint_cols: list[npt.NDArray[np.uint32]] = list()
    constraint_coef: list[npt.NDArray[np.float64]] = list()
    constraint_vals: list[float] = list()
    # Continuity constraints
    ic = 0
    for constraint in continuity_constraints:
        constraint_vals.append(constraint.rhs)
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        ic += 1

    # Form constraining
    for form in constrained_form_constaints:
        constraint = constrained_form_constaints[form]
        constraint_vals.append(constraint.rhs)
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        ic += 1

    # Strong BC constraints
    for ec in strong_bc_constraints:
        offset = int(element_offset[ec.i_e])
        for ci, cv in zip(ec.dofs, ec.coeffs, strict=True):
            constraint_rows.append(np.array([ic]))
            constraint_cols.append(np.array([ci + offset]))
            constraint_coef.append(np.array([1.0]))
            constraint_vals.append(float(cv))

            ic += 1

    # Weak BC constraints/additions
    for ec in weak_bc_constraints:
        offset = element_offset[ec.i_e]
        main_vec[ec.dofs] += ec.coeffs
    if constraint_coef:
        lagrange_mat = sp.csr_array(
            (
                np.concatenate(constraint_coef),
                (np.concatenate(constraint_rows), np.concatenate(constraint_cols)),
            )
        )
        lagrange_mat.resize((ic, element_offset[-1]))
        main_mat = cast(
            sp.csr_array,
            sp.block_array(
                ((main_mat, lagrange_mat.T), (lagrange_mat, None)), format="csr"
            ),
        )
        lagrange_vec = np.array(constraint_vals, np.float64)
        main_vec = np.concatenate((main_vec, lagrange_vec))
    else:
        lagrange_mat = None
        lagrange_vec = np.zeros(0, np.float64)

    # # TODO: Delet dis
    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(main_mat.toarray())
    # # plt.spy(main_mat)
    # plt.show()

    linear_matrix = sp.csc_array(main_mat)
    explicit_vec = main_vec

    if time_settings is not None:
        assert time_carry_index_array is not None
        time_carry_term = extract_carry(
            element_collection, time_carry_index_array, linear_vectors
        )
    else:
        time_carry_term = None
    del main_mat, main_vec

    system_decomp = sla.splu(linear_matrix)

    resulting_grids: list[pv.UnstructuredGrid] = list()

    grid = reconstruct_mesh_from_solution(
        system,
        recon_order,
        element_collection,
        cache,
        dof_offsets,
        solution,
    )
    grid.field_data["time"] = (0.0,)
    resulting_grids.append(grid)

    global_lagrange = np.zeros_like(lagrange_vec)
    max_mag = np.abs(explicit_vec).max()

    max_iterations = solver_settings.maximum_iterations
    relax = solver_settings.relaxation
    atol = solver_settings.absolute_tolerance
    rtol = solver_settings.relative_tolerance

    if time_settings is not None:
        nt = time_settings.nt
        dt = time_settings.dt
        changes = np.zeros(nt, np.float64)
        iters = np.zeros(nt, np.uint32)

        for time_index in range(nt):
            max_residual = np.inf
            # 2 / dt * old_solution_carry + time_carry_term
            current_carry = call_per_element_flex(
                element_collection.com,
                1,
                np.float64,
                lambda ie, x, y: 2 / dt * x[ie] + y[ie],
                old_solution_carry,
                time_carry_term,
            )
            new_solution, global_lagrange, iter_cnt, max_residual = non_linear_solve_run(
                system,
                max_iterations,
                relax,
                atol,
                rtol,
                print_residual,
                unknown_ordering,
                element_collection,
                leaf_elements,
                cache,
                compiled_system,
                explicit_vec,
                bl,
                br,
                tr,
                tl,
                orde,
                c_ser,
                dof_offsets,
                element_offset,
                linear_element_matrices,
                time_carry_index_array,
                current_carry,
                solution,
                global_lagrange,
                max_mag,
                vector_fields,
                system_decomp,
                lagrange_mat,
            )

            changes[time_index] = float(max_residual)
            iters[time_index] = iter_cnt
            updated_derivative = assign_leaves(
                element_collection,
                1,
                np.float64,
                compute_element_explicit(
                    np.concatenate(new_solution, dtype=np.float64),
                    element_offset[leaf_elements],
                    [f.order for f in system.unknown_forms],
                    projection_codes,
                    bl,
                    br,
                    tr,
                    tl,
                    orde,
                    vec_fields,
                    vec_field_offsets,
                    c_ser,
                ),
            )
            assert time_carry_index_array is not None
            new_solution_carry = extract_carry(
                element_collection, time_carry_index_array, updated_derivative
            )
            # Compute time carry
            new_time_carry_term = call_per_element_flex(
                element_collection.com,
                1,
                np.float64,
                lambda ie, x, y, z: 2 / dt * (x[ie] - y[ie]) - z[ie],
                new_solution_carry,
                old_solution_carry,
                time_carry_term,
            )
            # 2 / dt * (new_solution_carry - old_solution_carry) - time_carry_term

            solution = new_solution
            time_carry_term = new_time_carry_term
            old_solution_carry = new_solution_carry
            del new_solution_carry, new_time_carry_term, new_solution, updated_derivative

            if (time_index % time_settings.sample_rate) == 0 or time_index + 1 == nt:
                # Prepare to build up the 1D Splines

                grid = reconstruct_mesh_from_solution(
                    system,
                    recon_order,
                    element_collection,
                    cache,
                    dof_offsets,
                    solution,
                )
                grid.field_data["time"] = (float((time_index + 1) * dt),)
                resulting_grids.append(grid)

            if print_residual:
                print(
                    f"Time step {time_index:d} finished in {iter_cnt:d} iterations with"
                    f" residual of {max_residual:.5e}"
                )
    else:
        new_solution, global_lagrange, iter_cnt, changes = non_linear_solve_run(
            system,
            max_iterations,
            relax,
            atol,
            rtol,
            print_residual,
            unknown_ordering,
            element_collection,
            leaf_elements,
            cache,
            compiled_system,
            explicit_vec,
            bl,
            br,
            tr,
            tl,
            orde,
            c_ser,
            dof_offsets,
            element_offset,
            linear_element_matrices,
            None,
            None,
            solution,
            global_lagrange,
            max_mag,
            vector_fields,
            system_decomp,
            lagrange_mat,
        )
        iters = np.array((iter_cnt,), np.uint32)  # type: ignore

        solution = new_solution
        del new_solution

        # Prepare to build up the 1D Splines

        grid = reconstruct_mesh_from_solution(
            system,
            recon_order,
            element_collection,
            cache,
            dof_offsets,
            solution,
        )

        resulting_grids.append(grid)

    del c_ser, bl, br, tr, tl, orde
    # TODO: solution statistics
    orders, counts = np.unique(
        np.array(element_collection.orders_array), return_counts=True
    )
    stats = SolutionStatisticsUnsteady(
        element_orders={int(order): int(count) for order, count in zip(orders, counts)},
        n_total_dofs=explicit_vec.size,
        n_lagrange=int(lagrange_vec.size + np.array(lagrange_counts).sum()),
        n_elems=element_collection.com.element_cnt,
        n_leaves=len(leaf_elements),
        n_leaf_dofs=sum(int(total_dof_counts[int(ie)][0]) for ie in leaf_elements),
        iter_history=iters,
        residual_history=changes,
    )

    return tuple(resulting_grids), stats
