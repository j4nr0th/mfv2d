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
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import (
    Basis2D,
    ElementMassMatrixCache,
    compute_element_matrix,
    compute_element_vector,
)
from mfv2d.boundary import (
    BoundaryCondition2DSteady,
)
from mfv2d.element import (
    ElementCollection,
    FixedElementArray,
    FlexibleElementArray,
    ObjectElementArray,
    call_per_element,
    call_per_element_fix,
    call_per_element_flex,
    call_per_leaf,
    call_per_leaf_flex,
    element_dual_dofs,
    poly_x,
    poly_y,
    reconstruct,
)
from mfv2d.eval import CompiledSystem, _CompiledCodeMatrix
from mfv2d.kform import (
    Function2D,
    KElementProjection,
    KExplicit,
    KFormSystem,
    KFormUnknown,
    KWeight,
    UnknownFormOrder,
    UnknownOrderings,
)
from mfv2d.mimetic2d import Element2D, ElementLeaf2D, FemCache, vtk_lagrange_ordering
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


def rhs_2d_element_projection(
    right: KElementProjection, element_cache: ElementMassMatrixCache
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
        basis = element_cache.basis_2d
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

    return element_dual_dofs(UnknownFormOrder(right.weight.order + 1), element_cache, fn)


def _extract_rhs_2d(
    proj: Sequence[tuple[float, KExplicit]],
    weight: KWeight,
    element_cache: ElementMassMatrixCache,
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
    basis = element_cache.basis_2d
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
        rhs = rhs_2d_element_projection(f, element_cache)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def compute_element_rhs(
    ie: int,
    system: KFormSystem,
    element_caches: ObjectElementArray[ElementMassMatrixCache],
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

    elem_cache = element_caches[ie]
    for equation in system.equations:
        vecs.append(
            _extract_rhs_2d(equation.right.explicit_terms, equation.weight, elem_cache)
        )

    return np.concatenate(vecs, dtype=np.float64)


def assemble_matrix(
    leaf_matrices: FlexibleElementArray[np.float64, np.uint32],
) -> sp.csr_array:
    """Assemble global matrix."""
    matrices: list[npt.NDArray[np.float64]] = [
        mat for mat in leaf_matrices if mat.size != 0
    ]
    return cast(sp.csr_array, sp.block_diag(matrices, format="csr"))


def assemble_vector(
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Assemble global vector."""
    vectors: list[npt.NDArray[np.float64]] = [
        vec for vec in leaf_vectors if vec.size != 0
    ]

    return np.concatenate(vectors, dtype=np.float64)


def assemble_forcing(
    leaf_vectors: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Assemble global vector."""
    vectors: list[npt.NDArray[np.float64]] = [
        vec for vec in leaf_vectors if vec.size != 0
    ]

    return np.concatenate(vectors, dtype=np.float64)


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
    element_caches: ObjectElementArray[ElementMassMatrixCache],
    fields: FixedElementArray[np.object_],
) -> npt.NDArray[np.float64]:
    """Compute the element matrix."""
    elem_cache = element_caches[ie]
    vec_fields = tuple(fields[ie])

    mat = compute_element_matrix(
        unknowns.form_orders,
        expressions,
        vec_fields,
        elem_cache,
    )

    return mat


def compute_leaf_vector(
    ie: int,
    expressions: _CompiledCodeMatrix,
    unknowns: UnknownOrderings,
    element_caches: ObjectElementArray[ElementMassMatrixCache],
    fields: FixedElementArray[np.object_],
    solution: FlexibleElementArray[np.float64, np.uint32],
) -> npt.NDArray[np.float64]:
    """Compute the element vector."""
    elem_cache = element_caches[ie]
    vec_fields = tuple(fields[ie])

    vec = compute_element_vector(
        unknowns.form_orders,
        expressions,
        vec_fields,
        elem_cache,
        solution[ie],
    )

    return vec


def update_leaf_vector(
    ie: int,
    expressions: _CompiledCodeMatrix,
    unknowns: UnknownOrderings,
    element_caches: ObjectElementArray[ElementMassMatrixCache],
    fields: FixedElementArray[np.object_],
    solution: FlexibleElementArray[np.float64, np.uint32],
    result_array: FlexibleElementArray[np.float64, np.uint32],
) -> None:
    """Compute the element vector."""
    elem_cache = element_caches[ie]
    vec_fields = tuple(fields[ie])

    vec = compute_element_vector(
        unknowns.form_orders,
        expressions,
        vec_fields,
        elem_cache,
        solution[ie],
    )
    result_array[ie] = vec


def compute_element_dual(
    ie: int,
    ordering: UnknownOrderings,
    functions: Sequence[
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike] | None
    ],
    orders: FixedElementArray[np.uint32],
    element_caches: ObjectElementArray[ElementMassMatrixCache],
) -> npt.NDArray[np.float64]:
    """Compute element L2 projection."""
    order_1, order_2 = orders[ie]
    vecs: list[npt.NDArray[np.float64]] = list()
    for order, func in zip(ordering.form_orders, functions, strict=True):
        if func is None:
            vecs.append(np.zeros(order.full_unknown_count(order_1, order_2), np.float64))
        else:
            vecs.append(
                np.asarray(element_dual_dofs(order, element_caches[ie], func), np.float64)
            )

    return np.concatenate(vecs)


def compute_element_primal(
    ie: int,
    ordering: UnknownOrderings,
    dual_dofs: FlexibleElementArray[np.float64, np.uint32],
    orders: FixedElementArray[np.uint32],
    element_caches: ObjectElementArray[ElementMassMatrixCache],
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    order_1, order_2 = orders[ie]
    dual = np.array(dual_dofs[ie], np.float64, copy=True)
    elem_cache = element_caches[ie]
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    for form in ordering.form_orders:
        cnt = form.full_unknown_count(order_1, order_2)
        v = dual[offset : offset + cnt]
        if form in mats:
            m = mats[form]
        else:
            m = elem_cache.mass_from_order(form, inverse=True)
            mats[form] = m

        dual[offset : offset + cnt] = m @ v

        offset += cnt

    return dual


def compute_element_primal_to_dual(
    ie: int,
    ordering: UnknownOrderings,
    primal_dofs: FlexibleElementArray[np.float64, np.uint32],
    orders: FixedElementArray[np.uint32],
    element_caches: ObjectElementArray[ElementMassMatrixCache],
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    primal = np.array(primal_dofs[ie], np.float64, copy=True)
    order_1, order_2 = orders[ie]
    elem_cache = element_caches[ie]
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    for form in ordering.form_orders:
        cnt = form.full_unknown_count(order_1, order_2)
        v = primal[offset : offset + cnt]
        if form in mats:
            m = mats[form]
        else:
            m = elem_cache.mass_from_order(form, inverse=False)
            mats[form] = m

        primal[offset : offset + cnt] = m @ v

        offset += cnt

    return primal


def compute_leaf_full_mass_matrix(
    ie: int,
    ordering: UnknownOrderings,
    element_caches: ObjectElementArray[ElementMassMatrixCache],
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    elem_cache = element_caches[ie]
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    diags: list[npt.NDArray[np.float64]] = list()
    for form in ordering.form_orders:
        if form in mats:
            m = mats[form]
        else:
            m = elem_cache.mass_from_order(form, inverse=False)
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
    element_caches: ObjectElementArray[ElementMassMatrixCache],
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
        for off, idx, vals in zip(
            element_offsets, time_carry_index_array, time_carry_term
        ):
            base_vec[idx + int(off)] += vals
    else:
        assert time_carry_index_array is None
    residuals = np.zeros(max_iterations, np.float64)
    max_residual = 0.0
    vec_fields_array = FixedElementArray(element_collection.com, 0, np.object_)
    equation_values = FlexibleElementArray(
        element_collection.com, np.float64, solution.shapes
    )
    explicit_values = FlexibleElementArray(
        element_collection.com, np.float64, solution.shapes
    )

    while iter_cnt < max_iterations:
        # Recompute vector fields
        # Compute vector fields at integration points for leaf elements

        if len(vector_fields):
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

        call_per_leaf(
            element_collection,
            update_leaf_vector,
            compiled_system.lhs_full,
            unknown_ordering,
            element_caches,
            vec_fields_array,
            solution,
            equation_values,
        )

        main_value = assemble_forcing(equation_values)

        if compiled_system.rhs_codes is not None:
            main_vec = np.array(base_vec, copy=True)  # Make a copy
            # Update RHS implicit terms

            call_per_leaf(
                element_collection,
                update_leaf_vector,
                compiled_system.rhs_codes,
                unknown_ordering,
                element_caches,
                vec_fields_array,
                solution,
                explicit_values,
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
                element_caches,
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

            main_mat = assemble_matrix(element_matrices)
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
            out: FlexibleElementArray[np.float64, np.uint32],
        ) -> None:
            """Compute solution given update value."""
            out[ie] = sol[ie] + up[ie]

        call_per_element(
            element_collection.com,
            _update_solution,
            solution,
            sol_updates,
            solution,
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
