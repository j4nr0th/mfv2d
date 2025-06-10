"""Implementation of VMS computations."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import compute_element_projector
from mfv2d.element import (
    ElementCollection,
    FixedElementArray,
    FlexibleElementArray,
    call_per_element_fix,
    call_per_leaf_flex,
)
from mfv2d.eval import _CompiledCodeMatrix
from mfv2d.kform import Function2D, KFormSystem, KFormUnknown, UnknownOrderings
from mfv2d.mimetic2d import FemCache
from mfv2d.solve_system import (
    assemble_matrix,
    compute_element_vector_fields,
    compute_leaf_full_mass_matrix,
    compute_leaf_matrix,
)


def compute_mass_matrix(
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    orders: FixedElementArray[np.uint32],
    caches: FemCache,
    element_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
) -> sp.csr_array:
    """Compute (inverse) mass matrix for the elements."""
    leaf_mass_matrices = call_per_leaf_flex(
        elements,
        2,
        np.float64,
        compute_leaf_full_mass_matrix,
        unknown_orders,
        orders,
        elements.corners_array,
        caches,
    )

    vals: list[npt.NDArray[np.floating]] = list()
    cols: list[npt.NDArray[np.integer]] = list()
    rows: list[npt.NDArray[np.integer]] = list()
    for ie in leaf_indices:
        off = int(element_offsets[int(ie)][0])
        m = leaf_mass_matrices[int(ie)]
        n = m.shape[0]
        assert m.shape[1] == n
        vals.append(m.flatten())
        idx = np.arange(n)
        rows.append(np.repeat(idx, n) + off)
        cols.append(np.tile(idx, n) + off)

    return sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))
    )


def compute_advection_matrix(
    system: KFormSystem,
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    adv_codes: _CompiledCodeMatrix,
    fine_orders: FixedElementArray[np.uint32],
    coarse_orders: FixedElementArray[np.uint32],
    caches: FemCache,
    element_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    coarse_solution: FlexibleElementArray[np.float64, np.uint32],
    vector_fields: Sequence[Function2D | KFormUnknown],
    dof_offsets: FixedElementArray[np.uint32],
) -> sp.csr_array:
    """Compute mass matrix for the elements."""
    vec = call_per_element_fix(
        elements.com,
        np.object_,
        1,
        compute_element_vector_fields,
        system,
        elements.child_count_array,
        coarse_orders,
        fine_orders,
        caches,
        vector_fields,
        elements.corners_array,
        dof_offsets,
        coarse_solution,
    )

    leaf_matrices = call_per_leaf_flex(
        elements,
        2,
        np.float64,
        compute_leaf_matrix,
        adv_codes,
        unknown_orders,
        fine_orders,
        caches,
        elements.corners_array,
        vec,
    )

    vals: list[npt.NDArray[np.floating]] = list()
    cols: list[npt.NDArray[np.integer]] = list()
    rows: list[npt.NDArray[np.integer]] = list()
    for ie in leaf_indices:
        off = int(element_offsets[int(ie)][0])
        m = leaf_matrices[int(ie)]
        n = m.shape[0]
        assert m.shape[1] == n
        vals.append(m.flatten())
        idx = np.arange(n)
        rows.append(np.repeat(idx, n) + off)
        cols.append(np.tile(idx, n) + off)

    return sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))
    )


def _projection_if_leaf(
    ie: int,
    corners: FixedElementArray[np.float64],
    basis_cache: FemCache,
    orders_corase: FixedElementArray[np.uint32],
    orders_fine: FixedElementArray[np.uint32],
    unknowns: UnknownOrderings,
) -> npt.NDArray[np.float64]:
    """Compute projection if the element is actually a leaf."""
    o_fine = orders_fine[ie]
    o_coarse = orders_corase[ie]
    basis_coarse = basis_cache.get_basis2d(int(o_coarse[0]), int(o_coarse[1]))
    basis_fine = basis_cache.get_basis2d(
        int(o_fine[0]),
        int(o_fine[1]),
        basis_coarse.basis_xi.rule.order,  # Have to make sure we match integration orders
        basis_coarse.basis_eta.rule.order,
    )
    blocks = compute_element_projector(
        unknowns.form_orders,
        corners[ie],
        basis_fine,
        basis_coarse,
    )
    return np.astype(sp.block_diag(blocks).toarray(), np.float64, copy=False)


def compute_projection_matrix(
    unknowns: UnknownOrderings,
    elements: ElementCollection,
    basis_cache: FemCache,
    orders_corase: FixedElementArray[np.uint32],
    orders_fine: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    offsets_coarse: FixedElementArray[np.uint32],
    offsets_fine: FixedElementArray[np.uint32],
) -> sp.csr_array:
    """Compute the global projection matrix."""
    projection_matrices = call_per_leaf_flex(
        elements,
        2,
        np.float64,
        _projection_if_leaf,
        elements.corners_array,
        basis_cache,
        orders_corase,
        orders_fine,
        unknowns,
    )

    vals: list[npt.NDArray[np.floating]] = list()
    cols: list[npt.NDArray[np.integer]] = list()
    rows: list[npt.NDArray[np.integer]] = list()
    for ie in leaf_indices:
        off_c = int(offsets_coarse[int(ie)][0])
        off_f = int(offsets_fine[int(ie)][0])
        m = projection_matrices[int(ie)]
        n = m.shape[0]
        assert m.shape[1] == n
        vals.append(m.flatten())
        idx = np.arange(n)
        rows.append(np.repeat(idx, n) + off_c)
        cols.append(np.tile(idx, n) + off_f)

    return sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))
    )


def greens_fine_scales(
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    system_codes: _CompiledCodeMatrix,
    coarse_orders: FixedElementArray[np.uint32],
    fine_orders: FixedElementArray[np.uint32],
    cache2d_basis: FemCache,
    coarse_offsets: FixedElementArray[np.uint32],
    fine_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
):
    """Compute Green's fine-scale function operator."""
    fine_element_matrices = call_per_leaf_flex(
        elements,
        2,
        np.float64,
        compute_leaf_matrix,
        system_codes,
        unknown_orders,
        fine_orders,
        cache2d_basis,
        elements.corners_array,
        call_per_element_fix(elements.com, np.object_, 1, lambda _: ()),
    )

    fine_sys = assemble_matrix(
        unknown_orders,
        elements,
        fine_offsets,
        fine_element_matrices,
    )

    coarse_element_matrices = call_per_leaf_flex(
        elements,
        2,
        np.float64,
        compute_leaf_matrix,
        system_codes,
        unknown_orders,
        coarse_orders,
        cache2d_basis,
        elements.corners_array,
        call_per_element_fix(elements.com, np.object_, 1, lambda _: ()),
    )

    coarse_sys = assemble_matrix(
        unknown_orders,
        elements,
        coarse_offsets,
        coarse_element_matrices,
    )

    fine_mass = compute_mass_matrix(
        elements,
        unknown_orders,
        fine_orders,
        cache2d_basis,
        fine_offsets,
        leaf_indices,
    )

    projection = compute_projection_matrix(
        unknown_orders,
        elements,
        cache2d_basis,
        coarse_orders,
        fine_orders,
        leaf_indices,
        coarse_orders,
        fine_orders,
    )

    mat_fine = sla.spsolve(fine_sys, fine_mass)
    mat_coarse = projection @ fine_mass
    mat_coarse = sla.spsolve(coarse_sys, mat_coarse)
    mat_coarse = projection.T @ mat_coarse
    return mat_fine - mat_coarse


def suyash_green_operator(
    system: KFormSystem,
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    system_codes: _CompiledCodeMatrix,
    adv_codes: _CompiledCodeMatrix,
    coarse_orders: FixedElementArray[np.uint32],
    fine_orders: FixedElementArray[np.uint32],
    caches: FemCache,
    coarse_offsets: FixedElementArray[np.uint32],
    fine_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    coarse_solution: FlexibleElementArray[np.float64, np.uint32],
    vector_fields: Sequence[Function2D | KFormUnknown],
    dof_offsets: FixedElementArray[np.uint32],
):
    """Compute SG operator."""
    fine_scale_greens = greens_fine_scales(
        elements,
        unknown_orders,
        system_codes,
        coarse_orders,
        fine_orders,
        caches,
        coarse_offsets,
        fine_offsets,
        leaf_indices,
    )

    advection = compute_advection_matrix(
        system,
        elements,
        unknown_orders,
        adv_codes,
        fine_orders,
        coarse_orders,
        caches,
        fine_offsets,
        leaf_indices,
        coarse_solution,
        vector_fields,
        dof_offsets,
    )

    return sla.spsolve((1 - advection @ fine_scale_greens), advection)
