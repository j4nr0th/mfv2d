"""Implementation of VMS computations."""

from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import Basis1D, Basis2D, IntegrationRule1D
from mfv2d.element import (
    ElementCollection,
    FixedElementArray,
    FlexibleElementArray,
    call_per_element_flex,
    element_projections,
)
from mfv2d.eval import _CompiledCodeMatrix
from mfv2d.kform import KFormSystem, KFormUnknown, UnknownOrderings, VectorFieldFunction
from mfv2d.mimetic2d import BasisCache
from mfv2d.solve_system import (
    assemble_matrix,
    compute_leaf_element_matrices,
    compute_vector_fields_nonlin,
)


def compute_mass_matrix(
    elements: ElementCollection,
    unknown_orders: UnknownOrderings,
    proj_codes: _CompiledCodeMatrix,
    bl: npt.NDArray[np.float64],
    br: npt.NDArray[np.float64],
    tr: npt.NDArray[np.float64],
    tl: npt.NDArray[np.float64],
    orders: npt.NDArray[np.uint32],
    caches: MutableMapping[int, BasisCache],
    element_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    inverse: bool,
) -> sp.csr_array:
    """Compute mass matrix for the elements."""
    unique_orders = np.unique(orders)
    c_ser = list()
    for o in unique_orders:
        if int(o) in caches:
            c = caches[int(o)]
        else:
            c = BasisCache(int(o), int(o) + 2)
            caches[int(o)] = c
        c_ser.append(c.c_serialization())

    leaf_mass_matrices = compute_leaf_element_matrices(
        unknown_orders,
        elements,
        proj_codes,
        bl,
        br,
        tr,
        tl,
        orders,
        c_ser,
        np.zeros((orders.size + 1), np.uint64),
        tuple(),
    )

    vals: list[npt.NDArray[np.floating]] = list()
    cols: list[npt.NDArray[np.integer]] = list()
    rows: list[npt.NDArray[np.integer]] = list()
    for ie in leaf_indices:
        off = int(element_offsets[int(ie)][0])
        m = leaf_mass_matrices[int(ie)]
        n = m.shape[0]
        assert m.shape[1] == n
        if inverse:
            mat = np.linalg.inv(m)
        else:
            mat = m
        vals.append(mat.flatten())
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
    corner_array: FixedElementArray[np.float64],
    bl: npt.NDArray[np.float64],
    br: npt.NDArray[np.float64],
    tr: npt.NDArray[np.float64],
    tl: npt.NDArray[np.float64],
    fine_orders: FixedElementArray[np.uint32],
    coarse_orders: FixedElementArray[np.uint32],
    caches: MutableMapping[int, BasisCache],
    element_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    coarse_solution: FlexibleElementArray[np.float64, np.uint32],
    vector_fields: Sequence[VectorFieldFunction | KFormUnknown],
    dof_offsets: FixedElementArray[np.uint32],
) -> sp.csr_array:
    """Compute mass matrix for the elements."""
    unique_orders = np.unique(np.array(coarse_orders))
    c_ser = list()
    for o in unique_orders:
        if int(o) in caches:
            c = caches[int(o)]
        else:
            c = BasisCache(int(o), int(o) + 2)
            caches[int(o)] = c
        c_ser.append(c.c_serialization())

    vec_offsets, vec_values = compute_vector_fields_nonlin(
        system,
        leaf_indices,
        caches,
        vector_fields,
        corner_array,
        coarse_orders,
        fine_orders,
        dof_offsets,
        coarse_solution,
    )

    leaf_matrices = compute_leaf_element_matrices(
        unknown_orders,
        elements,
        adv_codes,
        bl,
        br,
        tr,
        tl,
        np.array(fine_orders, np.uint32),
        c_ser,
        vec_offsets,
        vec_values,
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
    child_count: FixedElementArray[np.uint32],
    corners: FixedElementArray[np.float64],
    basis_cache: Mapping[int, Basis2D],
    orders_corase: FixedElementArray[np.uint32],
    orders_fine: FixedElementArray[np.uint32],
    unknowns: UnknownOrderings,
) -> npt.NDArray[np.float64]:
    """Compute projection if the element is actually a leaf."""
    if int(child_count[ie]) != 0:
        return np.zeros(0, np.float64)

    blocks = element_projections(
        unknowns,
        corners[ie],
        basis_cache[int(orders_corase[ie][0])],
        basis_cache[int(orders_fine[ie][0])],
    )
    return np.astype(sp.block_diag(blocks).toarray(), np.float64, copy=False)


def compute_projection_matrix(
    unknowns: UnknownOrderings,
    elements: ElementCollection,
    basis_cache: MutableMapping[int, Basis2D],
    orders_corase: FixedElementArray[np.uint32],
    orders_fine: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    offsets_coarse: FixedElementArray[np.uint32],
    offsets_fine: FixedElementArray[np.uint32],
) -> sp.csr_array:
    """Compute the global projection matrix."""
    projection_matrices = call_per_element_flex(
        elements.com,
        2,
        np.float64,
        _projection_if_leaf,
        elements.child_count_array,
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
    proj_codes: _CompiledCodeMatrix,
    bl: npt.NDArray[np.float64],
    br: npt.NDArray[np.float64],
    tr: npt.NDArray[np.float64],
    tl: npt.NDArray[np.float64],
    coarse_orders: FixedElementArray[np.uint32],
    fine_orders: FixedElementArray[np.uint32],
    caches: MutableMapping[int, BasisCache],
    coarse_offsets: FixedElementArray[np.uint32],
    fine_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
):
    """Compute Green's fine-scale function operator."""
    coarse_order_array = np.array(coarse_orders, np.uint32)
    fine_order_array = np.array(fine_orders, np.uint32)
    c_ser = list()
    cache2d_basis: dict[int, Basis2D] = dict()
    for o in (
        int(v)
        for v in np.unique((np.unique(coarse_order_array), np.unique(fine_order_array)))
    ):
        if o in caches:
            c = caches[o]
        else:
            c = BasisCache(o, o + 2)
            caches[o] = c
        c_ser.append(c.c_serialization())
        irule = IntegrationRule1D(o + 2)
        cache2d_basis[o] = Basis2D(Basis1D(o, irule), Basis1D(o, irule))

    fine_element_matrices = compute_leaf_element_matrices(
        unknown_orders,
        elements,
        system_codes,
        bl,
        br,
        tr,
        tl,
        fine_order_array,
        c_ser,
        np.zeros(elements.com.element_cnt + 1, np.uint64),
        tuple(),
    )

    fine_sys = assemble_matrix(
        unknown_orders,
        elements,
        fine_offsets,
        fine_element_matrices,
    )

    coarse_element_matrices = compute_leaf_element_matrices(
        unknown_orders,
        elements,
        system_codes,
        bl,
        br,
        tr,
        tl,
        coarse_order_array,
        c_ser,
        np.zeros(elements.com.element_cnt + 1, np.uint64),
        tuple(),
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
        proj_codes,
        bl,
        br,
        tr,
        tl,
        fine_order_array,
        caches,
        fine_offsets,
        leaf_indices,
        True,
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
    proj_codes: _CompiledCodeMatrix,
    bl: npt.NDArray[np.float64],
    br: npt.NDArray[np.float64],
    tr: npt.NDArray[np.float64],
    tl: npt.NDArray[np.float64],
    coarse_orders: FixedElementArray[np.uint32],
    fine_orders: FixedElementArray[np.uint32],
    caches: MutableMapping[int, BasisCache],
    coarse_offsets: FixedElementArray[np.uint32],
    fine_offsets: FixedElementArray[np.uint32],
    leaf_indices: npt.NDArray[np.uint32],
    coarse_solution: FlexibleElementArray[np.float64, np.uint32],
    vector_fields: Sequence[VectorFieldFunction | KFormUnknown],
    dof_offsets: FixedElementArray[np.uint32],
):
    """Compute SG operator."""
    fine_scale_greens = greens_fine_scales(
        elements,
        unknown_orders,
        system_codes,
        proj_codes,
        bl,
        br,
        tr,
        tl,
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
        elements.corners_array,
        bl,
        br,
        tr,
        tl,
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
