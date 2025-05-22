"""Implementation of VMS computations."""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib._mimetic import compute_element_matrices
from interplib.element_tree import ElementTree
from interplib.kforms import KFormSystem
from interplib.kforms.eval import MatOpCode
from interplib.mimetic.mimetic2d import (
    BasisCache,
    Element2D,
    ElementLeaf2D,
    ElementNode2D,
)
from interplib.system2d import assemble_global_element_matrix, element_rhs


def compute_element_leaf_mass_matrix(
    unknown_form_orders: Sequence[int],
    element: ElementLeaf2D,
    cache: BasisCache,
    inverse: bool,
) -> sp.csr_array:
    """Compute the mass matrix for the element.

    Parameters
    ----------
    unknown_form_orders : Sequence of int
        Orders of unknown forms defined on the mesh.

    element : ElementLeaf2D
        Leaf element for which the mass matrix should be computed.

    cache : BasisCache
        Cache containing basis needed to compute the mass matrix.

    inverse : bool
        Should the inverse of the mass matrix be computed instead.

    Returns
    -------
    csr_array
        Element mass matrix, which takes all primal degrees of freedom and returns duals.
    """
    mass_matrices: dict[int, npt.NDArray[np.floating]] = dict()
    diag_entries: list[npt.NDArray[np.floating]] = list()

    mat: npt.NDArray[np.floating]
    for form in unknown_form_orders:
        if form in mass_matrices:
            diag_entries.append(mass_matrices[form])
            continue

        if form == 0:
            mat = element.mass_matrix_node(cache)
        elif form == 1:
            mat = element.mass_matrix_edge(cache)
        elif form == 2:
            mat = element.mass_matrix_surface(cache)
        else:
            raise ValueError(f"Unknown form order was specified as {form}.")

        if inverse:
            mat = np.linalg.inv(mat)

        diag_entries.append(mat)
        mass_matrices[form] = mat

    return cast(sp.csr_array, sp.block_diag(diag_entries, format="csr"))


def compute_element_mass_matrix(
    unknown_form_orders: Sequence[int],
    element: Element2D,
    caches: Mapping[int, BasisCache],
    inverse: bool,
) -> sp.csr_array:
    """Compute the mass matrix for the element.

    Parameters
    ----------
    unknown_form_orders : Sequence of int
        Orders of unknown forms defined on the mesh.

    element : Element2D
        Element for which the mass matrix should be computed.

    caches : Mapping of int -> BasisCache
        Mapping with caches containing basis needed to compute the mass matrix.

    inverse : bool
        Should the inverse of the mass matrix be computed instead.

    Returns
    -------
    csr_array
        Element mass matrix, which takes all primal degrees of freedom and returns duals.
        Lagrange multipliers are kept the same.
    """
    if type(element) is ElementLeaf2D:
        return compute_element_leaf_mass_matrix(
            unknown_form_orders, element, caches[element.order], inverse
        )
    assert type(element) is ElementNode2D

    m1, m2, m3, m4 = (
        compute_element_mass_matrix(unknown_form_orders, e, caches, inverse)
        for e in element.children()
    )

    self_cnt = element.total_dof_count(unknown_form_orders, False, False)
    lagrange_cnt = element.total_dof_count(unknown_form_orders, True, False) - self_cnt

    return cast(
        sp.csr_array,
        sp.block_diag(
            (sp.eye(self_cnt), m1, m2, m3, m4, sp.eye(lagrange_cnt)), format="csr"
        ),
    )


def compute_element_leaf_finer_projection(
    unknown_form_orders: Sequence[int], element: ElementLeaf2D, cache: BasisCache
) -> sp.csr_array:
    """Create projection to the element from the approximation of the continuous space.

    Parameters
    ----------
    unknown_form_orders : Sequence of int
        Orders of unknown forms defined on the mesh.

    element : ElementLeaf2D
        Leaf element for which the matrix should be computed.

    cache : BasisCache
        Cache containing basis needed to compute the mass matrix.

    Returns
    -------
    csr_array
        Element mapping matrix, which takes all primal degrees of freedom on the fine
        space and returns primal degrees of freedom on the dual mesh. If transposed it
        becomes a transformation from the dual degrees of freedom on the coarse mesh
        to dual degrees of freedom on the fine mesh.
    """
    blocks: list[npt.NDArray[np.float64]] = list()
    already_computed: dict[int, npt.NDArray[np.float64]] = dict()

    for form in unknown_form_orders:
        if form in already_computed:
            blocks.append(already_computed[form])

        (j00, j01), (j10, j11) = element.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01
        if form == 0:
            precomp = cache.mass_node_precomp_higher

            mat = np.sum(precomp * det[None, None, ...], axis=(-2, -1))
            pro = element.mass_matrix_node(cache)

        elif form == 1:
            precomp = cache.mass_edge_precomp_higher
            khh = j11**2 + j10**2
            kvv = j01**2 + j00**2
            kvh = j01 * j11 + j00 * j10
            khh = khh / det
            kvv = kvv / det
            kvh = kvh / det

            nb1 = element.order * (element.order + 1)
            nb2 = (cache.higher_order) * (cache.higher_order + 1)
            mat = np.empty((2 * nb1, 2 * nb2), np.float64)

            mat[0 * nb1 : 1 * nb1, 0 * nb2 : 1 * nb2] = np.sum(
                precomp[0 * nb1 : 1 * nb1, 0 * nb2 : 1 * nb2, ...] * khh[None, None, ...],
                axis=(-2, -1),
            )
            mat[1 * nb1 : 2 * nb1, 0 * nb2 : 1 * nb2] = np.sum(
                precomp[1 * nb1 : 2 * nb1, 0 * nb2 : 1 * nb2, ...] * kvh[None, None, ...],
                axis=(-2, -1),
            )
            mat[0 * nb1 : 1 * nb1, 1 * nb2 : 2 * nb2] = np.sum(
                precomp[0 * nb1 : 1 * nb1, 1 * nb2 : 2 * nb2, ...] * kvh[None, None, ...],
                axis=(-2, -1),
            )
            mat[1 * nb1 : 2 * nb1, 1 * nb2 : 2 * nb2] = np.sum(
                precomp[1 * nb1 : 2 * nb1, 1 * nb2 : 2 * nb2, ...] * kvv[None, None, ...],
                axis=(-2, -1),
            )
            pro = element.mass_matrix_edge(cache)
        elif form == 2:
            precomp = cache.mass_surf_precomp_higher

            mat = np.sum(precomp / det[None, None, ...], axis=(-2, -1))
            pro = element.mass_matrix_surface(cache)
        else:
            raise ValueError(f"Invalid form order given {form}.")

        block = np.astype(np.linalg.solve(pro, mat), np.float64, copy=False)
        already_computed[form] = block
        blocks.append(block)

    return cast(sp.csr_array, sp.block_diag(blocks, format="crs"))


def compute_element_finer_projection(
    unknown_form_orders: Sequence[int],
    coarse_element: Element2D,
    fine_element: Element2D,
    caches: Mapping[int, BasisCache],
) -> sp.csr_array:
    """Create projection to the element from the approximation of the continuous space.

    Parameters
    ----------
    unknown_form_orders : Sequence of int
        Orders of unknown forms defined on the mesh.

    element : ElementLeaf2D
        Leaf element for which the matrix should be computed.

    cache : BasisCache
        Cache containing basis needed to compute the mass matrix.

    Returns
    -------
    csr_array
        Element mapping matrix, which takes all primal degrees of freedom on the fine
        space and returns primal degrees of freedom on the dual mesh. If transposed it
        becomes a transformation from the dual degrees of freedom on the coarse mesh
        to dual degrees of freedom on the fine mesh.
    """
    if type(coarse_element) is ElementLeaf2D:
        return compute_element_leaf_finer_projection(
            unknown_form_orders, coarse_element, caches[coarse_element.order]
        )
    assert type(coarse_element) is ElementNode2D and type(fine_element) is ElementNode2D

    m1, m2, m3, m4 = (
        compute_element_finer_projection(unknown_form_orders, ec, ef, caches)
        for ec, ef in zip(coarse_element.children(), fine_element.children())
    )

    self_cnt_coarse = coarse_element.total_dof_count(unknown_form_orders, False, False)
    lagrange_cnt_coarse = (
        coarse_element.total_dof_count(unknown_form_orders, True, False) - self_cnt_coarse
    )

    self_cnt_fine = fine_element.total_dof_count(unknown_form_orders, False, False)
    lagrange_cnt_fine = (
        fine_element.total_dof_count(unknown_form_orders, True, False) - self_cnt_fine
    )

    return cast(
        sp.csr_array,
        sp.block_diag(
            (
                sp.csr_array((self_cnt_coarse, self_cnt_fine)),
                m1,
                m2,
                m3,
                m4,
                sp.csr_array((lagrange_cnt_coarse, lagrange_cnt_fine)),
            ),
            format="csr",
        ),
    )


def greens_fine_scales(
    unknown_form_orders: Sequence[int],
    coarse_tree: ElementTree,
    fine_tree: ElementTree,
    caches: Mapping[int, BasisCache],
    n_lagrange: int,
    symmetric_code: Sequence[Sequence[None | Sequence[MatOpCode | int | float]]],
    full_system: KFormSystem,
):
    """Compute Green's fine-scale function operator."""
    fine_element_mass_matrices = [
        compute_element_mass_matrix(unknown_form_orders, element, caches, inverse=True)
        for element in fine_tree.iter_top()
    ]
    fine_to_coarse_projections = [
        compute_element_finer_projection(
            unknown_form_orders, element_coarse, element_fine, caches
        )
        for element_coarse, element_fine in zip(
            coarse_tree.iter_top(), fine_tree.iter_top(), strict=True
        )
    ]
    matrix = cast(sp.csr_array, sp.block_diag(fine_element_mass_matrices, format="csr"))
    projection = cast(
        sp.csr_array, sp.block_diag(fine_to_coarse_projections, format="csr")
    )
    matrix.resize((matrix.shape[0] + n_lagrange, matrix.shape[1] + n_lagrange))

    c_ser = tuple(caches[cache_order].c_serialization() for cache_order in caches)

    matrices_fine = {
        int(i): m
        for i, m in zip(
            fine_tree.leaf_indices(),
            compute_element_matrices(
                unknown_form_orders,
                symmetric_code,
                np.asarray([e.bottom_left for e in fine_tree.iter_leaves()], np.float64),
                np.asarray([e.bottom_right for e in fine_tree.iter_leaves()], np.float64),
                np.asarray([e.top_right for e in fine_tree.iter_leaves()], np.float64),
                np.asarray([e.top_left for e in fine_tree.iter_leaves()], np.float64),
                np.asarray([e.order for e in fine_tree.iter_leaves()], np.uint32),
                (),
                np.array([], np.uint64),
                c_ser,
            ),
            strict=True,
        )
    }
    vectors_fine = {
        int(i): element_rhs(full_system, e, caches[e.order])
        for i, e in zip(fine_tree.leaf_indices(), fine_tree.iter_leaves(), strict=True)
    }
    fine_matrix, fine_vector = assemble_global_element_matrix(
        full_system, unknown_form_orders, fine_tree, matrices_fine, vectors_fine
    )

    matrices_coarse = {
        int(i): m
        for i, m in zip(
            coarse_tree.leaf_indices(),
            compute_element_matrices(
                unknown_form_orders,
                symmetric_code,
                np.asarray(
                    [e.bottom_left for e in coarse_tree.iter_leaves()], np.float64
                ),
                np.asarray(
                    [e.bottom_right for e in coarse_tree.iter_leaves()], np.float64
                ),
                np.asarray([e.top_right for e in coarse_tree.iter_leaves()], np.float64),
                np.asarray([e.top_left for e in coarse_tree.iter_leaves()], np.float64),
                np.asarray([e.order for e in coarse_tree.iter_leaves()], np.uint32),
                (),
                np.array([], np.uint64),
                c_ser,
            ),
            strict=True,
        )
    }
    vectors_coarse = {
        int(i): element_rhs(full_system, e, caches[e.order])
        for i, e in zip(
            coarse_tree.leaf_indices(), coarse_tree.iter_leaves(), strict=True
        )
    }
    coarse_matrix, coarse_vector = assemble_global_element_matrix(
        full_system, unknown_form_orders, coarse_tree, matrices_coarse, vectors_coarse
    )
    projection.resize((coarse_matrix.shape[0], fine_matrix.shape[1]))

    mat_fine = sla.spsolve(fine_matrix, matrix)
    mat_coarse = projection @ matrix
    mat_coarse = sla.spsolve(coarse_matrix, matrix)
    mat_coarse = projection.T @ matrix
    return mat_fine - mat_coarse
