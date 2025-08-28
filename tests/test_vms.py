"""Tests related to VMS."""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    ElementFemSpace2D,
    compute_element_matrix,
    compute_element_projector,
)
from mfv2d.continuity import add_system_constraints
from mfv2d.eval import CompiledSystem
from mfv2d.examples import unit_square_mesh
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import FemCache
from mfv2d.refinement import _fine_scale_greens_function
from mfv2d.system import KFormSystem
from scipy import sparse as sp
from scipy.sparse import linalg as sla


@pytest.mark.parametrize(
    ("nh", "nv", "element_order", "k"), ((5, 6, 3, 2), (3, 5, 4, 3), (2, 2, 1, 1))
)
def test_fine_green_adv_dif(nh: int, nv: int, element_order: int, k: int) -> None:
    """Check fine-scale Green's function works as one would expect.

    Fine-scale Green's operator should be such, that when the differential
    operator it is based is used as the projector to the coarse mesh, it
    should be zero.
    """
    mesh = unit_square_mesh(
        nh,
        nv,
        element_order,
        deformation=lambda xi, eta: (
            xi + 0.1 * np.sin(np.pi * xi) * np.sin(np.pi * eta),
            eta - 0.1 * np.sin(np.pi * xi) * np.sin(np.pi * eta),
        ),
    )

    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
    q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)

    v = u.weight
    p = q.weight

    system = KFormSystem(
        p * q + p.derivative * u == 0,
        v * q.derivative == 0,
        sorting=lambda f: f.order,
    )

    compiled = CompiledSystem(system)

    coarse_spaces: list[ElementFemSpace2D] = list()
    fine_spaces: list[ElementFemSpace2D] = list()

    coarse_matrices: list[npt.NDArray[np.float64]] = list()
    fine_matrices: list[npt.NDArray[np.float64]] = list()
    projection_matrices: list[sp.coo_array] = list()

    basis_cache = FemCache(order_difference=k)

    leaf_indices = tuple(int(idx) for idx in mesh.get_leaf_indices())
    for idx in leaf_indices:
        corners = np.astype(mesh.get_leaf_corners(idx), np.float64, copy=False)
        coarse_spaces.append(
            ElementFemSpace2D(
                basis_cache.get_basis2d(
                    order1=element_order,
                    order2=element_order,
                    int_order1=element_order + k,
                    int_order2=element_order + k,
                ),
                corners,
            )
        )
        fine_spaces.append(
            ElementFemSpace2D(
                basis_cache.get_basis2d(
                    order1=element_order + k,
                    order2=element_order + k,
                    int_order1=element_order + k,
                    int_order2=element_order + k,
                ),
                corners,
            )
        )

        coarse_matrices.append(
            np.astype(
                compute_element_matrix(
                    system.unknown_forms, compiled.lhs_full, coarse_spaces[-1]
                ),
                np.float64,
                copy=False,
            )
        )

        fine_matrices.append(
            np.astype(
                compute_element_matrix(
                    system.unknown_forms, compiled.lhs_full, fine_spaces[-1]
                ),
                np.float64,
                copy=False,
            )
        )

        projection_matrices.append(
            cast(
                sp.coo_array,
                sp.block_diag(
                    compute_element_projector(
                        system.unknown_forms,
                        corners,
                        coarse_spaces[-1].basis_2d,
                        fine_spaces[-1].basis_2d,
                    ),
                    format="coo",
                ),
            )
        )

    mesh.uniform_p_change(k, k)
    fine_offsets = np.cumsum(
        [
            0,
            *(
                system.unknown_forms.total_size(*mesh.get_leaf_orders(i_leaf))
                for i_leaf in leaf_indices
            ),
        ]
    )
    lag_mat_fine, lag_vec_fine = add_system_constraints(
        system, mesh, basis_cache, [], [], leaf_indices, fine_offsets, None
    )
    assert lag_mat_fine is not None
    mesh.uniform_p_change(-k, -k)

    fine_operator = sp.block_array(
        [
            [sp.block_diag(fine_matrices), lag_mat_fine.T],
            [lag_mat_fine, None],
        ],
        format="csc",
    )
    fine_decomp = sla.splu(fine_operator)
    fine_padding = lag_vec_fine.size
    del fine_matrices, lag_mat_fine, lag_vec_fine, fine_offsets

    coarse_offsets = np.cumsum(
        [
            0,
            *(
                system.unknown_forms.total_size(*mesh.get_leaf_orders(i_leaf))
                for i_leaf in leaf_indices
            ),
        ]
    )
    lag_mat_coarse, lag_vec_coarse = add_system_constraints(
        system, mesh, basis_cache, [], [], leaf_indices, coarse_offsets, None
    )
    assert lag_mat_coarse is not None

    coarse_operator = sp.block_array(
        [
            [sp.block_diag(coarse_matrices), lag_mat_coarse.T],
            [lag_mat_coarse, None],
        ],
        format="csc",
    )
    coarse_decomp = sla.splu(coarse_operator)
    coarse_padding = lag_vec_coarse.size
    del coarse_matrices, lag_mat_coarse, lag_vec_coarse, coarse_offsets

    rng = np.random.default_rng(seed=0)
    assert fine_operator.shape is not None
    forcing = rng.uniform(-1, +1, fine_operator.shape[0] - fine_padding)

    projector = cast(sp.csr_array, sp.block_diag(projection_matrices, format="csr"))
    del projection_matrices

    fine_result = _fine_scale_greens_function(
        projector, fine_decomp, coarse_decomp, forcing, fine_padding, coarse_padding
    )

    fine_forcing = (fine_operator @ np.pad(fine_result, (0, fine_padding)))[
        :-fine_padding
    ] @ projector
    res = coarse_decomp.solve(np.pad(fine_forcing, (0, coarse_padding)))[:-coarse_padding]
    assert pytest.approx(res) == 0


if __name__ == "__main__":
    for nh, nv, eo, k in ((5, 6, 3, 2), (3, 5, 4, 3), (2, 2, 1, 1)):
        test_fine_green_adv_dif(nh, nv, eo, k)
