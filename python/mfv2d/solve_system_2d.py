"""Solving the actual system."""

from __future__ import annotations

from collections.abc import Sequence

# from itertools import chain
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import (
    # Basis2D,
    ElementFemSpace2D,
    Mesh,
    # compute_element_matrix,
    # compute_element_projector,
)

# from mfv2d.boundary import BoundaryCondition2DSteady
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KEquation, KFormSystem, UnknownOrderings  # ,  KFormUnknown
from mfv2d.mimetic2d import FemCache, compute_leaf_dof_counts
from mfv2d.progress import HistogramFormat
from mfv2d.refinement import RefinementSettings, perform_mesh_refinement
from mfv2d.solve_system import (
    SolutionStatistics,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    # _add_system_constraints,
    _compute_linear_system,
    compute_element_dual,
    compute_element_primal,
    compute_element_primal_to_dual,
    # compute_element_rhs,
    find_time_carry_indices,
    non_linear_solve_run,
    reconstruct_mesh_from_solution,
)


def solve_system_2d(
    mesh: Mesh,
    system_settings: SystemSettings,
    solver_settings: SolverSettings = SolverSettings(
        maximum_iterations=100,
        relaxation=1,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    ),
    time_settings: TimeSettings | None = None,
    refinement_settings: RefinementSettings | None = None,
    *,
    recon_order: int | None = None,
    print_residual: bool = False,
) -> tuple[Sequence[pv.UnstructuredGrid], SolutionStatistics, Mesh]:
    """Solve the unsteady system on the specified mesh.

    Parameters
    ----------
    mesh : Mesh
        Mesh on which to solve the system on.

    system_settings : SystemSettings
        Settings specifying the system of equations and boundary conditions to solve for.

    solver_settings : SolverSettings, optional
        Settings specifying the behavior of the solver

    time_settings : TimeSettings or None, default: None
        When set to ``None``, the equations are solved without time dependence (steady
        state). Otherwise, it specifies which equations are time derivative related and
        time step count and size.

    refinement_settings : RefinementSettings or None, default: None
        Settings specifying refinement of the mesh. If ``None`` is given instead, then
        refinement is not performed after the solution and instead

    recon_order : int, optional
        When specified, all elements will be reconstructed using this polynomial order.
        Otherwise, they are reconstructed with their own order.

    print_residual : bool, default: False
        Print the maximum of the absolute value of the residual for each iteration of the
        solver.

    Returns
    -------
    grids : Sequence of pyvista.UnstructuredGrid
        Reconstructed solution as an unstructured grid of VTK's "lagrange quadrilateral"
        cells. This reconstruction is done on the nodal basis for all unknowns.
    stats : SolutionStatisticsNonLin
        Statistics about the solution. This can be used for convergence tests or timing.
    mesh : Mesh
        When refinement settings are specified, the mesh resulting from the given
        refinement is returned. If the settings are left as unspecified, then the original
        mesh is returned.
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

    # Make element matrices and vectors
    cache_2d = FemCache(order_difference=2)

    # Create modified system to make it work with time marching.
    if time_settings is not None:
        if time_settings.sample_rate < 1:
            raise ValueError("Sample rate can not be less than 1.")

        if len(time_settings.time_march_relations) < 1:
            raise ValueError("Problem has no time march relations.")

        system = update_system_for_time_march(time_settings, system)

    compiled_system = CompiledSystem(system)

    # Make a system that can be used to perform an L2 projection for the initial
    # conditions.
    initial_funcs = list()
    for eq in system.equations:
        base_form = eq.weight.base_form
        initial_funcs.append(
            None
            if base_form not in system_settings.initial_conditions
            else system_settings.initial_conditions[base_form]
        )
    unknown_ordering = UnknownOrderings(*(form.order for form in system.unknown_forms))
    leaf_indices = tuple(int(val) for val in mesh.get_leaf_indices())

    dof_offsets, element_total_dof_counts, element_offset = _compute_offsets_and_sizes(
        mesh, unknown_ordering, leaf_indices, 0
    )

    # Explicit right side
    element_fem_spaces: list[ElementFemSpace2D] = list()

    for leaf_idx in leaf_indices:
        order_1, order_2 = mesh.get_leaf_orders(leaf_idx)
        element_cache = ElementFemSpace2D(
            cache_2d.get_basis2d(order_1, order_2),
            np.astype(mesh.get_leaf_corners(leaf_idx), np.float64, copy=False),
        )

        element_fem_spaces.append(element_cache)

    # Prepare for evaluation of matrices/vectors

    initial_vectors: list[npt.NDArray[np.float64]] = list()
    initial_solution: list[npt.NDArray[np.float64]] = list()
    if system_settings.initial_conditions:
        for element_space in element_fem_spaces:
            dual_dofs = compute_element_dual(
                unknown_ordering, initial_funcs, element_space
            )

            initial_vectors.append(dual_dofs)
            initial_solution.append(
                compute_element_primal(unknown_ordering, dual_dofs, element_space)
            )

    if initial_solution:
        solution = np.concatenate(initial_solution)
    else:
        solution = np.zeros(element_offset[-1])

    time_carry_index_array: npt.NDArray[np.uint32] | None = None
    if time_settings is not None:
        time_carry_index_array = np.concatenate(
            [
                find_time_carry_indices(
                    tuple(
                        sorted(
                            system.weight_forms.index(form)
                            for form in time_settings.time_march_relations
                        )
                    ),
                    dof_offsets[i],
                )
                + element_offset[i]
                for i in range(mesh.leaf_count)
            ]
        )

        if initial_vectors:
            # compute carry
            old_solution_carry = np.concatenate(initial_vectors)[time_carry_index_array]
        else:
            assert time_carry_index_array is not None
            old_solution_carry = np.zeros(time_carry_index_array.size, np.float64)
    else:
        time_carry_index_array = None
        old_solution_carry = None

    linear_vectors, linear_element_matrices, lagrange_mat, lagrange_vec = (
        _compute_linear_system(
            element_fem_spaces,
            dof_offsets,
            element_offset,
            leaf_indices,
            system,
            compiled_system,
            mesh,
            cache_2d,
            unknown_ordering,
            constrained_forms,
            boundary_conditions if boundary_conditions is not None else list(),
            initial_solution,
        )
    )
    del initial_vectors, initial_solution

    main_mat = sp.block_diag(linear_element_matrices, format="csr")
    main_vec = np.concatenate(linear_vectors, dtype=np.float64)

    if lagrange_mat is not None:
        main_mat = cast(
            sp.csr_array,
            sp.block_array(
                ((main_mat, lagrange_mat.T), (lagrange_mat, None)), format="csr"
            ),
        )
        main_vec = np.concatenate((main_vec, lagrange_vec))

    # TODO: Delet dis
    # from matplotlib import pyplot as plt

    # plt.figure()
    # # plt.imshow(main_mat.toarray())
    # plt.spy(main_mat)
    # plt.show()

    linear_matrix = sp.csc_array(main_mat)
    explicit_vec = main_vec

    if time_settings is not None:
        assert time_carry_index_array is not None
        time_carry_term = main_vec[time_carry_index_array]

    else:
        time_carry_term = None
    del main_mat, main_vec

    system_decomp = sla.splu(linear_matrix)

    resulting_grids: list[pv.UnstructuredGrid] = list()

    grid = reconstruct_mesh_from_solution(
        system, recon_order, element_fem_spaces, dof_offsets, solution
    )
    grid.field_data["time"] = (0.0,)
    resulting_grids.append(grid)

    global_lagrange = np.zeros_like(lagrange_vec)
    max_mag = np.abs(explicit_vec).max()

    max_iterations = solver_settings.maximum_iterations
    relax = solver_settings.relaxation
    atol = solver_settings.absolute_tolerance
    rtol = solver_settings.relative_tolerance

    changes: npt.NDArray[np.float64]
    iters: npt.NDArray[np.uint32]

    if time_settings is not None:
        nt = time_settings.nt
        dt = time_settings.dt
        changes = np.zeros(nt, np.float64)
        iters = np.zeros(nt, np.uint32)

        for time_index in range(nt):
            # 2 / dt * old_solution_carry + time_carry_term
            assert old_solution_carry is not None and time_carry_term is not None
            current_carry = 2 / dt * old_solution_carry + time_carry_term

            new_solution, global_lagrange, iter_cnt, max_residual = non_linear_solve_run(
                max_iterations,
                relax,
                atol,
                rtol,
                print_residual,
                unknown_ordering,
                element_fem_spaces,
                compiled_system,
                explicit_vec,
                element_offset,
                linear_element_matrices,
                time_carry_index_array,
                current_carry,
                solution,
                global_lagrange,
                max_mag,
                system_decomp,
                lagrange_mat,
                False,
            )

            changes[time_index] = float(max_residual[()])
            iters[time_index] = iter_cnt
            projected_solution = np.concatenate(
                [
                    compute_element_primal_to_dual(
                        unknown_ordering,
                        new_solution[element_offset[ie] : element_offset[ie + 1]],
                        *mesh.get_leaf_orders(leaf_idx),
                        element_fem_spaces[ie],
                    )
                    for ie, leaf_idx in enumerate(leaf_indices)
                ]
            )
            assert time_carry_index_array is not None
            new_solution_carry = projected_solution[time_carry_index_array]

            # Compute time carry
            new_time_carry_term = (
                2 / dt * (new_solution_carry - old_solution_carry) - time_carry_term
            )

            solution = new_solution
            time_carry_term = new_time_carry_term
            old_solution_carry = new_solution_carry
            del new_solution_carry, new_time_carry_term, new_solution, projected_solution

            if (time_index % time_settings.sample_rate) == 0 or time_index + 1 == nt:
                # Prepare to build up the 1D Splines

                grid = reconstruct_mesh_from_solution(
                    system, recon_order, element_fem_spaces, dof_offsets, solution
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
            max_iterations,
            relax,
            atol,
            rtol,
            print_residual,
            unknown_ordering,
            element_fem_spaces,
            compiled_system,
            explicit_vec,
            element_offset,
            linear_element_matrices,
            None,
            None,
            solution,
            global_lagrange,
            max_mag,
            system_decomp,
            lagrange_mat,
            True,
        )

        changes = np.asarray(changes, np.float64)[: iter_cnt + 1]
        iters = np.array((iter_cnt,), np.uint32)

        solution = new_solution
        del new_solution

        # Prepare to build up the 1D Splines

        grid = reconstruct_mesh_from_solution(
            system, recon_order, element_fem_spaces, dof_offsets, solution
        )

        resulting_grids.append(grid)

    mesh_orders = [mesh.get_leaf_orders(leaf_idx) for leaf_idx in leaf_indices]
    orders, counts = np.unique(mesh_orders, axis=0, return_counts=True)
    stats = SolutionStatistics(
        element_orders={
            (int(order[0]), int(order[1])): int(count)
            for order, count in zip(orders, counts)
        },
        n_total_dofs=explicit_vec.size,
        n_lagrange=int(lagrange_vec.size),
        n_elems=mesh.element_count,
        n_leaves=mesh.leaf_count,
        n_leaf_dofs=element_total_dof_counts.sum(),
        iter_history=iters,
        residual_history=np.asarray(changes, np.float64),
    )

    if refinement_settings is not None:
        if refinement_settings.report_order_distribution:
            order_hist = HistogramFormat(5, 60, 5, label_format=lambda x: f"{x:.1f}")
            geo_order = np.linalg.norm(mesh_orders, axis=1) / np.sqrt(2)
            print("Initial mesh order distribution\n" + "=" * 60)
            print(order_hist.format(geo_order))
            print("=" * 60)
        else:
            order_hist = None

        output_mesh = perform_mesh_refinement(
            mesh,
            solution,
            element_offset,
            dof_offsets,
            [
                system.unknown_forms.index(form)
                for form in refinement_settings.required_forms
            ],
            system.unknown_forms,
            refinement_settings.error_calculation_function,
            refinement_settings.h_refinement_ratio,
            refinement_settings.refinement_limit,
            unknown_ordering,
            refinement_settings.report_error_distribution,
            element_fem_spaces,
            (None, None)
            if refinement_settings.reconstruction_orders is None
            else refinement_settings.reconstruction_orders,
        )
        if refinement_settings.report_order_distribution:
            assert order_hist is not None
            geo_order = np.linalg.norm(
                [
                    output_mesh.get_leaf_orders(ie)
                    for ie in output_mesh.get_leaf_indices()
                ],
                axis=1,
            ) / np.sqrt(2)
            print("Refined mesh order distribution\n" + "=" * 60)
            print(order_hist.format(geo_order))
            print("=" * 60)

    else:
        output_mesh = mesh

    return tuple(resulting_grids), stats, output_mesh


def _compute_offsets_and_sizes(
    mesh: Mesh, unknown_ordering: UnknownOrderings, leaf_indices: Sequence[int], dp: int
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Compute DoF offsets and sizes for the mesh and the system."""
    dof_sizes = np.array(
        [
            compute_leaf_dof_counts(orders[0] + dp, orders[1] + dp, unknown_ordering)
            for orders in (mesh.get_leaf_orders(leaf_idx) for leaf_idx in leaf_indices)
        ],
        np.uint32,
    )
    dof_offsets = np.pad(np.cumsum(dof_sizes, axis=1), ((0, 0), (1, 0)))
    element_total_dof_counts = dof_offsets[:, -1]
    element_offset = np.astype(
        np.pad(element_total_dof_counts.cumsum(), (1, 0)), np.uint32, copy=False
    )

    return dof_offsets, element_total_dof_counts, element_offset


def update_system_for_time_march(
    time_settings: TimeSettings, system: KFormSystem
) -> KFormSystem:
    """Update system for use with trapezoidal time march."""
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

    new_equations: list[KEquation] = list()
    for eq, m_idx in zip(system.equations, time_march_indices):
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

    return KFormSystem(*new_equations)


# def do_vms(
#     mesh: Mesh,
#     unknown_ordering: UnknownOrderings,
#     leaf_indices: Sequence[int],
#     basis_cache: FemCache,
#     element_fem_spaces: Sequence[ElementFemSpace2D],
#     order_diff: int,
#     boundary_conditions: Sequence[BoundaryCondition2DSteady],
#     constrained_forms: Sequence[tuple[float, KFormUnknown]],
#     original_system: KFormSystem,
#     symmetric_system: KFormSystem,
#     antisymmetric_system: KFormSystem,
#     coarse_forcing: npt.NDArray[np.float64],
#     coarse_solutions: Sequence[npt.NDArray[np.float64]],
#     coarse_lagrange_mat: None | sp.csr_array,
# ):
#     """Do the VMS here."""
#     fine_fem_spaces = [
#         ElementFemSpace2D(
#             Basis2D(
#                 basis_cache.get_basis1d(element_space.basis_xi.order + order_diff),
#                 basis_cache.get_basis1d(element_space.basis_eta.order + order_diff),
#             ),
#             element_space.corners,
#         )
#         for element_space in element_fem_spaces
#     ]

#     fine_dof_offsets, fine_element_total_dof_counts, fine_element_offset = (
#         _compute_offsets_and_sizes(mesh, unknown_ordering, leaf_indices, order_diff)
#     )

#     compiled_symmetric = CompiledSystem(symmetric_system)
#     fine_linear_vectors = [
#         compute_element_rhs(original_system, cache) for cache in fine_fem_spaces
#     ]
#     fine_lagrange_mat, _ = _add_system_constraints(
#         original_system,
#         mesh,
#         basis_cache,
#         unknown_ordering,
#         constrained_forms,
#         boundary_conditions,
#         leaf_indices,
#         fine_dof_offsets,
#         fine_element_offset,
#         fine_linear_vectors,
#     )

#     fine_linear_matrices = [
#         compute_element_matrix(
#             unknown_ordering.form_orders,
#             compiled_symmetric.linear_codes,
#             tuple(),
#             element_space,
#         )
#         for element_space in fine_fem_spaces
#     ]

#     element_projectors = [
#         compute_element_projector(
#             unknown_ordering.form_orders,
#             fine_space.corners,
#             fine_space.basis_2d,
#             coarse_space.basis_2d,
#         )
#         for coarse_space, fine_space in zip(
#             element_fem_spaces, fine_fem_spaces, strict=True
#         )
#     ]

#     projection_matrix = cast(sp.csr_array, sp.block_diag(element_projectors, "csr"))

#     compiled_antisymmetric = CompiledSystem(antisymmetric_system)
#     fine_antisym_matrices = [
#         compute_element_matrix(
#             unknown_ordering.form_orders,
#             compiled_antisymmetric.lhs_full,
#             compiled_antisymmetric.vector_field_specs,
#             element_space,
#             proj @ sol,
#         )
#         for element_space, proj, sol in zip(
#             fine_fem_spaces, element_projectors, coarse_solutions, strict=True
#         )
#     ]

#     coarse_linear_matrices = [
#         compute_element_matrix(
#             unknown_ordering.form_orders,
#             compiled_symmetric.linear_codes,
#             tuple(),
#             element_space,
#         )
#         for element_space in element_fem_spaces
#     ]

#     fine_forcing = np.concatenate(fine_linear_vectors)
#     residual = fine_forcing - projection_matrix @ coarse_forcing
#     # Assume Lagrange multipliers are satisfied.

#     system_matrix_fine = sp.block_diag(fine_linear_matrices)

#     system_matrix_coarse = sp.block_diag(coarse_linear_matrices)
#     if coarse_lagrange_mat is not None:
#         system_matrix_coarse = sp.block_array(
#             [
#                 [system_matrix_coarse, coarse_lagrange_mat.T],
#                 [coarse_lagrange_mat, None],
#             ]
#         )

#     fine_mass_inverse = sp.block_diag(
#         [
#             element_space.mass_from_order(order, inverse=True)
#             for order in unknown_ordering.form_orders
#             for element_space in fine_fem_spaces
#         ]
#     )
#     fine_advection = sp.block_diag(fine_antisym_matrices)
#     if fine_lagrange_mat is not None:
#         n = fine_lagrange_mat.shape[0]
#         residual = np.pad(residual, (0, n))
#         system_matrix_fine = sp.block_array(
#             [
#                 [system_matrix_fine, fine_lagrange_mat.T],
#                 [fine_lagrange_mat, None],
#             ]
#         )
#         fine_mass_inverse.resize(
#             fine_mass_inverse.shape[0] + n, fine_mass_inverse.shape[1] + n
#         )
#         fine_advection.resize(fine_advection.shape[0] + n, fine_advection.shape[1] + n)

#     fine_part = sla.spsolve(system_matrix_fine, fine_mass_inverse)

#     coarse_part = projection_matrix.T @ sla.spsolve(
#         system_matrix_coarse, projection_matrix @ fine_mass_inverse
#     )

#     fine_scale_green = fine_part - coarse_part
#     advected_green = fine_advection @ fine_scale_green
#     return advected_green
#     # sg_result = sla.spsolve(fine_mass_inverse)
