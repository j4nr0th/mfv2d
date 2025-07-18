"""Solving the actual system."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import ElementMassMatrixCache, Mesh, compute_element_matrix
from mfv2d.boundary import mesh_boundary_conditions
from mfv2d.continuity import connect_elements
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KEquation, KFormSystem, KFormUnknown, UnknownOrderings
from mfv2d.mimetic2d import (
    Constraint,
    ElementConstraint,
    FemCache,
    compute_leaf_dof_counts,
)
from mfv2d.solve_system import (
    SolutionStatistics,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    compute_element_dual,
    compute_element_primal,
    compute_element_primal_to_dual,
    compute_element_rhs,
    compute_element_vector_fields_nonlin,
    find_time_carry_indices,
    non_linear_solve_run,
    reconstruct_mesh_from_solution,
)


def solve_system_2d(
    mesh: Mesh,
    system_settings: SystemSettings,
    # refinement_settings: RefinementSettings = RefinementSettings(
    #     refinement_levels=0,
    #     division_predicate=None,
    #     division_function=divide_old,
    # ),
    solver_settings: SolverSettings = SolverSettings(
        maximum_iterations=100,
        relaxation=1,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    ),
    time_settings: TimeSettings | None = None,
    *,
    recon_order: int | None = None,
    print_residual: bool = False,
) -> tuple[Sequence[pv.UnstructuredGrid], SolutionStatistics]:
    """Solve the unsteady system on the specified mesh.

    Parameters
    ----------
    mesh : Mesh2D
        Mesh on which to solve the system on.

    system_settings : SystemSettings
        Settings specifying the system of equations and boundary conditions to solve for.

    refinement_settings : RefinementSettings, optional
        Settings specifying refinement of the mesh.

    solver_settings : SolverSettings, optional
        Settings specifying the behavior of the solver

    time_settings : TimeSettings or None, default: None
        When set to ``None``, the equations are solved without time dependence (steady
        state). Otherwise, it specifies which equations are time derivative related and
        time step count and size.

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

    vector_fields = system.vector_fields

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

    # Explicit right side
    explicit_vec: npt.NDArray[np.float64]
    element_caches: list[ElementMassMatrixCache] = list()
    linear_vectors: list[npt.NDArray[np.float64]] = list()
    leaf_indices = tuple(int(val) for val in mesh.get_leaf_indices())
    unknown_ordering = UnknownOrderings(*(form.order for form in system.unknown_forms))
    element_dof_counts: list[npt.NDArray[np.uint32]] = list()

    for leaf_idx in leaf_indices:
        element_cache = ElementMassMatrixCache(
            cache_2d.get_basis2d(*mesh.get_leaf_orders(leaf_idx)),
            np.astype(mesh.get_leaf_corners(leaf_idx), np.float64, copy=False),
        )

        element_caches.append(element_cache)
        linear_vectors.append(compute_element_rhs(system, element_cache))
        element_dof_counts.append(
            compute_leaf_dof_counts(leaf_idx, unknown_ordering, mesh)
        )

    # Prepare for evaluation of matrices/vectors

    dof_sizes = np.array(element_dof_counts, np.uint32)
    dof_offsets = np.pad(np.cumsum(dof_sizes, axis=1), ((0, 0), (1, 0)))
    element_total_dof_counts = dof_offsets[:, -1]

    element_offset = np.astype(
        np.pad(element_total_dof_counts.cumsum(), (1, 0)),
        np.uint32,
        copy=False,
    )

    initial_vectors: list[npt.NDArray[np.float64]] = list()
    initial_solution: list[npt.NDArray[np.float64]] = list()
    if system_settings.initial_conditions:
        for i, leaf_idx in enumerate(leaf_indices):
            order_1, order_2 = mesh.get_leaf_orders(leaf_idx)
            dual_dofs = compute_element_dual(
                unknown_ordering,
                initial_funcs,
                order_1,
                order_2,
                element_caches[i],
            )

            initial_vectors.append(dual_dofs)
            initial_solution.append(
                compute_element_primal(
                    unknown_ordering,
                    dual_dofs,
                    order_1,
                    order_2,
                    element_caches[i],
                )
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

    assert compiled_system.linear_codes

    # Compute vector fields at integration points for leaf elements
    linear_element_matrices: list[npt.NDArray[np.float64]] = list()

    for ie in range(mesh.leaf_count):
        element_cache = element_caches[ie]
        basis = element_cache.basis_2d
        if len(vector_fields):
            # Recompute vector fields
            # Compute vector fields at integration points for leaf elements
            vec_flds = compute_element_vector_fields_nonlin(
                system.unknown_forms,
                basis,
                basis,
                vector_fields,
                mesh.get_leaf_corners(ie),
                dof_offsets[ie],
                solution[element_offset[ie] : element_offset[ie + 1]],
            )
        else:
            vec_flds = tuple()

        linear_element_matrices.append(
            compute_element_matrix(
                unknown_ordering.form_orders,
                compiled_system.linear_codes,
                vec_flds,
                element_cache,
            )
        )

        # if initial_vectors and compiled_system.rhs_codes is not None:
        #     rhs_vec = compute_element_vector(
        #         unknown_ordering.form_orders,
        #         compiled_system.rhs_codes,
        #         vec_flds,
        #         element_cache,
        #         initial_solution[ie],
        #     )
        #     linear_vectors[ie] += rhs_vec

    del initial_vectors, initial_solution
    main_mat = sp.block_diag(linear_element_matrices, format="csr")
    main_vec = np.concatenate(linear_vectors, dtype=np.float64)

    # Generate constraints that force the specified for to have the (child element) sum
    # equal to a prescribed value.
    constrained_form_constaints: dict[KFormUnknown, Constraint] = dict()
    for k, form in constrained_forms:
        i_unknown = system.unknown_forms.index(form)
        constrained_form_constaints[form] = Constraint(
            k,
            *(
                ElementConstraint(
                    ie,
                    np.arange(
                        dof_offsets[ie, i_unknown],
                        dof_offsets[ie, i_unknown + 1],
                        dtype=np.uint32,
                    ),
                    np.ones(dof_offsets[ie, i_unknown + 1] - dof_offsets[ie, i_unknown]),
                )
                for ie in range(mesh.leaf_count)
            ),
        )

    reverse_mapping = {leaf_idx: i for i, leaf_idx in enumerate(leaf_indices)}
    if boundary_conditions is None:
        boundary_conditions = list()

    strong_bc_constraints, weak_bc_constraints = mesh_boundary_conditions(
        [eq.right for eq in system.equations],
        unknown_ordering,
        mesh,
        reverse_mapping,
        dof_offsets,
        [
            [bc for bc in boundary_conditions if bc.form == eq.weight.base_form]
            for eq in system.equations
        ],
        cache_2d,
    )

    continuity_constraints = connect_elements(
        unknown_ordering, mesh, reverse_mapping, dof_offsets
    )

    constraint_rows: list[npt.NDArray[np.uint32]] = list()
    constraint_cols: list[npt.NDArray[np.uint32]] = list()
    constraint_coef: list[npt.NDArray[np.float64]] = list()
    constraint_vals: list[float] = list()
    # Continuity constraints
    ic = 0
    for constraint in continuity_constraints:
        constraint_vals.append(constraint.rhs)
        # print(f"Continuity constraint {ic=}:")
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        #     print(ec)
        # print("")
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
        main_vec[ec.dofs + offset] += ec.coeffs

    if constraint_coef:
        lagrange_mat = sp.csr_array(
            (
                np.concatenate(constraint_coef),
                (
                    np.concatenate(constraint_rows, dtype=np.intp),
                    np.concatenate(constraint_cols, dtype=np.intp),
                ),
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
        system, recon_order, mesh, cache_2d, leaf_indices, dof_offsets, solution
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
                system,
                max_iterations,
                relax,
                atol,
                rtol,
                print_residual,
                unknown_ordering,
                mesh,
                cache_2d,
                element_caches,
                compiled_system,
                explicit_vec,
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

            changes[time_index] = float(max_residual[()])
            iters[time_index] = iter_cnt
            projected_solution = np.concatenate(
                [
                    compute_element_primal_to_dual(
                        unknown_ordering,
                        new_solution[element_offset[ie] : element_offset[ie + 1]],
                        *mesh.get_leaf_orders(leaf_idx),
                        element_caches[ie],
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
                    system,
                    recon_order,
                    mesh,
                    cache_2d,
                    leaf_indices,
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
            mesh,
            cache_2d,
            element_caches,
            compiled_system,
            explicit_vec,
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
            True,
        )

        changes = np.asarray(changes, np.float64)[: iter_cnt + 1]
        iters = np.array((iter_cnt,), np.uint32)

        solution = new_solution
        del new_solution

        # Prepare to build up the 1D Splines

        grid = reconstruct_mesh_from_solution(
            system, recon_order, mesh, cache_2d, leaf_indices, dof_offsets, solution
        )

        resulting_grids.append(grid)

    orders, counts = np.unique(
        [mesh.get_leaf_orders(leaf_idx) for leaf_idx in leaf_indices], return_counts=True
    )
    stats = SolutionStatistics(
        element_orders={int(order): int(count) for order, count in zip(orders, counts)},
        n_total_dofs=explicit_vec.size,
        n_lagrange=int(lagrange_vec.size),
        n_elems=mesh.element_count,
        n_leaves=mesh.leaf_count,
        n_leaf_dofs=element_total_dof_counts.sum(),
        iter_history=iters,
        residual_history=np.asarray(changes, np.float64),
    )

    return tuple(resulting_grids), stats


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

    return KFormSystem(*new_equations)
