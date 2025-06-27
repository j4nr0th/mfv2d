"""Solving the actual system."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import ElementMassMatrixCache
from mfv2d.element import (
    ElementCollection,
    FixedElementArray,
    FlexibleElementArray,
    call_per_element_fix,
    call_per_element_flex,
    call_per_leaf_flex,
    call_per_leaf_obj,
    compute_dof_sizes,
    compute_lagrange_sizes,
)
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KEquation, KFormSystem, UnknownOrderings
from mfv2d.mimetic2d import FemCache, Mesh2D
from mfv2d.solve_system import (
    Constraint,
    ElementConstraint,
    RefinementSettings,
    SolutionStatistics,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    assemble_matrix,
    assemble_vector,
    check_and_refine,
    compute_element_dual,
    compute_element_primal,
    compute_element_primal_to_dual,
    compute_element_rhs,
    compute_element_vector_fields,
    compute_leaf_matrix,
    divide_old,
    extract_carry,
    find_time_carry_indices,
    mesh_boundary_conditions,
    mesh_continuity_constraints,
    non_linear_solve_run,
    reconstruct_mesh_from_solution,
)


def solve_system_2d(
    mesh: Mesh2D,
    system_settings: SystemSettings,
    refinement_settings: RefinementSettings = RefinementSettings(
        refinement_levels=0,
        division_predicate=None,
        division_function=divide_old,
    ),
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
    cache_2d = FemCache(order_difference=2)

    unique_order_pairs = element_collection.orders_array.unique(axis=1)

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

        system = KFormSystem(*new_equations)
        del new_equations

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

    def compute_element_cache(
        ie: int,
        fem_cache: FemCache,
        orders: FixedElementArray[np.uint32],
        corners: FixedElementArray[np.float64],
    ) -> ElementMassMatrixCache:
        """Compute cache for that element."""
        order_1, order_2 = orders[ie]
        basis = fem_cache.get_basis2d(order_1, order_2)
        return ElementMassMatrixCache(basis, corners[ie])

    # Create element caches
    element_caches = call_per_leaf_obj(
        element_collection,
        ElementMassMatrixCache,
        compute_element_cache,
        cache_2d,
        element_collection.orders_array,
        element_collection.corners_array,
    )

    # Prepare for evaluation of matrices/vectors

    leaf_elements = np.flatnonzero(
        np.concatenate(element_collection.child_count_array.values) == 0
    )

    linear_vectors = call_per_leaf_flex(
        element_collection, 1, np.float64, compute_element_rhs, system, element_caches
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
        initial_vectors = call_per_leaf_flex(
            element_collection,
            1,
            np.float64,
            compute_element_dual,
            unknown_ordering,
            initial_funcs,
            element_collection.orders_array,
            element_caches,
        )

        initial_solution = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            compute_element_primal,
            unknown_ordering,
            initial_vectors,
            element_collection.orders_array,
            element_caches,
        )

    else:
        initial_vectors = None
        initial_solution = None

    if time_settings is not None:
        time_carry_index_array = call_per_element_flex(
            element_collection.com,
            1,
            np.uint32,
            find_time_carry_indices,
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
        if initial_solution is not None:
            solution = initial_solution

        time_carry_index_array = None
        old_solution_carry = None

    del initial_solution, initial_vectors

    assert compiled_system.linear_codes

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

    linear_element_matrices = call_per_leaf_flex(
        element_collection,
        2,
        np.float64,
        compute_leaf_matrix,
        compiled_system.linear_codes,
        unknown_ordering,
        element_caches,
        vec_fields_array,
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
        cache_2d,
    )

    really_unique = np.unique(unique_order_pairs)

    continuity_constraints = mesh_continuity_constraints(
        system,
        mesh,
        top_indices,
        unknown_ordering,
        element_collection,
        really_unique.size > 1 or really_unique[0] != 1,
        dof_offsets,
    )
    del really_unique

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
        main_vec[ec.dofs + offset] += ec.coeffs

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
        cache_2d,
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

            changes[time_index] = float(max_residual)
            iters[time_index] = iter_cnt
            projected_solution = call_per_leaf_flex(
                element_collection,
                1,
                np.float64,
                compute_element_primal_to_dual,
                unknown_ordering,
                new_solution,
                element_collection.orders_array,
                element_caches,
            )
            assert time_carry_index_array is not None
            new_solution_carry = extract_carry(
                element_collection, time_carry_index_array, projected_solution
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
            del new_solution_carry, new_time_carry_term, new_solution, projected_solution

            if (time_index % time_settings.sample_rate) == 0 or time_index + 1 == nt:
                # Prepare to build up the 1D Splines

                grid = reconstruct_mesh_from_solution(
                    system,
                    recon_order,
                    element_collection,
                    cache_2d,
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
            system,
            recon_order,
            element_collection,
            cache_2d,
            dof_offsets,
            solution,
        )

        resulting_grids.append(grid)

    orders, counts = np.unique(
        np.array(element_collection.orders_array), return_counts=True
    )
    stats = SolutionStatistics(
        element_orders={int(order): int(count) for order, count in zip(orders, counts)},
        n_total_dofs=explicit_vec.size,
        n_lagrange=int(lagrange_vec.size + np.array(lagrange_counts).sum()),
        n_elems=element_collection.com.element_cnt,
        n_leaves=len(leaf_elements),
        n_leaf_dofs=sum(int(total_dof_counts[int(ie)][0]) for ie in leaf_elements),
        iter_history=iters,
        residual_history=np.asarray(changes, np.float64),
    )

    return tuple(resulting_grids), stats
