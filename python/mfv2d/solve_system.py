"""Implementation of solve functions.

Functions in this file all deal with solving the full system. Examples
of these include the assembly of the global matrix, the application of the
boundary conditions, and computing the right side of the system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from mfv2d._mfv2d import (
    ElementFemSpace2D,
    Mesh,
    compute_element_matrix,
    compute_element_projector,
    compute_element_vector,
)
from mfv2d.boundary import BoundaryCondition2DSteady
from mfv2d.continuity import add_system_constraints
from mfv2d.eval import CompiledSystem
from mfv2d.kform import (
    KElementProjection,
    KExplicit,
    KFormUnknown,
    KWeight,
    UnknownFormOrder,
)
from mfv2d.mimetic2d import (
    FemCache,
    bilinear_interpolate,
    element_dual_dofs,
    reconstruct,
    vtk_lagrange_ordering,
)
from mfv2d.progress import ProgressTracker
from mfv2d.system import ElementFormSpecification, KFormSystem


def rhs_2d_element_projection(
    right: KElementProjection, element_space: ElementFemSpace2D
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
        return np.zeros(
            right.weight.order.full_unknown_count(*element_space.orders),
            np.float64,
        )

    return element_dual_dofs(right.weight.order, element_space, fn)


def _extract_rhs_2d(
    proj: Sequence[tuple[float, KExplicit]],
    weight: KWeight,
    element_space: ElementFemSpace2D,
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
    # Create empty vector into which to accumulate
    vec = np.zeros(
        weight.order.full_unknown_count(*element_space.orders),
        np.float64,
    )

    # Loop over all entries that are KElementProjection
    for k, f in proj:
        if not isinstance(f, KElementProjection):
            continue

        rhs = rhs_2d_element_projection(f, element_space)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def compute_element_rhs(
    system: KFormSystem,
    elem_cache: ElementFemSpace2D,
) -> npt.NDArray[np.float64]:
    """Compute rhs for an element.

    This basically means just concatenating the projections of the functions on the
    element for each of the equations in the system.

    Parameters
    ----------
    ie : int
        Index of the element.

    system : KFormSystem
        System for which to compute the rhs.

    element_spaces : FemCache
        Cache from which to get the basis from.

    Returns
    -------
    array
        Array with the resulting rhs.
    """
    return np.concatenate(
        [
            _extract_rhs_2d(equation.right.explicit_terms, equation.weight, elem_cache)
            for equation in system.equations
        ],
        dtype=np.float64,
    )


def reconstruct_mesh_from_solution(
    form_spec: ElementFormSpecification,
    recon_order: int | None,
    fem_spaces: list[ElementFemSpace2D],
    solution: npt.NDArray[np.float64],
    vms_solution: npt.NDArray[np.float64] | None,
) -> pv.UnstructuredGrid:
    """Reconstruct the unknown differential forms."""
    build: dict[KFormUnknown, list[npt.NDArray[np.float64]]] = {
        form: list() for form in form_spec.iter_forms()
    }
    if vms_solution is not None:
        vms_build: dict[KFormUnknown, list[npt.NDArray[np.float64]]] = {
            form: list() for form in form_spec.iter_forms()
        }
    else:
        vms_build = dict()

    xvals: list[npt.NDArray[np.float64]] = list()
    yvals: list[npt.NDArray[np.float64]] = list()
    order_list: list[tuple[int, int]] = list()

    node_array: list[npt.NDArray[np.int32]] = list()
    node_cnt = 0
    element_offset = 0

    used_nodes: dict[int, npt.NDArray[np.float64]] = dict()
    for element_space in fem_spaces:
        # Extract element DoFs
        orders = element_space.orders
        element_dof_count = form_spec.total_size(*orders)
        element_dofs = solution[element_offset : element_offset + element_dof_count]
        corners = element_space.corners

        order_list.append(orders)
        reconstruction_order = max(orders) if recon_order is None else recon_order
        if reconstruction_order not in used_nodes:
            used_nodes[reconstruction_order] = np.linspace(
                -1, +1, reconstruction_order + 1, dtype=np.float64
            )
        recon_nodes = used_nodes[reconstruction_order]

        ordering = vtk_lagrange_ordering(reconstruction_order) + node_cnt
        node_array.append(np.concatenate(((ordering.size,), ordering)))
        node_cnt += ordering.size
        ex = bilinear_interpolate(
            corners[:, 0], recon_nodes[None, :], recon_nodes[:, None]
        )
        ey = bilinear_interpolate(
            corners[:, 1], recon_nodes[None, :], recon_nodes[:, None]
        )

        xvals.append(ex.flatten())
        yvals.append(ey.flatten())
        # Loop over each of the primal forms
        for idx, form in enumerate(form_spec.iter_forms()):
            form_offset = form_spec.form_offset(idx, *orders)
            form_offset_end = form_offset + form_spec.form_size(idx, *orders)
            form_dofs = element_dofs[form_offset:form_offset_end]
            if not form.is_primal:
                raise ValueError("Can not reconstruct a non-primal form.")
            # Reconstruct unknown
            recon_v = reconstruct(
                element_space,
                form.order,
                form_dofs,
                recon_nodes[None, :],
                recon_nodes[:, None],
            )
            shape = (-1, 2) if form.order == UnknownFormOrder.FORM_ORDER_1 else (-1,)
            build[form].append(np.reshape(recon_v, shape))

            if vms_solution is not None:
                vms_dofs = vms_solution[
                    element_offset : element_offset + element_dof_count
                ][form_offset:form_offset_end]
                vms_dofs = (
                    element_space.mass_from_order(form.order, inverse=True) @ vms_dofs
                )
                recon_vms = reconstruct(
                    element_space,
                    form.order,
                    vms_dofs,
                    recon_nodes[None, :],
                    recon_nodes[:, None],
                )
                vms_build[form].append(np.reshape(recon_vms, shape))

        element_offset += element_dof_count

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

    for form in vms_build:
        vf = np.concatenate(vms_build[form], axis=0, dtype=np.float64)
        grid.point_data["vms-" + form.label] = vf

    grid.cell_data["orders"] = order_list

    return grid


def compute_element_dual(
    form_specs: ElementFormSpecification,
    functions: Sequence[
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike] | None
    ],
    element_space: ElementFemSpace2D,
) -> npt.NDArray[np.float64]:
    """Compute element L2 projection."""
    vecs: list[npt.NDArray[np.float64]] = list()
    for i_form, func in enumerate(functions):
        if func is None:
            vecs.append(
                np.zeros(
                    form_specs.form_size(i_form, *element_space.orders),
                    np.float64,
                )
            )
        else:
            vecs.append(
                np.asarray(
                    element_dual_dofs(form_specs[i_form][1], element_space, func),
                    np.float64,
                )
            )

    return np.concatenate(vecs)


def compute_element_dual_from_primal(
    form_specs: ElementFormSpecification,
    primal: npt.NDArray[np.float64],
    element_space: ElementFemSpace2D,
) -> npt.NDArray[np.float64]:
    """Compute dual dofs from primal."""
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    dual = np.empty_like(primal)
    for i_form in range(len(form_specs)):
        cnt = form_specs.form_size(i_form, *element_space.orders)
        v = primal[offset : offset + cnt]
        order = form_specs[i_form][1]
        if order in mats:
            m = mats[order]
        else:
            m = element_space.mass_from_order(order, inverse=False)
            mats[order] = m

        dual[offset : offset + cnt] = m @ v

        offset += cnt

    return dual


def compute_element_primal_from_dual(
    form_specs: ElementFormSpecification,
    dual: npt.NDArray[np.float64],
    element_space: ElementFemSpace2D,
) -> npt.NDArray[np.float64]:
    """Compute primal dofs from dual."""
    offset = 0
    mats: dict[UnknownFormOrder, npt.NDArray[np.float64]] = dict()
    primal = np.empty_like(dual)
    for i_form in range(len(form_specs)):
        cnt = form_specs.form_size(i_form, *element_space.orders)
        v = dual[offset : offset + cnt]
        order = form_specs[i_form][1]
        if order in mats:
            m = mats[order]
        else:
            m = element_space.mass_from_order(order, inverse=True)
            mats[order] = m

        primal[offset : offset + cnt] = m @ v

        offset += cnt

    return primal


def non_linear_solve_run(
    max_iterations: int,
    relax: float,
    atol: float,
    rtol: float,
    print_residual: bool,
    form_spec: ElementFormSpecification,
    element_fem_spaces: Sequence[ElementFemSpace2D],
    compiled_system: CompiledSystem,
    explicit_vec: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    linear_element_matrices: Sequence[npt.NDArray[np.float64]],
    time_carry_index_array: npt.NDArray[np.uint32] | None,
    time_carry_term: npt.NDArray[np.float64] | None,
    solution: npt.NDArray[np.float64],
    global_lagrange: npt.NDArray[np.float64],
    max_mag: float,
    system_decomp: sla.SuperLU,
    lagrange_mat: sp.csr_array | None,
    fine_scales: npt.NDArray[np.float64] | None,
    sg_operator: SuyashGreenOperator | None,
    return_all_residuals: bool = False,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
]:
    """Run the iterative non-linear solver.

    Based on how the compiled system looks, this may only take a single iteration,
    otherwise, it may run for as long as it needs to converge.
    """
    progress_tracker: None | ProgressTracker = None
    iter_cnt = 0
    base_vec = np.array(explicit_vec, copy=True)  # Make a copy
    if time_carry_term is not None:
        assert time_carry_index_array is not None
        base_vec[time_carry_index_array] += time_carry_term
    else:
        assert time_carry_index_array is None
    residuals = np.zeros(max_iterations, np.float64)
    max_residual = 0.0
    unresolved_scales = fine_scales

    while iter_cnt < max_iterations:
        if compiled_system.rhs_codes is not None:
            main_vec = np.array(base_vec, copy=True)  # Make a copy
        else:
            main_vec = base_vec  # Do not copy

        lhs_vectors: list[npt.NDArray[np.float64]] = list()
        lhs_matrices: list[npt.NDArray[np.float64]] = list()
        for ie, element_space in enumerate(element_fem_spaces):
            element_solution = solution[element_offsets[ie] : element_offsets[ie + 1]]

            lhs_vec = compute_element_vector(
                form_spec, compiled_system.lhs_full, element_space, element_solution
            )

            if compiled_system.rhs_codes is not None:
                rhs_vec = compute_element_vector(
                    form_spec, compiled_system.rhs_codes, element_space, element_solution
                )
                lhs_vec -= rhs_vec
                # main_vec[element_offsets[ie] : element_offsets[ie + 1]] += rhs_vec

                del rhs_vec
            lhs_vectors.append(lhs_vec)

            if compiled_system.nonlin_codes is not None:
                mat = compute_element_matrix(
                    form_spec,
                    compiled_system.nonlin_codes,
                    element_space,
                    element_solution,
                )
                lhs_matrices.append(linear_element_matrices[ie] + mat)

        main_value = np.concatenate(lhs_vectors)

        if lagrange_mat is not None:
            main_value += lagrange_mat.T @ global_lagrange
            main_value = np.concatenate(
                (main_value, lagrange_mat @ solution), dtype=np.float64
            )

        residual = main_vec - main_value
        if sg_operator is not None:
            sg_operator.update_nonlinear_advection(solution)
            unresolved_scales = sg_operator.compute_unresolved_contributions(
                solution, unresolved_scales
            )
            residual -= sg_operator.fine_results_to_coarse_dofs(
                unresolved_scales, dual=True
            )

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

        if len(lhs_matrices):
            main_mat = sp.block_diag(lhs_matrices, format="csr")
            if lagrange_mat is not None:
                main_mat = sp.block_array(
                    [[main_mat, lagrange_mat.T], [lagrange_mat, None]], format="csc"
                )
            else:
                main_mat = main_mat.tocsc()

            system_decomp = sla.splu(main_mat)
            del main_mat, lhs_matrices

        d_solution = np.asarray(
            system_decomp.solve(residual),
            dtype=np.float64,
            copy=None,
        )

        # update lagrange multipliers (haha pliers)
        if len(global_lagrange):
            solution += relax * d_solution[: -global_lagrange.size]
            global_lagrange += relax * d_solution[-global_lagrange.size :]
        else:
            solution += relax * d_solution

        iter_cnt += 1

        del main_vec

    if not return_all_residuals:
        return (
            solution,
            global_lagrange,
            iter_cnt,
            np.array(max_residual, np.float64),
            unresolved_scales,
        )

    return solution, global_lagrange, iter_cnt, residuals, unresolved_scales


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

    over_integration_order : int, default: 3
        The order by which the integration rule used to compute mass matrices of the
        elements should be greater from the basis used. This is important if a specific
        projection to a finer polynomial space is needed later.
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
    over_integration_order: int = 3


@dataclass(frozen=True)
class ConvergenceSettings:
    """Settings used to specify convergence of an iterative solver."""

    maximum_iterations: int = 100
    """Maximum number of iterations to improve the solution."""

    absolute_tolerance: float = 1e-6
    """When the largest update in the solution drops bellow this value,
    consider it converged."""

    relative_tolerance: float = 1e-5
    """When the largest update in the solution drops bellow the largest
    value of solution degrees of freedom scaled by this value, consider
    it converged."""


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
    """

    convergence: ConvergenceSettings
    """When should the solution be considered converged"""

    relaxation: float = 1.0
    """What fraction of the new solution should be used."""


def find_time_carry_indices(
    unknowns: Sequence[int],
    form_specs: ElementFormSpecification,
    order_1: int,
    order_2: int,
) -> npt.NDArray[np.uint32]:
    """Find what are the indices of DoFs that should be carried for the time march."""
    output: list[npt.NDArray[np.uint32]] = list()
    for iu, u in enumerate(unknowns):
        assert iu == 0 or unknowns[iu] < u, "Unknowns must be sorted."
        offset = form_specs.form_offset(u, order_1, order_2)
        size = form_specs.form_size(u, order_1, order_2)
        output.append(offset + np.arange(size, dtype=np.uint32))
    return np.concatenate(output, dtype=np.uint32)


@dataclass(frozen=True)
class SolutionStatistics:
    """Information about the solution."""

    element_orders: dict[tuple[int, int], int]
    n_total_dofs: int
    n_leaf_dofs: int
    n_lagrange: int
    n_elems: int
    n_leaves: int
    iter_history: npt.NDArray[np.uint32]
    residual_history: npt.NDArray[np.float64]


@dataclass(frozen=True)
class VMSSettings:
    """Estimate unresolved scales based on the variational multi-scale approach."""

    symmetric_system: KFormSystem
    """Symmetric system based on which the Green's function is computed."""

    nonsymmetric_system: KFormSystem
    """Non-symmetric system based on which the fine-scale function is computed."""

    order_increase: int
    """Increase of order for the fine mesh."""

    fine_scale_convergence: ConvergenceSettings
    """When are the fine scales considered accurately computed."""

    relaxation: float = 1.0
    """How much of the computed update to use for the next iteration."""


class SuyashGreenOperator:
    """Type used to apply the Suyash-Green operator."""

    unknown_forms: ElementFormSpecification
    compiled_advection: CompiledSystem
    coarse_decomp: sla.SuperLU
    fine_decomp: sla.SuperLU
    fine_spaces: Sequence[ElementFemSpace2D]
    fine_advection_operator: sp.csr_array
    coarse_advection_operator: sp.csr_array
    fine_linear_advection_operator: sp.coo_array
    coarse_linear_advection_operator: sp.coo_array
    projector_c2f: sp.csr_array
    projector_f2c: sp.csr_array
    coarse_padding: int
    fine_padding: int
    fine_offsets: npt.NDArray[np.uint32]
    fine_forcing: npt.NDArray[np.float64]
    convergence: ConvergenceSettings

    def __init__(
        self,
        system: KFormSystem,
        settings: VMSSettings,
        coarse_spaces: Sequence[ElementFemSpace2D],
        basis_cache: FemCache,
        mesh: Mesh,
        leaf_indices: Sequence[int],
        constrained_forms: Sequence[tuple[float, KFormUnknown]],
        strong_boundary_conditions: Sequence[BoundaryCondition2DSteady],
    ) -> None:
        self.convergence = settings.fine_scale_convergence
        self.unknown_forms = settings.symmetric_system.unknown_forms
        self.compiled_advection = CompiledSystem(settings.nonsymmetric_system)
        self.relaxation = settings.relaxation
        compiled_sym = CompiledSystem(settings.symmetric_system)

        fine_spaces: list[ElementFemSpace2D] = list()
        projectors_c2f: list[sp.coo_array] = list()
        projectors_f2c: list[sp.coo_array] = list()
        fine_advection_matrices: list[npt.NDArray[np.float64]] = list()
        coarse_advection_matrices: list[npt.NDArray[np.float64]] = list()
        coarse_matrices: list[npt.NDArray[np.float64]] = list()
        fine_matrices: list[npt.NDArray[np.float64]] = list()
        fine_forcing_vecs: list[npt.NDArray[np.float64]] = list()

        for fem_space in coarse_spaces:
            fine_space = ElementFemSpace2D(
                basis_cache.get_basis2d(
                    fem_space.order_1 + settings.order_increase,
                    fem_space.order_2 + settings.order_increase,
                    *fem_space.integration_orders,
                ),
                fem_space.corners,
            )
            if self.compiled_advection.nonlin_codes:
                fine_spaces.append(fine_space)

            projector_c2f = cast(
                sp.coo_array,
                sp.block_diag(
                    compute_element_projector(
                        self.unknown_forms, fem_space.basis_2d, fine_space
                    )
                ),
            )
            projectors_c2f.append(projector_c2f)

            projector_f2c = cast(
                sp.coo_array,
                sp.block_diag(
                    compute_element_projector(
                        self.unknown_forms, fine_space.basis_2d, fem_space
                    )
                ),
            )
            projectors_f2c.append(projector_f2c)

            fine_forcing = compute_element_rhs(system, fine_space)
            fine_forcing_vecs.append(fine_forcing)

            fine_advection_matrix = compute_element_matrix(
                self.unknown_forms, self.compiled_advection.linear_codes, fine_space
            )
            fine_advection_matrices.append(fine_advection_matrix)

            coarse_advection_matrix = compute_element_matrix(
                self.unknown_forms, self.compiled_advection.linear_codes, fem_space
            )
            coarse_advection_matrices.append(coarse_advection_matrix)

            fine_matrix = compute_element_matrix(
                self.unknown_forms, compiled_sym.lhs_full, fine_space
            )
            fine_matrices.append(fine_matrix)

            # coarse_matrix = projector_c2f.T @ fine_matrix @ projector_c2f
            coarse_matrix = compute_element_matrix(
                self.unknown_forms, compiled_sym.lhs_full, fem_space
            )
            coarse_matrices.append(coarse_matrix)

        self.projector_c2f = cast(
            sp.csr_array, sp.block_diag(projectors_c2f, format="csr")
        )
        self.projector_f2c = cast(
            sp.csr_array, sp.block_diag(projectors_f2c, format="csr")
        )

        self.fine_linear_advection_operator = cast(
            sp.coo_array, sp.block_diag(fine_advection_matrices, format="coo")
        )
        if self.compiled_advection.nonlin_codes is None:
            self.fine_advection_operator = self.fine_linear_advection_operator.tocsr()

        self.coarse_linear_advection_operator = cast(
            sp.coo_array, sp.block_diag(coarse_advection_matrices, format="coo")
        )
        if self.compiled_advection.nonlin_codes is None:
            self.coarse_advection_operator = self.coarse_linear_advection_operator.tocsr()

        self.fine_spaces = tuple(fine_spaces)
        del fine_advection_matrices

        mesh.uniform_p_change(+settings.order_increase, +settings.order_increase)
        self.fine_offsets = np.cumsum(
            [
                0,
                *(
                    system.unknown_forms.total_size(*mesh.get_leaf_orders(i_leaf))
                    for i_leaf in leaf_indices
                ),
            ]
        )

        fine_lag_mat, fine_lag_vec = add_system_constraints(
            system,
            mesh,
            basis_cache,
            constrained_forms,
            strong_boundary_conditions,
            leaf_indices,
            self.fine_offsets,
            fine_forcing_vecs,
        )
        mesh.uniform_p_change(-settings.order_increase, -settings.order_increase)

        self.fine_forcing = np.concatenate(fine_forcing_vecs, dtype=np.float64)

        if fine_lag_mat is not None:
            self.fine_sym_mat = sp.block_array(
                [
                    [sp.block_diag(fine_matrices), fine_lag_mat.T],
                    [fine_lag_mat, None],
                ],
                format="csc",
            )

            self.fine_decomp = sla.splu(self.fine_sym_mat)
        else:
            self.fine_sym_mat = sp.block_diag(fine_matrices, format="csc")
            self.fine_decomp = sla.splu(self.fine_sym_mat)

        self.fine_padding = fine_lag_vec.size
        del fine_matrices, fine_lag_vec, fine_lag_mat, fine_spaces

        coarse_offsets = np.cumsum(
            [
                0,
                *(
                    system.unknown_forms.total_size(*mesh.get_leaf_orders(i_leaf))
                    for i_leaf in leaf_indices
                ),
            ]
        )

        coarse_lag_mat, coarse_lag_vec = add_system_constraints(
            system,
            mesh,
            basis_cache,
            constrained_forms,
            strong_boundary_conditions,
            leaf_indices,
            coarse_offsets,
            None,
        )
        del coarse_offsets

        if coarse_lag_mat is not None:
            self.coarse_sym_mat = sp.block_array(
                [
                    [sp.block_diag(coarse_matrices), coarse_lag_mat.T],
                    [coarse_lag_mat, None],
                ],
                format="csc",
            )
            self.coarse_decomp = sla.splu(self.coarse_sym_mat)
        else:
            self.coarse_sym_mat = sp.block_diag(coarse_matrices, format="csc")
            self.coarse_decomp = sla.splu(self.coarse_sym_mat)

        self.coarse_padding = coarse_lag_vec.size
        del coarse_matrices, coarse_lag_vec, coarse_lag_mat

    def compute_unresolved_contributions(
        self,
        coarse_solution: npt.NDArray[np.float64],
        initial_guess: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """Compute unresolved scales given the residual vector on the fine mesh."""
        residual = self.fine_forcing - (
            self.fine_advection_operator
            @ self.projector_c2f
            @ coarse_solution  # [: coarse_solution.size - self.coarse_padding]
        )

        agr = self.fine_advection_operator @ self.fine_scale_greens_function(residual)
        if initial_guess is None:
            u = np.array(agr)
        else:
            u = np.array(initial_guess)

        del residual

        for _ in range(self.convergence.maximum_iterations):
            u_new = agr - self.fine_advection_operator @ self.fine_scale_greens_function(
                u
            )
            max_du = np.abs(u - u_new).max()
            max_u = np.abs(u_new).max()
            if self.relaxation == 1.0:
                u = u_new
            else:
                u *= 1 - self.relaxation
                u += self.relaxation * u_new
            if (
                max_u == 0  # Sometimes we are just that good!
                or max_du < max_u * self.convergence.relative_tolerance
                or max_du < self.convergence.absolute_tolerance
            ):
                break

        # return np.pad(u @ self.projector_c2f, (0, self.coarse_padding))
        return u

    def recover_unresolved(
        self,
        coarse_solution: npt.NDArray[np.float64],
        unresolved_contribution: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Recover unresolved scales from unresolved forcing."""
        residual = (
            self.fine_forcing
            - (
                self.fine_advection_operator
                @ self.projector_c2f
                @ coarse_solution[: coarse_solution.size - self.coarse_padding]
            )
            - unresolved_contribution
        )
        return self.fine_scale_greens_function(residual)

    def fine_results_to_coarse_dofs(
        self, x: npt.NDArray[np.float64], *, dual: bool
    ) -> npt.NDArray[np.float64]:
        """Project fine scale results to coarse scale DoFs and pad for constraints."""
        if dual:
            y = x @ self.projector_c2f
        else:
            y = self.projector_f2c @ x

        return np.pad(y, (0, self.coarse_padding))

    def update_nonlinear_advection(self, coarse_dofs: npt.NDArray[np.float64]) -> None:
        """Update non-linear advection terms if needed."""
        if self.compiled_advection.nonlin_codes is None:
            return
        assert len(self.fine_spaces) != 0

        fine_dofs = self.projector_c2f @ coarse_dofs[: -self.coarse_padding]

        nonlinear_matrices = [
            compute_element_matrix(
                self.unknown_forms,
                self.compiled_advection.nonlin_codes,
                fem_space,
                fine_dofs[self.fine_offsets[ie] : self.fine_offsets[ie + 1]],
            )
            for ie, fem_space in enumerate(self.fine_spaces)
        ]
        nonlin_mat = sp.block_diag(nonlinear_matrices, format="coo")
        self.fine_advection_operator = (
            self.fine_linear_advection_operator + nonlin_mat
        ).tocsr()

    def fine_scale_greens_function(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply the fine-scale Green's function to the vector."""
        result_fine = self.fine_decomp.solve(np.pad(x, (0, self.fine_padding)))[: x.size]
        coarse_sol = self.coarse_decomp.solve(
            np.pad(x @ self.projector_c2f, (0, self.coarse_padding))
        )
        result_coarse = (
            self.projector_c2f @ (coarse_sol[: coarse_sol.size - self.coarse_padding])
        )
        result = result_fine - result_coarse
        return result


def compute_linear_system(
    element_fem_spaces: Sequence[ElementFemSpace2D],
    element_offset: npt.NDArray[np.uint32],
    leaf_indices: Sequence[int],
    system: KFormSystem,
    compiled: CompiledSystem,
    mesh: Mesh,
    basis_cache: FemCache,
    constrained_forms: Sequence[tuple[float, KFormUnknown]],
    boundary_conditions: Sequence[BoundaryCondition2DSteady],
    initial_solution: list[npt.NDArray[np.float64]],
) -> tuple[
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    None | sp.csr_array,
    npt.NDArray[np.float64],
]:
    """Compute system matrix and vector.

    Parameters
    ----------
    dk : int
        Order increase from baseline.
    """
    linear_vectors = [compute_element_rhs(system, cache) for cache in element_fem_spaces]

    if initial_solution:
        linear_matrices = [
            compute_element_matrix(
                system.unknown_forms,
                compiled.linear_codes,
                element_space,
                element_solution,
            )
            for element_space, element_solution in zip(
                element_fem_spaces, initial_solution, strict=True
            )
        ]
    else:
        linear_matrices = [
            compute_element_matrix(
                system.unknown_forms, compiled.linear_codes, element_space
            )
            for element_space in element_fem_spaces
        ]

    # Generate constraints that force the specified for to have the (child element) sum
    # equal to a prescribed value.
    lagrange_mat, lagrange_vec = add_system_constraints(
        system,
        mesh,
        basis_cache,
        constrained_forms,
        boundary_conditions,
        leaf_indices,
        element_offset,
        linear_vectors,
    )

    return linear_vectors, linear_matrices, lagrange_mat, lagrange_vec
