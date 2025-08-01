"""Functions related to error calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from mfv2d._mfv2d import (
    ElementFemSpace2D,
    IntegrationRule1D,
    Mesh,
    compute_element_matrix,
    compute_element_projector,
    compute_element_vector,
    compute_legendre,
)
from mfv2d.boundary import BoundaryCondition2DSteady, _element_weak_boundary_condition
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KBoundaryProjection, KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import (
    FemCache,
    bilinear_interpolate,
    compute_leaf_dof_counts,
    find_surface_boundary_id_line,
    jacobian,
    reconstruct,
)
from mfv2d.progress import HistogramFormat
from mfv2d.solve_system import compute_element_rhs
from mfv2d.system import KFormSystem


def compute_legendre_coefficients(
    order_1: int,
    order_2: int,
    nodes_xi: npt.NDArray[np.float64],
    nodes_eta: npt.NDArray[np.float64],
    weighted_function: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Legendre coefficients from function values at integration nodes.

    Parameters
    ----------
    order_1 : int
        Order of the coefficients in the first direction.

    order_2 : int
        Order of the coefficients in the second direction.

    weighted_function : array
        Product of function, integration weight, and Jacobian for all integration points.

    Returns
    -------
    (order_1 + 1, order_2 + 1) array
        Array of coefficients for Legendre basis.
    """
    leg1 = compute_legendre(order_1, nodes_xi.flatten())
    leg2 = compute_legendre(order_2, nodes_eta.flatten())

    rleg = np.sum(
        weighted_function[None, None, ...]
        * (leg1[None, :, None, :] * leg2[:, None, :, None]),
        axis=(-2, -1),
    )
    n1 = np.arange(order_1 + 1)
    n2 = np.arange(order_2 + 1)
    norms1 = 2 / (2 * n1 + 1)
    norms2 = 2 / (2 * n2 + 1)
    rleg /= norms1[None, :] * norms2[:, None]

    return rleg


class ErrorCalculationFunctionFull(Protocol):
    """Type that can compute error."""

    def __call__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
        order_1: int,
        order_2: int,
        xi: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        **kwargs: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute the error.

        Parameters
        ----------
        x : array
            x-coordinates of integration points.

        y : array
            y-coordinates of integration points.

        w : array
            Integration weights at the specified points, multiplied by the Jacobian.

        **kwargs : array
            Values of desired forms at specified positions.

        Returns
        -------
        float
            Error measure of the element. Not negative.

        float
            Cost of h-refinement.
        """
        ...


class ErrorCalculationFunctionSimple(Protocol):
    """Type that can compute error."""

    def __call__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
        **kwargs: npt.NDArray[np.float64] | int,
    ) -> tuple[float, float]:
        """Compute the error.

        Parameters
        ----------
        x : array
            x-coordinates of integration points.

        y : array
            y-coordinates of integration points.

        w : array
            Integration weights at the specified points, multiplied by the Jacobian.

        **kwargs : array
            Values of desired forms at specified positions.

        Returns
        -------
        float
            Error measure of the element. Not negative.

        float
            Cost of h-refinement.
        """
        ...


@dataclass(frozen=True)
class RefinementLimitUnknownCount:
    """Limit refinement based on change in number of degrees of freedom."""

    maximum_fraction: float
    maximum_count: int


@dataclass(frozen=True)
class RefinementLimitElementCount:
    """Limit refinement based on the number of elements."""

    maximum_fraction: float
    maximum_count: int


@dataclass(frozen=True)
class RefinementLimitErrorValue:
    """Limit refinement based on value of the errror."""

    minimum_fraction: float
    minimum_value: float


RefinementLimit = (
    RefinementLimitUnknownCount | RefinementLimitElementCount | RefinementLimitErrorValue
)

CustomErrorFunction = ErrorCalculationFunctionFull | ErrorCalculationFunctionSimple


@dataclass(frozen=True)
class ErrorEstimateCustom:
    """Settings for refinement that is to be done with user-defined function."""

    required_forms: Sequence[KFormUnknown]
    """Forms that are needed by the error calculation function."""

    error_calculation_function: CustomErrorFunction
    """Function called to calculate error estimate and h-refinement cost."""

    reconstruction_orders: tuple[int, int] | None = None
    """Order at which error should be reconstructed."""


@dataclass(frozen=True)
class ErrorEstimateLocalInverse:
    """Settings for refinement that is based on local inverse."""

    strong_forms: Sequence[KFormUnknown]
    """Forms for which the boundary conditions on each element must be
    given strongly."""

    order_increase: int
    """Order at which residual should be reconstructed prior to inversion."""

    target_form: KFormUnknown
    """Error of this form is used as a guid for refinement."""

    reconstruction_orders: tuple[int, int] | None = None
    """Order at which error should be reconstructed."""


ErrorEstimate = ErrorEstimateCustom | ErrorEstimateLocalInverse


@dataclass(frozen=True)
class RefinementSettings:
    """Settings pertaining to refinement of a mesh."""

    error_estimate: ErrorEstimate
    """How the error ought to be estimated."""

    refinement_limit: RefinementLimit
    """Limit for mesh refinement."""

    h_refinement_ratio: float = 0.0
    """Ratio between element error and h-refinement cost where refinement can happen."""

    report_error_distribution: bool = False
    """Should the error distribution be reported."""

    report_order_distribution: bool = False
    """Should the order distribution be reported."""


def perform_mesh_refinement(
    mesh: Mesh,
    solution: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    dof_offsets: npt.NDArray[np.uint32],
    system: KFormSystem,
    error_estimator: ErrorEstimate,
    h_refinement_ratio: float,
    refinement_limit: RefinementLimit,
    report_error_distribution: bool,
    element_fem_spaces: Sequence[ElementFemSpace2D],
    boundary_conditions: Sequence[BoundaryCondition2DSteady],
    basis_cache: FemCache,
    leaf_order_mapping: dict[int, int],
) -> Mesh:
    """Perform a round of mesh refinement.

    Parameters
    ----------
    mesh : Mesh
        Mesh on which to perform refinement on.

    settings : RefinementSettings
        Specifications of how the refinement shoud be performed.

    solution : array
        Solution for degrees of freedom.

    element_offsets : array
        Array with offsets for the beginning of
    """
    indices = mesh.get_leaf_indices()

    match error_estimator:
        case ErrorEstimateCustom():
            element_error, href_cost = error_estimate_with_custom_estimator(
                len(indices),
                solution,
                element_offsets,
                dof_offsets,
                error_estimator.required_forms,
                system.unknown_forms,
                error_estimator.error_calculation_function,
                element_fem_spaces,
                error_estimator.reconstruction_orders[0]
                if error_estimator.reconstruction_orders is not None
                else None,
                error_estimator.reconstruction_orders[1]
                if error_estimator.reconstruction_orders is not None
                else None,
            )

        case ErrorEstimateLocalInverse():
            element_error, href_cost = error_estimate_with_local_inversion(
                mesh,
                solution,
                element_offsets,
                boundary_conditions,
                element_fem_spaces,
                error_estimator.order_increase,
                basis_cache,
                system,
                CompiledSystem(system),
                [form.order for form in system.unknown_forms],
                leaf_order_mapping,
                error_estimator.target_form,
                error_estimator.reconstruction_orders[0]
                if error_estimator.reconstruction_orders is not None
                else None,
                error_estimator.reconstruction_orders[1]
                if error_estimator.reconstruction_orders is not None
                else None,
            )

        case _:
            raise TypeError(
                f"Invalid type for error estimator {type(error_estimator).__name__}"
            )

    if report_error_distribution:
        error_log = np.log10(element_error)
        hist = HistogramFormat(5, 60, 5, label_format=lambda x: f"10^({x:.2g})")
        print("Error estimate distribution\n" + "=" * 60)
        print(hist.format(error_log))
        print("=" * 60)
        del error_log, hist

    return refine_mesh_based_on_error(
        mesh,
        solution.size,
        h_refinement_ratio,
        refinement_limit,
        [form.order for form in system.unknown_forms],
        indices,
        element_error,
        href_cost,
    )


def refine_mesh_based_on_error(
    mesh: Mesh,
    total_unknowns: int,
    h_refinement_ratio: float,
    refinement_limit: RefinementLimit,
    unknown_form_orders: Sequence[UnknownFormOrder],
    leaf_indices: npt.NDArray[np.uint32],
    element_error: npt.NDArray[np.float64],
    href_cost: npt.NDArray[np.float64],
) -> Mesh:
    """Refine the given mesh based on given element error and h-refinement cost."""
    error_order = np.flip(np.argsort(element_error))
    ordered_indices = leaf_indices[error_order]
    cost_fraction = href_cost / element_error
    mesh = mesh.copy()

    match refinement_limit:
        case RefinementLimitElementCount() as ecnt_limit:
            max_element_refinements = min(
                mesh.leaf_count * ecnt_limit.maximum_fraction, ecnt_limit.maximum_count
            )
            elements_refined = 0
            for i_leaf, idx in zip(error_order, ordered_indices):
                if elements_refined >= max_element_refinements:
                    break

                order_1, order_2 = mesh.get_leaf_orders(idx)

                if (
                    cost_fraction[i_leaf] <= h_refinement_ratio
                    and (order_1 > 1)
                    and (order_2 > 1)
                ):
                    new_orders = (order_1 // 2, order_2 // 2)
                    mesh.split_element(
                        idx, new_orders, new_orders, new_orders, new_orders
                    )
                else:
                    order_1 += 1
                    order_2 += 1
                    mesh.set_leaf_orders(idx, order_1, order_2)

                elements_refined += 1

        case RefinementLimitUnknownCount() as ucnt_limit:
            max_unknowns_increase = min(
                total_unknowns * ucnt_limit.maximum_fraction, ucnt_limit.maximum_count
            )
            unknowns_added = 0
            for i_leaf, idx in zip(error_order, ordered_indices):
                if unknowns_added >= max_unknowns_increase:
                    break

                order_1, order_2 = mesh.get_leaf_orders(idx)
                # original_unknowns = dof_offsets[i_leaf, -1]
                original_unknowns = sum(
                    order.full_unknown_count(order_1, order_2)
                    for order in unknown_form_orders
                )

                if (
                    cost_fraction[i_leaf] <= h_refinement_ratio
                    and (order_1 > 1)
                    and (order_2 > 1)
                ):
                    new_orders = (order_1 // 2, order_2 // 2)
                    mesh.split_element(
                        idx, new_orders, new_orders, new_orders, new_orders
                    )
                    new_unknowns = (
                        4
                        * compute_leaf_dof_counts(*new_orders, unknown_form_orders).sum()
                    )

                else:
                    order_1 += 1
                    order_2 += 1
                    mesh.set_leaf_orders(idx, order_1, order_2)

                    new_unknowns = compute_leaf_dof_counts(
                        order_1, order_2, unknown_form_orders
                    ).sum()

                unknowns_added += new_unknowns - original_unknowns

        case RefinementLimitErrorValue() as eval_limit:
            total_error = np.sum(element_error)
            minimum_error_required = min(
                total_error * eval_limit.minimum_fraction, eval_limit.minimum_value
            )

            for i_leaf, idx in zip(error_order, ordered_indices):
                if element_error[i_leaf] < minimum_error_required:
                    break

                order_1, order_2 = mesh.get_leaf_orders(idx)

                if (
                    cost_fraction[i_leaf] <= h_refinement_ratio
                    and (order_1 > 1)
                    and (order_2 > 1)
                ):
                    new_orders = (order_1 // 2, order_2 // 2)
                    mesh.split_element(
                        idx, new_orders, new_orders, new_orders, new_orders
                    )

                else:
                    order_1 += 1
                    order_2 += 1
                    mesh.set_leaf_orders(idx, order_1, order_2)

        case _:
            raise TypeError(
                f"Invalid type for refinement limit: {type(refinement_limit).__name__}"
            )

    return mesh


def error_estimate_with_custom_estimator(
    leaf_count: int,
    solution: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    dof_offsets: npt.NDArray[np.uint32],
    required_unknowns: Sequence[KFormUnknown],
    unknown_forms: Sequence[KFormUnknown],
    error_calculation_function: CustomErrorFunction,
    element_fem_spaces: Sequence[ElementFemSpace2D],
    recon_order_1: int | None,
    recon_order_2: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute element error estimates using a custom, user-supplied function.

    Parameters
    ----------
    leaf_count : int
        Number of leaves in the mesh.

    solution : array
        Full solution vector.

    element_offsets : array
        Array of offsets to the beginning of each element for the solution vector.

    dof_offsets : array
        Offsets of each unknown within each element.

    required_unknowns : Sequence of KFormUnknown
        Unknowns which are needed by the user function.

    unknown_forms : Sequence of KFormUnknown
        Unknown forms in the order they appear in the system.

    error_calculation_function : CustomErrorFunction
        Error function that is used to compute the error estimates.

    elemenebt_fem_spaces : Sequence of ElementFemSpace2D
        Element FEM spaces.

    recon_order_1 : int or None
        Order at which all element should be reconstructed. If not present, the element
        integration order is used.


    recon_order_2 : int or None
        Order at which all element should be reconstructed. If not present, the element
        integration order is used.

    Returns
    -------
    element_error : array
        Array with error estimates for each element.

    href_cost : array
        Array with estimates of increase in error due to h-refinement.
    """
    required_unknown_indices = [
        unknown_forms.index(unknown) for unknown in required_unknowns
    ]
    int_nodes: dict[int, npt.NDArray[np.double]] = dict()
    element_error = np.empty(leaf_count)
    href_cost = np.empty(leaf_count)
    for i_leaf in range(leaf_count):
        element_solution = solution[element_offsets[i_leaf] : element_offsets[i_leaf + 1]]
        offsets = dof_offsets[i_leaf]

        element_space = element_fem_spaces[i_leaf]
        corners = element_space.corners
        basis = element_space.basis_2d
        order_1 = basis.basis_xi.order
        order_2 = basis.basis_eta.order

        if recon_order_1 is None:
            nodes_xi = basis.basis_xi.rule.nodes[None, :]
            int_nodes[order_1] = nodes_xi
        else:
            if recon_order_1 not in int_nodes:
                nodes_xi = IntegrationRule1D(recon_order_1).nodes[None, :]
                int_nodes[recon_order_1] = nodes_xi
            else:
                nodes_xi = int_nodes[recon_order_1]

        if recon_order_2 is None:
            nodes_eta = basis.basis_eta.rule.nodes[:, None]
            int_nodes[order_2] = nodes_eta
        else:
            if recon_order_2 not in int_nodes:
                nodes_eta = IntegrationRule1D(recon_order_2).nodes[:, None]
                int_nodes[recon_order_2] = nodes_eta
            else:
                nodes_eta = int_nodes[recon_order_2]

        x = bilinear_interpolate(corners[:, 0], nodes_xi, nodes_eta)
        y = bilinear_interpolate(corners[:, 1], nodes_xi, nodes_eta)
        form_vals: dict[str, npt.NDArray[np.float64]] = dict()
        for form_idx in required_unknown_indices:
            form = unknown_forms[form_idx]
            form_vals[form.label] = reconstruct(
                element_space,
                form.order,
                element_solution[offsets[form_idx] : offsets[form_idx + 1]],
                nodes_xi,
                nodes_eta,
            )
        jac = jacobian(corners, nodes_xi, nodes_eta)
        det = (jac[0][0] * jac[1][1]) - (jac[0][1] * jac[1][0])
        elem_vals = error_calculation_function(
            x=x,
            y=y,
            w=basis.basis_xi.rule.weights[None, :]
            * basis.basis_eta.rule.weights[:, None]
            * det,
            order_1=order_1,
            order_2=order_2,
            xi=np.astype(nodes_xi, np.float64, copy=False),
            eta=np.astype(nodes_eta, np.float64, copy=False),
            **form_vals,
        )
        if elem_vals[0] < 0:
            raise ValueError(
                "Error calculation function returned a negative error estimate."
            )

        element_error[i_leaf], href_cost[i_leaf] = elem_vals
    return element_error, href_cost


def error_estimate_with_local_inversion(
    mesh: Mesh,
    solution: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    boundary_conditions: Sequence[BoundaryCondition2DSteady],
    element_fem_spaces: Sequence[ElementFemSpace2D],
    recon_order: int,
    basis_cache: FemCache,
    system: KFormSystem,
    compiled: CompiledSystem,
    unknown_ordering: Sequence[UnknownFormOrder],
    leaf_order_mapping: dict[int, int],
    unknown_target: KFormUnknown,
    recon_order_1: int | None,
    recon_order_2: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute element error estimates using a local inversion.

    Returns
    -------
    element_error : array
        Array with error estimates for each element.

    href_cost : array
        Array with estimates of increase in error due to h-refinement.
    """
    assert unknown_target in system.unknown_forms

    element_error = np.empty(mesh.leaf_count)
    href_cost = np.empty(mesh.leaf_count)
    residuals: list[npt.NDArray[np.float64]] = list()
    projected_solution: list[npt.NDArray[np.float64]] = list()
    higher_order_spaces: list[ElementFemSpace2D] = list()

    # Compute the residual
    for i_leaf in range(mesh.leaf_count):
        element_solution = solution[element_offsets[i_leaf] : element_offsets[i_leaf + 1]]
        element_space = element_fem_spaces[i_leaf]

        corners = element_space.corners

        basis = element_space.basis_2d
        order_1 = basis.basis_xi.order
        order_2 = basis.basis_eta.order
        higher_basis = basis_cache.get_basis2d(
            order_1 + recon_order, order_2 + recon_order
        )

        higher_space = ElementFemSpace2D(higher_basis, corners)
        higher_order_spaces.append(higher_space)

        fine_rhs = compute_element_rhs(system, higher_space)
        coarse_forcing = compute_element_vector(
            unknown_ordering,
            compiled.lhs_full,
            compiled.vector_field_specs,
            element_space,
            element_solution,
        )
        if compiled.rhs_codes:
            coarse_forcing -= compute_element_vector(
                unknown_ordering,
                compiled.rhs_codes,
                compiled.vector_field_specs,
                element_space,
                element_solution,
            )
        projector = compute_element_projector(
            unknown_ordering,
            higher_space.corners,
            element_space.basis_2d,
            higher_space.basis_2d,
        )

        projected_solution.append(projector @ element_solution)

        fine_forcing = projector @ coarse_forcing
        del projector

        residuals.append(fine_rhs - fine_forcing)
        del fine_forcing, fine_rhs

        # compute_element_matrix, compute_element_vector, compute_element_projector
        # element_error[i_leaf], href_cost[i_leaf] = elem_vals

    # Right sides are still missing natural BC contributions
    # without these, there will be way to much residuals on the
    # boundaries.

    boundary_indices = mesh.boundary_indices
    for equation in system.equations:
        # Check if corresponding equation has any weak terms
        form = equation.weight.base_form
        boundary_terms = [
            (v, f)
            for v, f in equation.right.explicit_terms
            if (type(f) is KBoundaryProjection and f.func is not None)
        ]

        if not boundary_terms:
            # No boundary terms, skip it
            continue

        form_index = system.unknown_forms.index(form)

        strong_indices = [bc.indices for bc in boundary_conditions if bc.form == form]
        if strong_indices:
            skip_indices = np.unique(np.concatenate(strong_indices))
        else:
            skip_indices = np.zeros(0, np.uint32)
        del strong_indices

        weak_indices = np.isin(boundary_indices, skip_indices)

        line_index: int
        primal = mesh.primal
        dual = mesh.dual
        for line_index in weak_indices:
            dual_line = dual.get_line(line_index + 1)
            if dual_line.begin:
                surf_id = dual_line.begin
                assert not dual_line.end
            elif dual_line.end:
                surf_id = dual_line.end
                assert not dual_line.begin
            else:
                raise ValueError(
                    "Incorrect boundary indices for the mesh - dual line has no valid"
                    " points."
                )

            primal_surface = primal.get_surface(surf_id)
            side = find_surface_boundary_id_line(primal_surface, line_index)
            bc_data = _element_weak_boundary_condition(
                mesh,
                surf_id.index,
                side,
                unknown_ordering,
                form_index,
                boundary_terms,
                basis_cache,
            )
            for bc in bc_data:
                i_element = leaf_order_mapping[bc.i_e]
                residuals[i_element][bc.dofs] += bc.coeffs

    # Residual is now complete. Now error can be estimated based on the local
    # inverse of the residual

    unknown_index = system.unknown_forms.index(unknown_target)
    for idx_leaf, (fem_space, residual, element_solution, coarse_space) in enumerate(
        zip(
            higher_order_spaces,
            residuals,
            projected_solution,
            element_fem_spaces,
            strict=True,
        )
    ):
        local_lhs = compute_element_matrix(
            unknown_ordering,
            compiled.lhs_full,
            compiled.vector_field_specs,
            fem_space,
            element_solution,
        )
        local_error_dofs = np.linalg.solve(local_lhs, residual)

        offset = sum(
            form.order.full_unknown_count(
                fem_space.basis_xi.order, fem_space.basis_eta.order
            )
            for form in system.unknown_forms[:unknown_index]
        )
        count = unknown_target.order.full_unknown_count(
            fem_space.basis_xi.order, fem_space.basis_eta.order
        )
        target_dofs = local_error_dofs[offset : offset + count]

        rule_1 = (
            fem_space.basis_xi.rule
            if recon_order_1 is None
            else basis_cache.get_integration_rule(recon_order_1)
        )
        rule_2 = (
            fem_space.basis_eta.rule
            if recon_order_2 is None
            else basis_cache.get_integration_rule(recon_order_2)
        )

        reconstructed_error = reconstruct(
            fem_space,
            unknown_target.order,
            target_dofs,
            rule_1.nodes[None, :],
            rule_2.nodes[:, None],
        )

        jac = jacobian(fem_space.corners, rule_1.nodes[None, :], rule_2.nodes[:, None])
        det = (jac[0][0] * jac[1][1]) - (jac[0][1] * jac[1][0])
        weights = rule_1.weights[None, :] * rule_2.weights[:, None] * det
        weighted_error = reconstructed_error * weights
        error_coefficients = compute_legendre_coefficients(
            fem_space.basis_xi.order,
            fem_space.basis_eta.order,
            np.astype(rule_1.nodes, np.float64, copy=False),
            np.astype(rule_2.nodes, np.float64, copy=False),
            weighted_error,
        )

        reconstructed_form = reconstruct(
            fem_space,
            unknown_target.order,
            element_solution[offset : offset + count],
            rule_1.nodes[None, :],
            rule_2.nodes[:, None],
        )
        form_coefficients = compute_legendre_coefficients(
            fem_space.basis_xi.order,
            fem_space.basis_eta.order,
            np.astype(rule_1.nodes, np.float64, copy=False),
            np.astype(rule_2.nodes, np.float64, copy=False),
            reconstructed_form * weights,
        )

        norm_1 = 2 / (2 * np.arange(fem_space.basis_xi.order + 1) + 1)
        norm_2 = 2 / (2 * np.arange(fem_space.basis_eta.order + 1) + 1)
        norm_2d = norm_1[None, :] * norm_2[:, None]

        measure = (
            form_coefficients * (form_coefficients + 2 * error_coefficients) / norm_2d
        )

        limit_1 = (coarse_space.basis_xi.order + 1) // 2
        limit_2 = (coarse_space.basis_eta.order + 1) // 2

        h_cost = np.sum(
            measure[limit_2:, limit_1:]
            + measure[:limit_2, limit_1:]
            + measure[limit_2:, :limit_1]
        )
        error_l2 = np.sum(weighted_error * reconstructed_error)

        href_cost[idx_leaf] = h_cost
        element_error[idx_leaf] = error_l2

    return element_error, href_cost
