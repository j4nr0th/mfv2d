"""Functions related to error calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from mfv2d._mfv2d import ElementFemSpace2D, IntegrationRule1D, Mesh, compute_legendre
from mfv2d.kform import KFormUnknown, UnknownOrderings
from mfv2d.mimetic2d import (
    bilinear_interpolate,
    compute_leaf_dof_counts,
    jacobian,
    reconstruct,
)
from mfv2d.progress import HistogramFormat


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


@dataclass(frozen=True)
class RefinementSettings:
    """Settings pertaining to refinement of a mesh."""

    required_forms: Sequence[KFormUnknown]
    """Forms that are needed by the error calculation function."""

    error_calculation_function: (
        ErrorCalculationFunctionFull | ErrorCalculationFunctionSimple
    )
    """Function called to calculate error estimate and h-refinement cost."""

    refinement_limit: RefinementLimit
    """Limit for mesh refinement."""

    h_refinement_ratio: float = 0.0
    """Ratio between element error and h-refinement cost where refinement can happen."""

    report_error_distribution: bool = False
    """Should the error distribution be reported."""

    report_order_distribution: bool = False
    """Should the order distribution be reported."""

    reconstruction_orders: tuple[int, int] | None = None
    """Order at which error should be reconstructed."""


def perform_mesh_refinement(
    mesh: Mesh,
    solution: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    dof_offsets: npt.NDArray[np.uint32],
    required_unknown_indices: Sequence[int],
    unknown_forms: Sequence[KFormUnknown],
    error_calculation_function: ErrorCalculationFunctionFull
    | ErrorCalculationFunctionSimple,
    h_refinement_ratio: float,
    refinement_limit: RefinementLimit,
    unknown_ordering: UnknownOrderings,
    report_error_distribution: bool,
    element_fem_spaces: Sequence[ElementFemSpace2D],
    reconstruction_orders: tuple[int | None, int | None],
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
    assert len(unknown_forms) >= len(required_unknown_indices)
    # assert len(unknown_forms) <= unknown_ordering.count
    mesh = mesh.copy()

    element_error = np.empty(mesh.leaf_count)
    href_cost = np.empty(mesh.leaf_count)

    indices = mesh.get_leaf_indices()
    recon_order_1, recon_order_2 = reconstruction_orders
    int_nodes: dict[int, npt.NDArray[np.double]] = dict()

    for i_leaf, idx in enumerate(indices):
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

    if report_error_distribution:
        error_log = np.log10(element_error)
        hist = HistogramFormat(5, 60, 5, label_format=lambda x: f"10^({x:.2g})")
        print("Error estimate distribution\n" + "=" * 60)
        print(hist.format(error_log))
        print("=" * 60)
        del error_log, hist

    error_order = np.flip(np.argsort(element_error))
    ordered_indices = indices[error_order]
    cost_fraction = href_cost / element_error

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
                solution.size * ucnt_limit.maximum_fraction, ucnt_limit.maximum_count
            )
            unknowns_added = 0
            for i_leaf, idx in zip(error_order, ordered_indices):
                if unknowns_added >= max_unknowns_increase:
                    break

                order_1, order_2 = mesh.get_leaf_orders(idx)
                original_unknowns = dof_offsets[i_leaf, -1]

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
                        4 * compute_leaf_dof_counts(*new_orders, unknown_ordering).sum()
                    )

                else:
                    order_1 += 1
                    order_2 += 1
                    mesh.set_leaf_orders(idx, order_1, order_2)

                    new_unknowns = compute_leaf_dof_counts(
                        order_1, order_2, unknown_ordering
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
                original_unknowns = dof_offsets[i_leaf, -1]

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
