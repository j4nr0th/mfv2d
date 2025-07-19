"""Functions related to error calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pyvista as pv

from mfv2d._mfv2d import (
    IntegrationRule1D,
    Mesh,
    compute_legendre,
    legendre_l2_to_h1_coefficients,
)
from mfv2d.kform import KFormUnknown, UnknownOrderings
from mfv2d.mimetic2d import (
    FemCache,
    bilinear_interpolate,
    compute_leaf_dof_counts,
    reconstruct,
)


def _compute_legendre_coefficients(
    order_1: int,
    order_2: int,
    rule_1: IntegrationRule1D,
    rule_2: IntegrationRule1D,
    func_vals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Legendre coefficients from function values at integration nodes.

    Parameters
    ----------
    order_1 : int
        Order of the coefficients in the first direction.

    order_2 : int
        Order of the coefficients in the second direction.

    rule_1 : IntegrationRule1D
        Integration rule to use in the first direction.

    rule_2 : IntegrationRule1D
        Integration rule to use in the second direction.

    func_vals : array
        Array of function values at the positions of integration
        rules.

    Returns
    -------
    (order_1 + 1, order_2 + 1) array
        Array of coefficients for Legendre basis.
    """
    leg1 = compute_legendre(order_1, rule_1.nodes)
    leg2 = compute_legendre(order_2, rule_2.nodes)
    wleg1 = leg1 * rule_1.weights[None, ...]
    wleg2 = leg2 * rule_2.weights[None, ...]

    rleg = np.sum(
        func_vals[None, None, ...] * (wleg1[None, :, None, :] * wleg2[:, None, :, None]),
        axis=(-2, -1),
    )
    n1 = np.arange(order_1 + 1)
    n2 = np.arange(order_2 + 1)
    norms1 = 2 / (2 * n1 + 1)
    norms2 = 2 / (2 * n2 + 1)
    rleg /= norms1[None, :] * norms2[:, None]

    return rleg


def compute_element_field_legendre_coefficients(
    ie: int, mesh: Mesh, grid: pv.UnstructuredGrid, key: str, order_1: int, order_2: int
) -> npt.NDArray[np.float64]:
    """Compute Legendre coefficients of a field on the element.

    Parameters
    ----------
    e : ElementLeaf2D
        Element on which the error coefficients

    mesh : pyvista.UnstructuredGrid
        Mesh from which the field is computed from.

    key : str
        Key of the field which is to be converted.

    order_1 : int
        Order of the coefficients in the first direction.

    order_2 : int
        Order of the coefficients in the second direction.

    Returns
    -------
    (order_1 + 1, order_2 + 1) array
        Array of coefficients for Legendre basis.
    """
    # These rules are GLL, so they need an extra order for proper integration.
    rule_1 = IntegrationRule1D(order_1 + 1)
    rule_2 = IntegrationRule1D(order_2 + 1)

    corners = mesh.get_leaf_corners(ie)
    nodes_x = bilinear_interpolate(
        corners[:, 0], rule_1.nodes[None, :], rule_2.nodes[:, None]
    )
    nodes_y = bilinear_interpolate(
        corners[:, 1], rule_1.nodes[None, :], rule_2.nodes[:, None]
    )

    points = pv.PolyData(
        np.stack(
            (nodes_x.flatten(), nodes_y.flatten(), np.zeros_like(nodes_x).flatten()),
            axis=-1,
        )
    )
    sampled = points.sample(grid)

    func_vals = np.asarray(sampled[key], np.float64).reshape(nodes_x.shape)

    return _compute_legendre_coefficients(order_1, order_2, rule_1, rule_2, func_vals)


def legendre_l2_to_h1_2d(c: npt.ArrayLike, /) -> npt.NDArray[np.float64]:
    """Convert Legendre coefficients to those defined by the H1 seminorm."""
    coeffs = np.asarray(c, np.float64)
    if coeffs.ndim != 2:
        raise ValueError("Coefficients must be a 2D array!")

    out = np.empty_like(coeffs)

    # Throught the first dimension
    for i in range(coeffs.shape[0]):
        out[i, :] = legendre_l2_to_h1_coefficients(coeffs[i, :])

    # Throught the second dimension
    for j in range(out.shape[1]):
        out[:, j] = legendre_l2_to_h1_coefficients(out[:, j])

    return out


class ErrorCalculationFunction(Protocol):
    """Type that can compute error."""

    def __call__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute the error.

        Parameters
        ----------
        x : array
            x-coordinates of integration points.

        y : array
            y-coordinates of integration points.

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

    required_forms: tuple[KFormUnknown]
    """Forms that are needed by the error calculation function."""

    error_calculation_function: ErrorCalculationFunction
    """Function called to calculate error estimate and h-refinement cost."""

    h_refinement_ratio: float
    """Ratio between element error and h-refinement cost where refinement can happen."""

    refinement_limit: RefinementLimit
    """Limit for mesh refinement."""

    def __init__(
        self,
        required_forms: Sequence[KFormUnknown],
        error_calculation_function: ErrorCalculationFunction,
        h_refinement_ratio: float,
        refinement_limit: RefinementLimit,
    ) -> None:
        object.__setattr__(self, "required_forms", tuple(required_forms))
        object.__setattr__(self, "error_calculation_function", error_calculation_function)
        object.__setattr__(self, "h_refinement_ratio", h_refinement_ratio)
        object.__setattr__(self, "refinement_limit", refinement_limit)


def perform_mesh_refinement(
    mesh: Mesh,
    solution: npt.NDArray[np.float64],
    element_offsets: npt.NDArray[np.uint32],
    dof_offsets: npt.NDArray[np.uint32],
    required_unknown_indices: Sequence[int],
    unknown_forms: Sequence[KFormUnknown],
    error_calculation_function: ErrorCalculationFunction,
    h_refinement_ratio: float,
    refinement_limit: RefinementLimit,
    unknown_ordering: UnknownOrderings,
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
    basis_cache = FemCache(2)

    indices = mesh.get_leaf_indices()

    for i_leaf, idx in enumerate(indices):
        element_solution = solution[element_offsets[i_leaf] : element_offsets[i_leaf + 1]]
        offsets = dof_offsets[i_leaf]

        corners = mesh.get_leaf_corners(idx)
        order_1, order_2 = mesh.get_leaf_orders(idx)
        basis = basis_cache.get_basis2d(order_1, order_2)

        nodes_xi = basis.basis_xi.node[None, :]
        nodes_eta = basis.basis_eta.node[:, None]
        x = bilinear_interpolate(corners[:, 0], nodes_xi, nodes_eta)
        y = bilinear_interpolate(corners[:, 1], nodes_xi, nodes_eta)
        form_vals: dict[str, npt.NDArray[np.float64]] = dict()
        for form_idx in required_unknown_indices:
            form = unknown_forms[form_idx]
            form_vals[form.label] = reconstruct(
                corners,
                form.order,
                element_solution[offsets[form_idx] : offsets[form_idx + 1]],
                nodes_xi,
                nodes_eta,
                basis,
            )
        elem_vals = error_calculation_function(x, y, *form_vals)
        if elem_vals[0] < 0:
            raise ValueError(
                "Error calculation function returned a negative error estimate."
            )

        element_error[i_leaf], href_cost[i_leaf] = elem_vals

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
                    and (order_1 > 3)
                    and (order_2 > 3)
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
                    and (order_1 > 3)
                    and (order_2 > 3)
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
                    and (order_1 > 3)
                    and (order_2 > 3)
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
