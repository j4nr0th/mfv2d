"""Functions related to error calculations."""

import numpy as np
import numpy.typing as npt
import pyvista as pv

from mfv2d._mfv2d import (
    IntegrationRule1D,
    compute_legendre,
    legendre_l2_to_h1_coefficients,
)
from mfv2d.mimetic2d import ElementLeaf2D, bilinear_interpolate


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
    e: ElementLeaf2D, mesh: pv.UnstructuredGrid, key: str, order_1: int, order_2: int
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

    corners = np.array(
        (e.bottom_left, e.bottom_right, e.top_right, e.top_left), np.float64
    )
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
    sampled = points.sample(mesh)

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
