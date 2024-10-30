"""Spline auxiliary functions and classes."""

from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D, Spline1Di

__all__ = [
    "SplineBC",
    "element_interpolating_splinei",
]


@dataclass(init=False, frozen=True)
class SplineBC:
    r"""Represents a boundary condition for a spline.

    This represents the boundary condition expressed as:

    :math:`\sum\limits_{n=0}^N k_n \frac{d^n}{{dt}^n}(y) = v`,

    where the values :math:`k_n` correspond to entries in ``coeffs[n]``.

    Parameters
    ----------
    coeffs : (N,) array_like
        Represents coefficients of the spline's derivatives.
    value : float
        Value of the weighted sum for the boundary condition.
    """

    coefficiens: npt.NDArray[np.float64]
    value: float

    def __init__(self, coeffs: npt.ArrayLike, value: float) -> None:
        k = np.array(coeffs, np.float64)
        if len(k.shape) != 1:
            raise ValueError("Coefficient array must be 1D")
        v = float(value)
        if np.all(k == 0.0):
            raise ValueError("All coefficients were zero.")
        object.__setattr__(self, "coefficiens", k)
        object.__setattr__(self, "value", v)


def _element_polynomial_coefficients(n: int) -> npt.NDArray[np.float64]:
    """Return coefficients for element polynomials.

    Parameters
    ----------
    n : int
        Order of polynomials. Number of terms will be one higher.

    Returns
    -------
    (n + 1, n + 1) array
        Matrix, where each row has the coefficients of the polynomial
        basis in from the constant term, to the :math:`t^n` term.
    """
    assert n & 1 == 0
    a = np.zeros((n + 1, n + 1), np.float64)
    # First row is anti-derivative
    k = np.arange(n + 1)
    a[0, :] = 1 / (k + 1)

    f = 1.0
    k = np.ones(n + 1)
    for r in range(n // 2):
        a[2 * r + 1, r] = f
        a[2 * r + 2, :] = k
        f *= r + 1
        k[r:] *= np.arange(n + 1 - r)
    return np.linalg.inv(a).T


def _element_interpolating_basis(n: int) -> tuple[Polynomial1D]:
    """Create interpolating basis for element spline with degree n.

    Parameters
    ----------
    n : int
        Order of polynomials to use. Number of terms will be one more.

    Returns
    -------
    tuple of Polynomial1D
        Tuple of basis polynomials for element-centered interpolation.
    """
    assert n & 1 == 0
    m = _element_polynomial_coefficients(n)

    # re-order rows of m
    def _permute_gen(n: int) -> Generator[int, None, None]:
        yield 0
        for i in range(n // 2):
            yield 2 * i + 1
        for i in range(n // 2):
            yield 2 * i + 2

    rows = _permute_gen(n)
    return tuple(Polynomial1D(m[i, :]) for i in rows)


def _element_interpolation_system(
    n: int,
    basis: Iterable[Polynomial1D],
    avg: npt.ArrayLike,
    bc_left: Iterable[SplineBC],
    bc_right: Iterable[SplineBC],
) -> npt.NDArray[np.float64]:
    """Create interpolation matrix for natural element.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    avg : (M,) array_like
        Averages on the elements which are to be interpolated.
    bc_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    ((M + 1) * n//2, (M + 1) * n//2) array
        System matrix which gives coefficients required to construct the spine.
    ((M + 1) * n//2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 0
    values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.stack(values, axis=0)
    averages = np.array(avg, np.float64)
    assert len(averages.shape) == 1
    nelem = int(averages.shape[0])
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)

    n_node = n // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = np.zeros(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n)
        if bc.coefficiens.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f"coefficients (got {bc.coefficiens.shape} when expecting"
                f" {(n,)})."
            )
        for j in range(n):
            m[p, inode] += bc.coefficiens[j] * v[j, 1:, 0]
            k[p] -= bc.coefficiens[j] * v[j, 0, 0] * averages[0]
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = n_node * i + np.arange(n)
        iright = n_node * (i + 1) + np.arange(n)
        for j in range(n_node, n):
            m[p, ileft] -= v[j, 1:, 1]
            m[p, iright] += v[j, 1:, 0]
            k[p] = v[j, 0, 1] * avg[i] - v[j, 0, 0] * avg[i + 1]
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        inode = np.arange(n) + (nelem - 1) * n_node
        if bc.coefficiens.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficiens.shape} when expecting"
                f" {(n,)})."
            )
        for j in range(n):
            m[p, inode] += bc.coefficiens[j] * v[j, 1:, 1]
            k[p] -= bc.coefficiens[j] * v[j, 0, 1] * averages[-1]
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m, k


def element_interpolating_splinei(
    n: int,
    avgs: npt.ArrayLike,
    bcs_left: Iterable[SplineBC],
    bcs_right: Iterable[SplineBC],
) -> Spline1Di:
    """Create interpolating spline, which has specified averages.

    Parameters
    ----------
    n : int
        Order of the spline. Must be even, otherwise a ``ValueError`` will be raised.
    avgs : (N,) array_like
        Averages of elements to hit.
    bcs_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bcs_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    Spline1Di
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    if (n & 1) != 0:
        raise ValueError(f"Spline order must be even (instead it was {n}).")
    averages = np.array(avgs, np.float64)
    if len(averages.shape) != 1:
        raise ValueError(
            f"Averages should be a 1D array, instead they have the shape {averages.shape}"
        )
    basis = _element_interpolating_basis(n)
    m, k = _element_interpolation_system(n, basis, averages, bcs_left, bcs_right)
    r = np.linalg.solve(m, k)
    poly: list[Polynomial1D] = []
    for i, a in enumerate(averages[:]):
        p = basis[0] * float(a) + sum(basis[j + 1] * r[n // 2 * i + j] for j in range(n))
        poly.append(p)
    spl = Spline1Di(tuple(p.coefficients for p in poly))
    return spl


def _nodal_polynomial_coefficients(n: int) -> npt.NDArray[np.float64]:
    """Return coefficients for nodal polynomials.

    Parameters
    ----------
    n : int
        Order of polynomials. Number of terms will be one higher.

    Returns
    -------
    (n + 1, n + 1) array
        Matrix, where each row has the coefficients of the polynomial
        basis in from the constant term, to the :math:`t^n` term.
    """
    assert n & 1 == 1
    a = np.zeros((n + 1, n + 1), np.float64)
    f = 1.0
    k = np.ones(n + 1)
    for r in range((n + 1) // 2):
        a[2 * r + 0, r] = f
        a[2 * r + 1, :] = k
        f *= r + 1
        k[r:] *= np.arange(n + 1 - r)
    return np.linalg.inv(a).T


def _nodal_interpolating_basis(n: int) -> tuple[Polynomial1D]:
    """Create interpolating basis for nodal spline with degree n.

    Parameters
    ----------
    n : int
        Order of polynomials to use. Number of terms will be one more.

    Returns
    -------
    tuple of Polynomial1D
        Tuple of basis polynomials for nodal interpolation.
    """
    assert n & 1 == 1
    m = _element_polynomial_coefficients(n)

    # re-order rows of m
    def _permute_gen(n: int) -> Generator[int, None, None]:
        for i in range((n + 1) // 2):
            yield 2 * i
        for i in range((n + 1) // 2):
            yield 2 * i + 1

    return tuple(Polynomial1D(m[i, :]) for i in _permute_gen(n))


def _nodal_interpolation_system(
    n: int,
    basis: Iterable[Polynomial1D],
    nds: npt.ArrayLike,
    bc_left: Iterable[SplineBC],
    bc_right: Iterable[SplineBC],
) -> npt.NDArray[np.float64]:
    """Create interpolation matrix for natural nodal spline.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    nds : (M,) array_like
        Nodal values which are to be interpolated.
    bc_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    ((M + 1) * n//2, (M + 1) * n//2) array
        System matrix which gives coefficients required to construct the spine.
    ((M + 1) * n//2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 1
    values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.stack(values, axis=0)
    nodes = np.array(nds, np.float64)
    assert len(nodes.shape) == 1
    nelem = int(nodes.shape[0]) - 1
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)

    n_node = (n - 1) // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = np.zeros(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n)
        if bc.coefficiens.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f"coefficients (got {bc.coefficiens.shape} when expecting"
                f" {(n,)})."
            )
        for j in range(n):
            m[p, inode] += bc.coefficiens[j] * v[j, 1:, 0]
            k[p] -= bc.coefficiens[j] * v[j, 0, 0] * nodes[0]
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = n_node * i + np.arange(n)
        iright = n_node * (i + 1) + np.arange(n)
        for j in range(n_node, n):
            m[p, ileft] -= v[j, 1:, 1]
            m[p, iright] += v[j, 1:, 0]
            k[p] = v[j, 0, 1] * nodes[i] - v[j, 0, 0] * nodes[i + 1]
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        inode = np.arange(n) + (nelem - 1) * n_node
        if bc.coefficiens.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficiens.shape} when expecting"
                f" {(n,)})."
            )
        for j in range(n):
            m[p, inode] += bc.coefficiens[j] * v[j, 1:, 1]
            k[p] -= bc.coefficiens[j] * v[j, 0, 1] * nodes[-1]
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m, k


def nodal_interpolating_splinei(
    n: int,
    nds: npt.ArrayLike,
    bcs_left: Iterable[SplineBC],
    bcs_right: Iterable[SplineBC],
) -> Spline1Di:
    """Create interpolating spline, which has specified nodal values.

    Parameters
    ----------
    n : int
        Order of the spline. Must be odd, otherwise a ``ValueError`` will be raised.
    nds : (N,) array_like
        Nodes to hit.
    bcs_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bcs_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    Spline1Di
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    if (n & 1) != 1:
        raise ValueError(f"Spline order must be even (instead it was {n}).")
    nodes = np.array(nds, np.float64)
    if len(nodes.shape) != 1:
        raise ValueError(
            f"Nodes should be a 1D array, instead they have the shape {nodes.shape}"
        )
    basis = _nodal_interpolating_basis(n)
    m, k = _nodal_interpolation_system(n, basis, nodes, bcs_left, bcs_right)
    r = np.linalg.solve(m, k)
    poly: list[Polynomial1D] = []
    for i in range(nodes.shape[0] - 1):
        p = (
            basis[0] * float(nodes[i])
            + basis[1] * float(nodes[i + 1])
            + sum(basis[j + 2] * r[n // 2 * i + j] for j in range(n - 1))
        )
        poly.append(p)
    spl = Spline1Di(tuple(p.coefficients for p in poly))
    return spl
