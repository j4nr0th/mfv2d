"""Spline auxiliary functions and classes."""

from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D, Spline1D


@dataclass(init=False, frozen=True)
class SplineBoundaryCondition:
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


def _construct_interpolation_matrix(
    n: int,
    avg: npt.ArrayLike,
    bc_left: Iterable[SplineBoundaryCondition],
    bc_right: Iterable[SplineBoundaryCondition],
) -> npt.NDArray[np.float64]:
    """Create interpolation matrix for natural element.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    avg : (M,) array_like
        Averages on the elements which are to be interpolated.

    Returns
    -------
    ((M + 1) * n//2, (M + 1) * n//2) array
        System matrix which gives coefficients required to construct the spine.
    ((M + 1) * n//2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 0
    polynomials = _element_interpolating_basis(n)
    values: list[npt.NDArray[np.float64]] = []
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative() for poly in polynomials)
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


if __name__ == "__main__":
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("GTK3Agg")
    NPLT = 513
    tplt = np.linspace(0, 1, NPLT)
    n = 8
    basis = _element_interpolating_basis(n)
    for ib, p in enumerate(basis):
        plt.plot(tplt, p(tplt), label=f"$\\phi_{ib} = {p}$")
    plt.grid()
    plt.legend()
    plt.show()
    averages = [3.0, 2.0, 3.0, 10.2, 3.0]
    m, k = _construct_interpolation_matrix(
        n,
        averages,
        [
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0),
        ],
        [
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.0),
            SplineBoundaryCondition([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0),
        ],
    )
    plt.spy(m)
    plt.show()
    r = np.linalg.solve(m, k)
    print(r)
    # exit()

    poly: list[Polynomial1D] = []
    for i, a in enumerate(averages[:]):
        p = basis[0] * a + sum(basis[j + 1] * r[n // 2 * i + j] for j in range(n))
        # for j in range(n // 2):
        #     p += basis[2 * j + 1] * r[n // 2 * i + j]
        #     p += basis[2 * j + 2] * r[n // 2 * (i + 1) + j]
        print(p.derivative().derivative()(0.0), p.derivative().derivative()(1.0))
        # print(
        #     p.derivative().derivative().derivative()(0.0),
        #     p.derivative().derivative().derivative()(1.0),
        # )
        poly.append(p)

    for ip, p in enumerate(poly):
        plt.plot(tplt + ip, p(tplt), label=f"$s_{ip}$")
    plt.scatter(np.arange(len(averages)) + 0.5, averages, label="averages")
    plt.legend()
    plt.grid()
    plt.show()

    spl = Spline1D(np.arange(len(poly) + 1), tuple(p.coefficients for p in poly))
    tplt = np.linspace(0, len(poly), NPLT * len(poly))
    plt.plot(tplt, spl(tplt), label="spline")
    plt.scatter(np.arange(len(averages)) + 0.5, averages, label="averages")
    plt.legend()
    plt.grid()
    plt.show()
