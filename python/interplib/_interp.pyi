from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt

def test() -> str: ...
def lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def dlagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def d2lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def hermite(
    x: npt.NDArray[np.float64],
    bc1: tuple[float, float, float],
    bc2: tuple[float, float, float],
) -> npt.NDArray[np.float64]: ...
def bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute Bernstein polynomials of given order at given locations.

    Parameters
    ----------
    n : int
       Order of polynomials used.
    x : (M,) array_like
       Flat array of locations where the values should be interpolated.

    Returns
    -------
    (M, n) arr
       Matrix containing values of Bernstein polynomial :math:`B^M_j(x_i)` as the
       element ``array[i, j]``.
    """
    ...

def bernstein_coefficients(x: npt.ArrayLike, /) -> npt.NDArray[np.double]:
    """Compute Bernstein polynomial coefficients from a power series polynomial.

    Parameters
    ----------
    x : array_like
       Coefficients of the polynomial from 0-th to the highest order.

    Returns
    -------
    array
       Array of coefficients of Bernstein polynomial series.
    """
    ...

def compute_gll(
    order: int, max_iter: int = 10, tol: float = 1e-15
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre-Lobatto integration nodes and weights.

    If you are often re-using these, consider caching them.

    Parameters
    ----------
    order : int
       Order of the scheme. The number of node-weight pairs is one more.
    max_iter : int, default: 10
       Maximum number of iterations used to further refine the values.
    tol : float, default: 1e-15
       Tolerance for stopping the refinement of the nodes.

    Returns
    -------
    array
       Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.
    array
       Array of integration weights which correspond to the nodes.
    """
    ...

class Basis1D:
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Basis1D: ...
    @property
    def antiderivative(self) -> Basis1D: ...

class Polynomial1D(Basis1D):
    r"""Function with increasing integer power basis.

    Given a set of coefficients :math:`\left\{ a_0, \dots, a_n \right\}`,
    the resulting polynomial will be:

    .. math::

        p(x) = \sum\limits_{k=0}^n a_k x^k

    Parameters
    ----------
    coefficients : (N,) array_like
        Coefficients of the polynomial, starting at the constant term, up to the
        term with the highest power.
    """

    def __new__(cls, coefficients: npt.ArrayLike, /) -> Self: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Coefficients of the polynomial."""
        ...

    def __add__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __radd__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __neg__(self) -> Polynomial1D: ...
    def __mul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __rmul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __pos__(self) -> Polynomial1D: ...
    def __pow__(self, i: int) -> Polynomial1D: ...
    @property
    def derivative(self) -> Polynomial1D:
        """Return the derivative of the polynomial."""
        ...

    @property
    def antiderivative(self) -> Polynomial1D:
        """Return the antiderivative of the polynomial."""
        ...

    @property
    def order(self) -> int:
        """Order of the polynomial."""
        ...

    def __setitem__(self, key: int, value: float | np.floating) -> None: ...
    @classmethod
    def lagrange_nodal_basis(cls, nodes: npt.ArrayLike, /) -> tuple[Polynomial1D, ...]:
        """Return Lagrange nodal polynomial basis on the given set of nodes.

        Parameters
        ----------
        nodes : (N,) array_like
            Nodes used for Lagrange polynomials.

        Returns
        -------
        tuple of ``N`` :class:`Polynomial1D`
            Lagrange basis polynomials, which are one at the node of their index
         and zero at all other.
        """
        ...

    @classmethod
    def lagrange_nodal_fit(
        cls, nodes: npt.ArrayLike, values: npt.ArrayLike, /
    ) -> Polynomial1D:
        """Use Lagrange nodal polynomial basis to fit a function.

        Equivalent to calling::

            sum(b * y for (b, y) in zip(Polynomial1D.lagrange_nodal_basis(nodes), values)

        Parameters
        ----------
        nodes : (N,) array_like
            Nodes used for Lagrange polynomials.
        values : (N,) array_like
            Values of the function at the nodes.

        Returns
        -------
        Polynomial1D
            Polynomial of order ``N`` which matches function exactly at the nodes.
        """
        ...

    def offset_by(self, t: float | np.floating, /) -> Polynomial1D:
        r"""Compute polynomial offset by specified amount.

        The offset polynomial :math:`p^\prime(t)` is such that:

        .. math::

            p^\prime(t) = p(t + t_0),

        where :math:`p(t)` is the original polynomial and :math:`t_0` is the offset.

        Parameters
        ----------
        t : float
           Amount by which to offset the polynomial by.

        Returns
        -------
        Polynomial1D
           Polynomial which has been offset by the specified amount.

        """
        ...

    def scale_by(self, a: float | np.floating, /) -> Polynomial1D:
        r"""Compute polynomial scale by specified amount.

        The offset polynomial :math:`p^\prime(t)` is such that:

        .. math::

            p^\prime(t) = p(a \cdot t),

        where :math:`p(t)` is the original polynomial and :math:`a` is the scaling
        parameter.

        Parameters
        ----------
        a : float
           Amount by which to scale the polynomial by.

        Returns
        -------
        Polynomial1D
           Polynomial which has been scaled by the specified amount.

        """
        ...

class Spline1D(Basis1D):  # TODO: implement other methods of Poylnomial1D
    r"""Piecewise polynomial function, defined between nodes.

    Given the set of nodes :math:`\left\{x_0, \dots, x_n \right\}` and sets of
    coefficients :math:`\left\{A_0, \dots A_{n-1} \right\}`, the polynomial evaluated
    between nodes :math:`x_i` and :math:`x_{i + 1}` will be:

    .. math::

        p(x) = \sum\limits_{k=0}^M A_i^k \left( 2 \frac{x - x_i}{x_{i+1} - x_{i}} - 1
        \right)^k ,

    where :math:`A_i^k` is the k-th coefficient of the set :math:`A_i`.

    These splines thus allow for simple stitching of solutions of different 1D elements
    together, as those are typically defined on a reference element, where the
    computational space :math:`\xi \in \left[-1, +1\right]` is then mapped to
    physical space.

    Parameters
    ----------
    nodes : (N + 1,) array_like
        Sequence of :math:`N + 1` values, which mark where a different set of
        coefficients is to be used.
    coefficients : (N, M) array_like
        Sequence of :math:`M` coefficients for each of the :math:`N` elements.
    """

    def __new__(cls, nodes: npt.ArrayLike, coefficients: npt.ArrayLike, /) -> Self: ...
    @property
    def nodes(self) -> npt.NDArray[np.float64]:
        """Nodes of the spline."""
        ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Coefficients of the polynomials."""
        ...
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Spline1D:
        """Return derivative of the spline."""
        ...
    @property
    def antiderivative(self) -> Spline1D:
        """Return antiderivative of the spline."""
        ...

class Spline1Di(Spline1D):  # TODO: implement other methods of Poylnomial1D
    def __new__(cls, coefficients: npt.ArrayLike, /) -> Self: ...
    @property
    def nodes(self) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Spline1D: ...
    @property
    def antiderivative(self) -> Spline1D: ...
