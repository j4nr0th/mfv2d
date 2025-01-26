"""Functions for implementation of product basis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self, overload

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D
from interplib.interp2d import Function2D, Polynomial2D


@dataclass(frozen=True)
class BasisProduct2D(Function2D):
    r"""A basis function, which is a product of two 1D basis.

    Given basis functions :math:`\phi(x)` and :math:`\psi(x)` with a constant weight
    :math:`w`, the product basis function is written as:

    .. math::

        \Phi(\xi, \eta) = w \cdot \phi(\xi) \cdot \psi(\eta)

    Parameters
    ----------
    b1 : Polynomial1D
        First basis function.

    b2 : Polynomial1D
        Second basis function.
    """

    b1: Polynomial1D
    b2: Polynomial1D

    @overload
    def __call__(self, x0: None, x1: float) -> Polynomial1D: ...
    @overload
    def __call__(self, x0: float, x1: None) -> Polynomial1D: ...
    @overload
    def __call__(
        self, x0: npt.ArrayLike, x1: npt.ArrayLike
    ) -> npt.NDArray[np.float64]: ...

    def __call__(
        self,
        x0: npt.ArrayLike | None | float,
        x1: npt.ArrayLike | float | None,
    ) -> npt.NDArray[np.float64] | Polynomial1D:
        """Evaluate the basis function at given positions.

        If one of the parameters is missing, then a 1D function in the other
        parameter will be returned.

        Parameters
        ----------
        x0 : (N,) array or None
            Array of values for the first parameter. If missing, a function in x1
            will be returned.
        x1 : (N,) array or None
            Array of values for the first parameter. If missing, a function in x0
            will be returned.

        Returns
        -------
        (N,) array
            Array of basis function values at given positions.
        Polynomial1D
            Function of the remaining parameter.
        """
        if x0 is not None and x1 is not None:
            return np.astype(self.b1(x0) * self.b2(x1), np.float64)
        if x0 is not None and x1 is None:
            a = np.asarray(x0, np.float64)
            v = float(self.b1(float(a)))
            return float(v) * self.b2
        if x0 is None and x1 is not None:
            a = np.asarray(x1, np.float64)
            v = float(self.b2(float(a)))
            return v * self.b1
        raise TypeError("Invalid parameters.")

    def partial(self, dimension: int, /) -> BasisProduct2D:
        """Compute the partial derivative of a basis."""
        if dimension not in [0, 1]:
            raise ValueError(
                f"The dimension should be 0 or 1, but is {dimension} instead."
            )
        if dimension == 0:
            return BasisProduct2D(self.b1.derivative, self.b2)
        elif dimension == 1:
            return BasisProduct2D(self.b1, self.b2.derivative)
        assert False

    @classmethod
    def outer_product_basis(
        cls,
        basis1: Iterable[Polynomial1D],
        basis2: Iterable[Polynomial1D] | None = None,
        /,
    ) -> tuple[tuple[Self, ...], ...]:
        r"""Create outer product basis.

        Given a set of 1D basis :math:`\Phi = \left\{\phi_0,\phi_1,\dots,\phi_n\right\}`
        and a set of 1D basis :math:`\Psi = \left\{\psi_0,\psi_1,\dots,\psi_m\right\}`
        the set of 2D basis :math:`p_{i,j} \in P` is created for:

        .. math::

            p_{i, j}(\xi, \eta) = \phi_i(\xi) \cdot \psi_j(\eta),\quad i \in [0, n], j
            \in [0, m]

        Parameters
        ----------
        basis1 : Iterable of N Basis1D
            Iterable of the 1D basis functions.
        basis2 : Iterable of M Basis1D, optional
            Iterable of 1D basis functions. If not provided, then the values in ``basis1``
            are used instead.

        Returns
        -------
        (N, M) tuple of BasisProduct2D
        """
        basis1_tuple = tuple(basis1)
        basis2_tuple = tuple(basis2) if basis2 is not None else basis1_tuple
        return tuple(tuple(cls(b1, b2) for b2 in basis2_tuple) for b1 in basis1_tuple)

    def __mul__(self, other: float | BasisProduct2D) -> BasisProduct2D:
        """Multiply two product basis together or scale by scalar."""
        if isinstance(other, BasisProduct2D):
            return BasisProduct2D(self.b1 * other.b1, self.b2 * other.b2)
        try:
            v = float(other)
            if v > 0:
                root = np.sqrt(v)
                return BasisProduct2D(self.b1 * root, self.b2 * root)
            else:
                root = np.sqrt(-v)
                return BasisProduct2D(self.b1 * root, self.b2 * -root)
        except Exception:
            return NotImplemented

    def __rmul__(self, other: float | BasisProduct2D) -> BasisProduct2D:
        """Reversed multiply."""
        return self.__mul__(other)

    def __add__(self, other: BasisProduct2D) -> BasisProduct2D:
        """Add two basis product basis together."""
        # BUG: WRONG!
        return BasisProduct2D(self.b1 + other.b1, self.b2 + other.b2)

    def as_polynomial(self) -> Polynomial2D:
        """Convert the factored version into a polynomial."""
        return Polynomial2D(*(k * self.b1 for k in self.b2.coefficients))
