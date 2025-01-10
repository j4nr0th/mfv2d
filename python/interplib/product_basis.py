"""Functions for implementation of product basis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from interplib._interp import Basis1D


@dataclass(frozen=True)
class BasisProduct2D:
    r"""A basis function, which is a product of two 1D basis.

    Given basis functions :math:`\phi(x)` and :math:`\psi(x)`, the product basis
    function is written as:

    .. math::

        \Phi(\xi, \eta) = \phi(\xi) \cdot \psi(\eta)

    Parameters
    ----------
    b1 : Basis1D
        First basis function.

    b2 : Basis1D
        Second basis function.
    """

    b1: Basis1D
    b2: Basis1D

    def __call__(
        self,
        x0: npt.ArrayLike,
        x1: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the basis function at given positions.

        Parameters
        ----------
        x : (N, 2) array
            Array of position vectors.

        Returns
        -------
        (N,) array
            Array of basis function values at given positions.
        """
        return np.astype(self.b1(x0) * self.b2(x1), np.float64)

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
        cls, basis: Iterable[Basis1D], /
    ) -> tuple[tuple[Self, ...], ...]:
        r"""Create outer product basis.

        Given a set of 1D basis :math:`\Phi = \left\{\phi_0,\phi_1,\dots,\phi_n\right\}`,
        the set of 2D basis :math:`\psi \in \Psi` is created for:

        .. math::

            \psi_{i, j}(\xi, \eta) = \phi_i(\xi) \cdot \phi_j(\eta),\quad i, j \in [0, n]

        Parameters
        ----------
        basis : Iterable of Basis1D
            Iterable of the 1D basis functions.

        Returns
        -------
        (N, N) tuple of BasisProduct2D
        """
        basis_tuple = tuple(basis)
        return tuple(tuple(cls(b1, b2) for b2 in basis_tuple) for b1 in basis_tuple)
