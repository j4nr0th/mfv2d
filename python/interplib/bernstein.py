"""Implementation of functions related to Bernstein polynomials."""

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt

from interplib._interp import Basis1D, Polynomial1D, bernstein1d, bernstein_coefficients


class Bernstein1D(Basis1D):
    r"""Class used to represent polynomials as a series of Bernstein basis.

    Bernstein basis polynomials of order :math:`n` are
    :math:`B^n = \left\{ B^n_0, \dots, B^n_n \right\}` is defined as:

    .. math::

        B^n_k = {n \choose k} t^k (1 - t)^k = {n \choose k} t^k \sum_{i = 0}^{n - k}
        {n - k \choose i} (-t)^i

    Parameters
    ----------
    coefficients : array_like
        Coefficients of the Bernstein basis. Their order will be equal to the number of
        coefficients.

    Examples
    --------
    As a quick example, first let us plot some of the Bernstein basis polynomials:

    .. jupyter-execute::

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from interplib import Bernstein1D
        >>>
        >>> order = 5
        >>> xplt = np.linspace(0, 1, 128)
        >>> plt.figure()
        >>> for i in range(order):
        ...     coeffs = np.arange(order) == i
        ...     basis = Bernstein1D(coeffs)
        ...     plt.plot(xplt, basis(xplt), label=f"$B^{order - 1}_{i}$")
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()
        >>> # No output for doctest

    To show the key strength of these polynomials, let's check at how they
    fare at computing high order polynomials. As such, let's fit a polynomial
    to a test function :math:`f(x) = 3 \sin(5 \pi x)` with the order of 30.

    .. jupyter-execute::

        >>> from interplib import Polynomial1D
        >>>
        >>> order = 30
        >>> sample_fn = lambda x: 3 * np.sin(5 * np.pi * x)
        >>> sample_points = (1 - np.cos(np.linspace(0, np.pi, order))) / 2
        >>> poly = Polynomial1D.lagrange_nodal_fit(
        ...     sample_points, sample_fn(sample_points)
        ... )
        >>> bern = Bernstein1D.fit_nodal(
        ...     sample_points, sample_fn(sample_points)
        ... )
        >>> plt.figure()
        >>> err_pwr = poly(xplt) - sample_fn(xplt)
        >>> err_brn = bern(xplt) - sample_fn(xplt)
        >>> plt.plot(xplt, err_pwr, label="Power Series")
        >>> plt.plot(xplt, err_brn, label="Bernstein Polynomial")
        >>> plt.gca().set(xlabel="$x$", ylabel="Error")
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()
        >>> print("Power series max error:", np.max(np.abs(err_pwr)))
        >>> print("Bernstein polynomial max error:", np.max(np.abs(err_brn)))
    """

    _coefficients: npt.NDArray[np.double]

    def __eq__(self, other) -> bool:
        """Compare if the polynomial is equal to another."""
        if isinstance(other, Bernstein1D):
            return np.allclose(self._coefficients, other._coefficients)
        return NotImplemented

    def __init__(self, coefficients: npt.ArrayLike) -> None:
        self._coefficients = np.array(coefficients, np.float64, ndmin=1)
        if self._coefficients.ndim != 1:
            raise ValueError(
                f"Input array is not 1D, but has {self._coefficients.ndim} dimensions"
            )

    @classmethod
    def from_power_series(cls, series: npt.ArrayLike | Polynomial1D) -> Self:
        """Create a Bernstein polynomial representation of a power series polynomial."""
        pwr: npt.NDArray[np.float64]
        if isinstance(series, Polynomial1D):
            pwr = series.coefficients
        else:
            pwr = np.array(series, np.float64, ndmin=1)
            if pwr.ndim != 1:
                raise ValueError(
                    f"The coefficients were not given as 1D array, instead {pwr.ndim} "
                    "dimensions were given."
                )
        return cls(bernstein_coefficients(pwr))

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Coefficients of the Bernstein basis."""
        return np.array(self._coefficients, np.float64)

    @property
    def order(self) -> int:
        """Order of the polynomial."""
        return self._coefficients.size - 1

    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]:
        """Evaluate the polynomial at specified locations.

        Parameters
        ----------
        array_like
            Positions where the polynomial should be evaluated.

        Returns
        -------
        array
            Values of the polynomial at specified positions. It will have the same shape
            as the input.
        """
        base_input = np.asarray(x)
        shape = base_input.shape
        flat_input = base_input.reshape((-1,))
        coeff_mat = bernstein1d(self._coefficients.size - 1, flat_input)
        return np.astype(np.reshape(coeff_mat @ self._coefficients, shape), np.float64)

    @property
    def derivative(self) -> Bernstein1D:
        """Analytical derivative of the polynomial."""
        new_coeffs = np.zeros(self._coefficients.size - 1, np.float64)

        new_coeffs -= self._coefficients[:-1]
        new_coeffs += self._coefficients[+1:]

        return Bernstein1D(new_coeffs * (self._coefficients.size - 1))

    @property
    def antiderivative(self) -> Bernstein1D:
        """Analytical antiderivative of the polynomial."""
        new_coeffs = np.zeros(self._coefficients.size + 1, np.float64)
        for i, v in enumerate(self._coefficients):
            new_coeffs[i + 1 :] += v

        return Bernstein1D(new_coeffs / self._coefficients.size)

    def __repr__(self) -> str:
        """Return representation of the object."""
        return f"Bernstein1D({self._coefficients})"

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        out = str()
        n = self._coefficients.size - 1
        for i, k in enumerate(self._coefficients):
            v = abs(k)
            s = f"{v} B({n},{i})"
            if k < 0:
                out += " - " + s
            else:
                out += " + " + s
        return out[1:]

    @classmethod
    def fit_nodal(cls, nodes: npt.ArrayLike, values: npt.ArrayLike) -> Self:
        """Fit the polynomial to specified nodes exactly.

        Parameters
        ----------
        nodes : array_like
            Array of nodes at which the values are given.
        values : array_like
            Values of the function at the given ``nodes``.

        Returns
        -------
        Self
            Bernstein polynomial which exactly matches the specified points.
        """
        x = np.asarray(nodes)
        if x.ndim != 1:
            raise ValueError(
                f"The input nodes are not a 1D, but instead have {x.ndim} dimensions."
            )
        y = np.asarray(values)
        if x.shape != y.shape:
            raise ValueError("The shapes of nodes and values don't match.")
        n = x.size
        mat = bernstein1d(n - 1, x)
        return cls(np.linalg.solve(mat, y))
