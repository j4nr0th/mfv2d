"""Two dimensional polynomials.

For now these are implemented with Python on the back of C types for 1D functions.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy import typing as npt

from interplib._interp import Polynomial1D


class Function2D:
    """Two dimensional function."""

    def partial(self, dim: int, /) -> Function2D:
        """Partial derivative with respect to the specified dimension."""
        raise NotImplementedError

    def antiderivative(self, dim: int, /) -> Function2D:
        """Anti-derivative with respect to the specified dimension."""
        raise NotImplementedError

    def __call__(
        self, x1: npt.ArrayLike, x2: npt.ArrayLike, /
    ) -> npt.NDArray[np.float64]:
        """Call the function."""
        raise NotImplementedError


class Polynomial2D(Function2D):
    r"""Two dimensional polynomial in power basis.

    The function is represented as sum of 1d polynomials multiplied
    with different powers of the second parameter:

    .. math::

        p(x_1, x_2) = \sum\limits_{i=0}^{N_p} {x_2}^i p_i(x_1)

    An important aspect of this type is that it evaluates compositions. This means that
    if ``a``, ``b``, and ``c`` are all type :class:`Polynomial2D`, then it holds that in
    the absence of rounding errors::

        a(b(x1, x2), c(x1, x2)) == a(b, c)(x1, x2)

    In a similar way, if ``b`` and ``c`` are instead :class:`Polyomial1D` objects, then::

        a(b(x), c(x)) == a(b, c)(x)

    This is intended for use when the function is defined in space where coordinates are
    mapped either by a single parameter (such as on a line) or two variables.
    """

    polynomials: tuple[Polynomial1D, ...]

    def __init__(self, *polynomials: Polynomial1D) -> None:
        self.polynomials = polynomials

    @overload
    def __call__(
        self, x: npt.ArrayLike, y: npt.ArrayLike, /
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def __call__(self, x1: None, x2: float, /) -> Polynomial1D: ...

    @overload
    def __call__(self, x1: float, x2: None, /) -> Polynomial1D: ...

    @overload
    def __call__(self, x1: Polynomial2D, x2: Polynomial2D, /) -> Polynomial2D: ...

    @overload
    def __call__(self, x1: Polynomial1D, x2: Polynomial1D, /) -> Polynomial1D: ...

    def __call__(
        self,
        x1: npt.ArrayLike | None | float | Polynomial2D | Polynomial1D,
        x2: npt.ArrayLike | float | None | Polynomial2D | Polynomial1D,
        /,
    ) -> npt.NDArray[np.float64] | Polynomial1D | Polynomial2D:
        """Evaluate a function either at a point or along a line with one value fixed."""
        if isinstance(x1, Polynomial1D) and isinstance(x2, Polynomial1D):
            out_polynomial = Polynomial1D((0,))
            q1_poly: int | Polynomial1D = 1
            for p in self.polynomials:
                k1 = Polynomial1D((0,))
                p1_poly: int | Polynomial1D = 1
                for i, c in enumerate(p.coefficients):
                    k1 += c * p1_poly
                    p1_poly *= x1

                q1_poly *= x2
                out_polynomial += q1_poly * k1
            return out_polynomial

        if isinstance(x1, Polynomial2D) and isinstance(x2, Polynomial2D):
            # This is painful and ugly, but straightforward
            out_poly = Polynomial2D(Polynomial1D((0,)))
            q_poly: int | Polynomial2D = 1
            for p in self.polynomials:
                k = Polynomial2D(Polynomial1D((0,)))
                p_poly: int | Polynomial2D = 1
                for i, c in enumerate(p.coefficients):
                    k += c * p_poly
                    p_poly *= x1

                out_poly += q_poly * k
                q_poly *= x2
            return out_poly
        if x1 is not None and x2 is not None:
            v1 = np.asarray(x1, np.float64)
            v2 = np.asarray(x2, np.float64)
            # if v1.shape != v2.shape:
            #     raise ValueError("The two arrays must have the same shape.")

            out = 0.0 * v1 * v2
            for i, poly in enumerate(self.polynomials):
                out += v2**i * poly(v1)

            return np.astype(out, np.float64)

        if x1 is None and x2 is not None:
            coeff = float(x2)  # type: ignore
            p_out = sum(coeff**i * p for i, p in enumerate(self.polynomials))
            assert isinstance(p_out, Polynomial1D)
            return p_out

        if x1 is not None and x2 is None:
            coeff = float(x1)  # type: ignore
            p_out = Polynomial1D([p(coeff) for p in self.polynomials])
            return p_out

        raise ValueError("Both x1 and x2 may not be None.")

    def partial(self, dim: int, /) -> Polynomial2D:
        """Partial derivative with respect to the specified dimension."""
        if dim == 0:
            return Polynomial2D(*(p.derivative for p in self.polynomials))

        if dim == 1:
            return Polynomial2D(
                *((i + 1) * p for i, p in enumerate(self.polynomials[1:]))
            )

        raise ValueError("Dimension can not be 2 or more.")

    def antiderivative(self, dim: int, /) -> Function2D:
        """Anti-derivative with respect to the specified dimension."""
        if dim == 0:
            return Polynomial2D(*(p.antiderivative for p in self.polynomials))

        if dim == 1:
            return Polynomial2D(
                Polynomial1D((0,)),
                *((1 / (i + 1)) * p for i, p in enumerate(self.polynomials)),
            )

        raise ValueError("Dimension can not be 2 or more.")

    def __add__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Add two polynomials together."""
        if isinstance(other, Polynomial2D):
            p_longer: tuple[Polynomial1D, ...]
            p_shorter: tuple[Polynomial1D, ...]
            if len(self.polynomials) < len(other.polynomials):
                p_longer = other.polynomials
                p_shorter = self.polynomials
            else:
                p_shorter = other.polynomials
                p_longer = self.polynomials
            out_polys: list[Polynomial1D] = []
            for i in range(len(p_shorter)):
                out_polys.append(p_longer[i] + p_shorter[i])
            out_polys.extend(p_longer[len(p_shorter) :])
            return Polynomial2D(*out_polys)
        try:
            v = float(other)
            return Polynomial2D(self.polynomials[0] + v, *self.polynomials[1:])
        except Exception:
            return NotImplemented

    def __radd__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Add two polynomials together."""
        return self.__add__(other)

    def __sub__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Subtracts two polynomials."""
        if isinstance(other, Polynomial2D):
            out_polys: list[Polynomial1D] = list()

            l1 = len(self.polynomials)
            l2 = len(other.polynomials)
            for p in range(min(l1, l2)):
                out_polys.append(self.polynomials[p] + ((-1) * other.polynomials[p]))
            if l1 > l2:
                for i in range(l2, l1):
                    out_polys.append(self.polynomials[i])
            else:
                for i in range(l1, l2):
                    out_polys.append((-1) * other.polynomials[i])

            return Polynomial2D(*out_polys)
        try:
            v = float(other)
            return Polynomial2D(self.polynomials[0] + (-v), *self.polynomials[1:])
        except Exception:
            return NotImplemented

    def __rsub__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Subtract polynomials."""
        return self.__sub__(other)

    def __mul__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Multiply with either another polynomial or a constant."""
        if isinstance(other, Polynomial2D):
            n1 = len(self.polynomials)
            n2 = len(other.polynomials)
            polynomials: list[Polynomial1D] = [Polynomial1D(())] * (n1 + n2 - 1)
            for i1, p1 in enumerate(self.polynomials):
                for i2, p2 in enumerate(other.polynomials):
                    polynomials[i1 + i2] += p1 * p2
            return Polynomial2D(*polynomials)
        try:
            v = float(other)
            return Polynomial2D(*(v * p for p in self.polynomials))
        except Exception:
            return NotImplemented

    def __rmul__(self, other: Polynomial2D | float) -> Polynomial2D:
        """Multiply with either another polynomial or a constant."""
        return self.__mul__(other)
