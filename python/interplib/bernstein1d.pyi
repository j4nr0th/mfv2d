"""Class supporting Bernstein basis representaitons of polynomials."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D  # , bernstein1d

class Bernstein1D(Polynomial1D):
    """Class which represents a polynomial in Bernstein basis instead of power basis.

    Parameters
    ----------
    coefficients : array_like
        Coefficients of the polynomial in power basis.
    """

    def __init__(self, coefficients: npt.ArrayLike, /) -> None: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    @property
    def power_basis(self) -> Polynomial1D: ...
    def __add__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __radd__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __neg__(self) -> Bernstein1D: ...
    def __mul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __rmul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __pos__(self) -> Polynomial1D: ...
    def __pow__(self, i: int) -> Polynomial1D: ...
    @property
    def derivative(self) -> Polynomial1D: ...
    @property
    def antiderivative(self) -> Polynomial1D: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: int) -> np.float64: ...
    def __setitem__(self, key: int, value: float | np.floating) -> None: ...
    def __iter__(self) -> Iterator[np.float64]: ...

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     tplt = np.linspace(0, 1, 257)

#     for order in [0, 1, 2, 4, 10]:
#         plt.figure()
#         plt.title(f"Order {order}")
#         vals = bernstein1d(order, tplt)
#         for i in range(vals.shape[1]):
#             plt.plot(tplt, vals[:, i], label=f"$B^{{{order}}}_{{{i}}}(t)$")
#         plt.grid()
#         plt.legend()
#         plt.show()
