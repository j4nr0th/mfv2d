"""Functions and classes dedicated to interpolation of curves."""

from typing import Iterable

import numpy as np
import numpy.typing as npt

from interplib._interp import Spline1Di


class Curve:
    """Interpolation of a curve."""

    splines: tuple[Spline1Di, ...]

    def __init__(self, splines: Iterable[Spline1Di]) -> None:
        self.splines = tuple(splines)

    def __call__(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute curve interpolated at positions x."""
        return np.stack(tuple(s(x) for s in self.splines), axis=0)

    @property
    def length_derivative2(self) -> Spline1Di:
        """Return ds^2, which is the arc length derivative squared."""
        raise NotImplementedError
