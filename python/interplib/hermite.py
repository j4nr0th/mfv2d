"""Interpolation functions related to Hermite cubic splines"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D
from interplib._interp import hermite as _hermite

_H0 = Polynomial1D([1, 0, -3, 2])
"""
math:`H_0(t) = 2 t^3 - 3 t^2 + 1`
"""
_H1 = Polynomial1D([0, 0, 3, -2])
"""
    :math:`H_1(t) = -2 t^3 + 3 t^2`
"""
_H2 = Polynomial1D([0, 1, -2, 1])
"""
    :math:`H_2(t) = t^3 - 2 t^2 + t`
"""
_H3 = Polynomial1D([0, 0, -1, 1])
""" 
    :math:`H_3(t) = t^3 - t^2`
"""

@dataclass
class SplineBC:
    r"""Represents boundary conditions for a spline in terms of its derivatives.

    It represents the equation 
    :math:`k_1 \frac{d f}{d t} + k_2 \frac{d^2 f}{{d t}^2} = v`

    Parameters
    ----------
    k1 : float
        Coefficient of the first derivative.
    k2 : float
        Coefficient of the second derivative.
    v : float
        Weighted sum of the first and second derivatives.
    """
    k1: float
    k2: float
    v: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the object as a tuple of the three values."""
        return (float(self.k1), float(self.k2), float(self.v))


@dataclass(frozen=True, eq=False, init=False)
class HermiteSpline:
    """Spline with second order continuity at the nodes.

    Spline is defined per segment as a combination of its four basis:

    :math:`H_0(t) = 2 t^3 - 3 t^2 + 1`

    :math:`H_1(t) = -2 t^3 + 3 t^2`

    :math:`H_2(t) = t^3 - 2 t^2 + t`
    
    :math:`H_3(t) = t^3 - t^2`
    
    Parameters
    ----------
    x : (N,) array_like
        One dimensional array-like specifying nodal values of the spline.
    bc_left : SplineBC, optional
        Boundary condition at the first node of the spline. If ``None``,
        a natural boundary condition will be used (no second derivative).
    bc_right : SplineBC, optional
        Boundary condition at the last node of the spline. If ``None``,
        a natural boundary condition will be used (no second derivative).
    """
    nodes: npt.NDArray[np.float64]
    derivatives: npt.NDArray[np.float64]

    def __init__(
            self,
            x: npt.ArrayLike, 
            bc_left: SplineBC | None = None, 
            bc_right: SplineBC | None = None,
            copy_nodes: bool = True,
            **kwargs,
        ) -> None:
        if "skipinit" in kwargs:
            return
        nds: npt.NDArray[np.float64]
        if not copy_nodes and not isinstance(x, np.ndarray):
            if isinstance(x, np.ndarray):
                if x.dtype != np.float64:
                    raise ValueError(
                        f"Spline was passed {copy_nodes=}, but nodes were "
                        "not an array of float64.")
                nds = x
            else:
                raise TypeError(
                    f"Spline was passed {copy_nodes=}, but nodes were not"
                    f" a numpy array (instead {type(x).__name__})."
                    )
        else:
            nds = np.array(x, np.float64)
        if len(nds.shape) != 1 or nds.dtype != np.float64:
                raise ValueError(
                    f"Spline was passed {copy_nodes=}, but nodes were "
                    "not 1D array."
                    )
        try:
            if bc_left is None:
                bc_left = SplineBC(k1=0.0, k2=1.0, v=0.0)
            if bc_right is None:
                bc_right = SplineBC(k1=0.0, k2=1.0, v=0.0)
            coeffs = _hermite(nds, bc_left.as_tuple(), bc_right.as_tuple())
        except Exception as e:
            raise RuntimeError("Could not compute interpolation coefficients") from e
        object.__setattr__(self, "nodes", nds)
        object.__setattr__(self, "derivatives", coeffs)

    @property
    def n(self) -> int:
        """Return number of nodes in the spline."""
        return int(self.nodes.shape[0])

    def __call__(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return spline interpolation at integer nodes.
        
        Parameters
        ----------
        t : array_like
            Positions where the spline should be evaluated. Note that
            ``HermiteSpline(i) = HermiteSpline.nodes[i]`` for integer ``i``.

        Returns
        -------
        array
            Array of interpolated values at positions specified by ``t``.
        """
        frac, inte = np.modf(t)
        inte = inte.astype(np.uint64)
        over = (self.n - 1== inte) & (frac == 0)
        inte[over] = self.n - 2
        frac[over] = 1.0
        h1 = 2 * frac**3 - 3 * frac**2 + 1
        h2 = -2 * frac**3 + 3 * frac**2
        h3 = frac**3 - 2 * frac**2 + frac
        h4 = frac**3 - frac**2
        y = (h1 * self.nodes[inte] + h2 * self.nodes[inte + 1] +
             h3 * self.derivatives[inte] + h4 * self.derivatives[inte + 1])
        return y
    
    def subsection(
            self,
            ibegin: int,
            iend: int,
            include_endpoint: bool = False
        ) -> HermiteSpline:
        """Returns a spline, which is a subsection of the current.
        
        Parameters
        ----------
        ibegin : int
            Index of the first point to include.
        iend : int
            Index of the last point.
        include_endpoint : bool, default: False
            Should the last point be included in the spline.

        Returns
        -------
        HermiteSpline
            Subset of the spline between the chosen points.
        """
        spline = HermiteSpline([], skipinit=None)

        if include_endpoint:
            iend += 1
        object.__setattr__(spline, "nodes", self.nodes[ibegin:iend])
        object.__setattr__(spline, "derivatives", self.derivatives[ibegin:iend])
        return spline




        



