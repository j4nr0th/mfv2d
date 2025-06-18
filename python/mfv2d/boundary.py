"""File containing implementation of boundary conditions.

For now the only one really used anywhere is the :class:`BoundaryCondition2DSteady`,
but perhaps in the future, for unsteady problems unsteady BCs may be used.

So far the intention is to use these to prescribe strong Dirichelt or Neumann conditions,
while weak boundary conditions are introduced in the equation as boundary integral terms.
That simplifies evaulation makes tracking all of these much simpler.

So far there is no plans to introduce any other types of strong boundary conditions,
though based on what is already supported, it would not be too much of a stretch to
add support for prescribing arbitrary relations on the boundary.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mfv2d.kform import Function2D, KFormUnknown


@dataclass(frozen=True, init=False)
class BoundaryCondition2D:
    """Base class for 2D boundary conditions."""

    form: KFormUnknown
    indices: npt.NDArray[np.uint64]

    def __init__(self, form: KFormUnknown, indices: npt.ArrayLike) -> None:
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "indices", np.array(indices, np.uint64))
        if self.indices.ndim != 1:
            raise ValueError("Indices array is not a 1D array.")
        object.__setattr__(self, "indices", np.unique(self.indices))


@dataclass(frozen=True)
class BoundaryCondition2DSteady(BoundaryCondition2D):
    """Boundary condition for a 2D problem with no time dependence.

    These boundary conditions specifiy values of differential forms on the
    the given indices directly. These are enforced "strongly" by adding
    a Lagrange multiplier to the system.

    Parameters
    ----------
    form : KFormUnknown
        Form for which the value is to be prescribed.

    indices : array_like
        One dimensional array of edges on which this is prescribed.
        TODO: check if 0-based on 1-based

    func : (array, array) -> array_like
        Function that can be evaluated to obtain values of differential forms
        at those points. For 1-froms it should return an array_like with an
        extra last dimension, which contains the two vector components.
    """

    func: Function2D

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Function2D,
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)


@dataclass(init=False, frozen=True)
class BoundaryCondition2DUnsteady(BoundaryCondition2D):
    """Unsteady boundary condition for a 2D problem."""

    func: Function2D

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]],
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)
