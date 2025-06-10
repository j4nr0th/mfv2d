"""File containing implementation of boundary conditions."""

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
    """Boundary condition for a 2D problem."""

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
