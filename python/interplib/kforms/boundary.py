"""File containing implementation of boundary conditions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from interplib.kforms.kform import KFormUnknown


@dataclass(init=False, frozen=True)
class BoundaryCondition1D:
    """Represents a boundary condition for a 1D problem."""

    pass


@dataclass(init=False, frozen=True)
class BoundaryCondition1DStrong(BoundaryCondition1D):
    """Represents a strong boundary condition on 1D boundary.

    Parameters
    ----------
    forms :
        Forms on which the boundary condition is prescribed.
    value : float
        Value to which to force the degree of freedom to.
    """

    value: float
    forms: dict[KFormUnknown, float]

    def __init__(self, forms: Mapping[KFormUnknown, float], value: float) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "forms", {form: forms[form] for form in forms})
        for form in self.forms:
            if form.order != 0:
                raise ValueError(
                    "Only 0-forms can be used for strong boundary conditions."
                )


@dataclass(frozen=True)
class BoundaryCondition1DWeak(BoundaryCondition1D):
    """Represents a weak boundary condition.

    Parameters
    ----------
    form :
        Form on which the boundary condition is prescribed.
    value : float
        Value of the form on the boundary.
    """

    form: KFormUnknown
    value: float

    def __post_init__(self) -> None:
        """Check that the paratmeters were correct."""
        if self.form.order != 1:
            raise ValueError("Only 1-forms can used for boundary conditions.")


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

    func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]],
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)


@dataclass(init=False, frozen=True)
class BoundaryCondition2DUnsteady(BoundaryCondition2D):
    """Unsteady boundary condition for a 2D problem."""

    func: Callable[[float, npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]

    def __init__(
        self,
        form: KFormUnknown,
        indices: npt.ArrayLike,
        func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]],
    ) -> None:
        super().__init__(form, indices)
        object.__setattr__(self, "func", func)
