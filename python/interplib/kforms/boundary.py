"""File containing implementation of boundary conditions."""

from collections.abc import Mapping
from dataclasses import dataclass

from interplib.kforms.kform import KFormPrimal


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
    forms: dict[KFormPrimal, float]

    def __init__(self, forms: Mapping[KFormPrimal, float], value: float) -> None:
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

    form: KFormPrimal
    value: float

    def __post_init__(self) -> None:
        """Check that the paratmeters were correct."""
        if self.form.order != 1:
            raise ValueError("Only 1-forms can used for boundary conditions.")
