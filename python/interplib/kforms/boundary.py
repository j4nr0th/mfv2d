"""File containing implementation of boundary conditions."""

from collections.abc import Mapping
from dataclasses import dataclass

from interplib.kforms.kform import KFormPrimal


@dataclass(init=False, frozen=True)
class BoundaryCondition1DStrong:
    """Represents a strong boundary condition on 1D boundary.

    Parameters
    ----------
    forms :
        Form on which the boundary condition is prescribed.
    value : float
        Value to which to force the degree of freedom to.
    """

    value: float
    forms: dict[KFormPrimal, float]

    def __init__(self, forms: Mapping[KFormPrimal, float], value: float) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "forms", {form: forms[form] for form in forms})
