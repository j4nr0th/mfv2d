"""Types for describint the system and differential forms and system."""

from collections.abc import Callable, Iterator
from typing import Any, Self, SupportsIndex

from mfv2d._mfv2d import _ElementFormSpecification
from mfv2d.kform import KEquation, KForm, KFormUnknown, KWeight, UnknownFormOrder


class ElementFormSpecification(_ElementFormSpecification):
    """Specifications of element forms on an element.

    Parameters
    ----------
    *specs : tuple of KFormUnknowns
        Specifications for differential forms on the element. Each form must be
        unique.
    """

    def __new__(cls, *forms: KFormUnknown) -> Self:
        """Create a new form specification."""
        specs = tuple((form.label, form.order) for form in forms)
        return super().__new__(cls, *specs)

    def __getitem__(self, idx: SupportsIndex) -> tuple[str, UnknownFormOrder]:
        """Get the entry at the specified index."""
        label, order = super().__getitem__(idx)
        return (label, UnknownFormOrder(order))

    def get_form(self, idx: SupportsIndex, /) -> KFormUnknown:
        """Get the entry at the specified index, but converted to a form."""
        label, order = self[idx]
        return KFormUnknown(label, UnknownFormOrder(order))

    def __iter__(self) -> Iterator[tuple[str, UnknownFormOrder]]:
        """Iterate over labels and orders of forms specified."""
        iterator = super().__iter__()
        for label, order in iterator:
            yield (label, UnknownFormOrder(order))

    def iter_forms(self) -> Iterator[KFormUnknown]:
        """Iterate over forms in the specifications."""
        for label, order in self:
            yield KFormUnknown(label, order)

    def __contains__(self, item: tuple[str, int] | KFormUnknown) -> bool:
        """Check if the item is contained in the specifications."""
        if isinstance(item, KFormUnknown):
            return super().__contains__((item.label, item.order.value))

        return super().__contains__(item)

    def index(self, value: tuple[str, int] | KFormUnknown) -> int:
        """Return the index of the form with the given label and order in the specs.

        Parameters
        ----------
        value : tuple of (str, int) or KFormUnknown
            Label and index of the form, or the form itself.

        Returns
        -------
        int
            Index of the form in the specification.
        """
        if isinstance(value, KFormUnknown):
            return super().index((value.label, value.order.value))
        return super().index(value)

    def __eq__(self, other) -> bool:
        """Check if the other is identical to itself."""
        if not isinstance(other, ElementFormSpecification):
            return NotImplemented

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self[i] != other[i]:
                return False

        return True


class KFormSystem:
    """System of equations of differential forms, which are optionally sorted.

    This is a collection of equations, which fully describe a problem to be solved for
    the degrees of freedom of differential forms.

    Parameters
    ----------
    *equations : KFormEquation
        Equations which are to be used.
    sorting : (KForm) -> Any, optional
        Callable passed to the :func:`sorted` builtin to sort the primal forms. This
        corresponds to sorting the columns of the system matrix.
    """

    unknown_forms: ElementFormSpecification
    equations: tuple[KEquation, ...]

    def __init__(
        self,
        *equations: KEquation,
        sorting: Callable[[KForm], Any] | None = None,
    ) -> None:
        weights: list[KWeight] = []
        equation_list: list[KEquation] = []
        for ie, equation in enumerate(equations):
            weight = equation.weight
            if weight in weights:
                raise ValueError(
                    f"Weight form is not unique to the equation {ie}, as it already"
                    f" appears in equation {weights.index(weight)}."
                )
            weights.append(weight)
            equation_list.append(equation)

        if sorting is not None:
            self.weight_forms = tuple(sorted(weights, key=sorting))
        else:
            self.weight_forms = tuple(weights)

        self.unknown_forms = ElementFormSpecification(
            *(w.base_form for w in self.weight_forms)
        )

        self.equations = tuple(equation_list[self.weight_forms.index(w)] for w in weights)
