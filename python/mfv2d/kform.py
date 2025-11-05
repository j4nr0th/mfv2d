"""Types that allow for defining relations between forms.

These are not directly used for computations, but rather serve as a way to express
the relations between forms by taking advantage of Python's operator overloading.
For actual evaluation, these must be translated into stack machine codes using the
:mod:`mfv2d.eval` module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, TypeAliasType, overload

import numpy as np
import numpy.typing as npt

Function2D = TypeAliasType(
    "Function2D",
    Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike],
)
"""Type of a function taking two inputs and returning an array-like."""


class UnknownFormOrder(IntEnum):
    """Orders of unknown differential forms.

    This enum is intended to replace just passing integers for the
    order of forms, since this is easier to catch by type-checkers.
    """

    FORM_ORDER_0 = 1
    FORM_ORDER_1 = 2
    FORM_ORDER_2 = 3

    def full_unknown_count(self, order_1: int, order_2: int) -> int:
        """Return the total number of DoFs based on orders for a (full) leaf element."""
        if self == UnknownFormOrder.FORM_ORDER_0:
            return (order_1 + 1) * (order_2 + 1)
        if self == UnknownFormOrder.FORM_ORDER_1:
            return order_1 * (order_2 + 1) + (order_1 + 1) * order_2
        if self == UnknownFormOrder.FORM_ORDER_2:
            return order_1 * order_2

        raise ValueError

    @property
    def dual(self) -> UnknownFormOrder:
        """Return what the dual of the form is."""
        return UnknownFormOrder(2 - (self.value - 2))


@dataclass(frozen=True)
class Term:
    """Represents a term in the k-form expressions.

    This type contains the most basic functionality and is mainly intended to help with
    type hints.

    Parameters
    ----------
    label : str
        How to identify the term.
    """

    label: str

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return self.label


@dataclass(frozen=True)
class KForm(Term):
    """Differential K form.

    It is described by and order and identifier, that is used to print it.
    It offers the following overloaded operations:

    - ``*`` with another :class:`KForm` results in :class:`KInnerProduct`
    - ``*`` with a callable results in :class:`KInteriorProduct`
    - ``~`` will result in a :class:`KHodge` of itself

    Its exterior derivative is also availabel through the :meth:`KForm.derivative`
    method.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    """

    order: UnknownFormOrder
    label: str

    def __post_init__(self) -> None:
        """Check that order is correctly set."""
        object.__setattr__(self, "order", UnknownFormOrder(self.order))

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order.value - 1})"

    @overload
    def __mul__(self, other: KForm, /) -> KInnerProduct: ...
    @overload
    def __mul__(self, other: Function2D, /) -> KInteriorProduct: ...
    def __mul__(self, other: KForm | Function2D, /) -> KInnerProduct | KInteriorProduct:
        """Inner product with a weight."""
        if isinstance(other, KForm):
            return KInnerProduct(other, self)
        if callable(other):
            return KInteriorProduct(
                f"i_{{{other.__name__}}}({self.label})",
                UnknownFormOrder(self.order.value - 1),
                self,
                other,
            )
        return NotImplemented

    @overload
    def __rmul__(self, other: KForm, /) -> KInnerProduct: ...
    @overload
    def __rmul__(self, other: Function2D, /) -> KInteriorProduct: ...
    def __rmul__(self, other: KForm | Function2D, /) -> KInnerProduct | KInteriorProduct:
        """Inner product with a weight."""
        return self.__mul__(other)

    def __invert__(self) -> KHodge:
        """Hodge the k-form."""
        return KHodge(self)

    @property
    def derivative(self) -> KFormDerivative:
        """Derivative of the form."""
        return KFormDerivative(self)

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        raise NotImplementedError

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        raise NotImplementedError

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Order in primal basis."""
        raise NotImplementedError

    @property
    def core_form(self) -> KWeight | KFormUnknown:
        """Most basic form, be it unknown or weight."""
        raise NotImplementedError

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        raise NotImplementedError


@dataclass(frozen=True)
class KFormUnknown(KForm):
    """Differential form which is to be computed.

    Parameters
    ----------
    dual : bool, default: False
        Is the form represented by the dual or primal basis.
    """

    dual: bool = False

    def __xor__(self, other: KFormUnknown | KHodge):
        """Return a non-linear interior product term."""
        return KInteriorProductNonlinear(
            f"i_({self.label})({other.label})",
            UnknownFormOrder(other.order - 1),
            other,
            self,
        )

    @property
    def weight(self) -> KWeight:
        """Create a weight based on this form."""
        return KWeight(self.label, self.order, self)

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return False

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.order.dual

    @property
    def is_primal(self) -> bool:
        """Check if form is primal or dual."""
        return not self.dual

    @property
    def core_form(self) -> KFormUnknown:
        """Most basic form, be it unknown or weight."""
        return self

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return True


@dataclass(frozen=True, eq=False)
class KWeight(KForm):
    """Differential K form represented with the dual basis.

    Provides operators for forming element and boundary projections
    throught the ``@`` and ``^`` operators respectively.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    base_form : KForm
        Form, which the weight is based on.
    """

    base_form: KFormUnknown

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order.value - 1}*)"

    @overload  # type: ignore[override]
    def __mul__(self, other: KForm, /) -> KInnerProduct: ...

    @overload
    def __mul__(self, other: Callable | Literal[0], /) -> KElementProjection: ...

    def __mul__(
        self, other: KForm | Callable | Literal[0], /
    ) -> KInnerProduct | KElementProjection:
        """Inner product with a weight."""
        if isinstance(other, KForm):
            return KInnerProduct(other, self)
        if callable(other):
            return KElementProjection(f"<{self.label}, {other.__name__}>", self, other)
        if other == 0:
            return KElementProjection("0", self, None)
        return NotImplemented

    @overload  # type: ignore[override]
    def __rmul__(self, other: KForm, /) -> KInnerProduct: ...

    @overload
    def __rmul__(self, other: Callable | Literal[0], /) -> KElementProjection: ...

    def __rmul__(
        self, other: KForm | Callable | Literal[0], /
    ) -> KInnerProduct | KElementProjection:
        """Inner product with a weight."""
        return self.__mul__(other)

    def __xor__(self, other: Callable) -> KBoundaryProjection:
        """Create boundary projection for the right hand side."""
        if callable(other):
            return KBoundaryProjection(f"<{self.label}, {other.__name__}>", self, other)
        return NotImplemented

    def __matmul__(self, other: Callable | Literal[0], /) -> KElementProjection:
        """Create projection for the right hand side."""
        if isinstance(other, int) and other == 0:
            return KElementProjection("0", self, None)
        if callable(other):
            return KElementProjection(f"<{self.label}, {other.__name__}>", self, other)
        return NotImplemented

    @property
    def weight(self) -> KWeight:
        """Return itself."""
        return self

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return True

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.order.dual

    @property
    def is_primal(self) -> bool:
        """Check if form is primal or dual."""
        return self.base_form.is_primal

    @property
    def core_form(self) -> KWeight:
        """Most basic form, be it unknown or weight."""
        return self

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return True


@dataclass(frozen=True, init=False)
class KHodge(KForm):
    r"""Hodge represents a transformation from primal to dual basis.

    A continuous Hodge :math:`\star` is defined as a mapping, which transfers a k-form
    on an n-dimensional manifold into a (n-k)-form:

    .. math::

        \star: \Lambda^{(k)} \leftarrow \Lambda^{(n - k)}

    The discrete version of the Hodge also maps a k-form onto a (n-k) form, but with a
    very specific choice of basis. If a polynomial can be written in terms of primal basis
    matrix :math:`\boldsymbol{\Psi}^{(k)}` and degree-of-freedom vector
    :math:`\vec{p}^{(k)}`, which are defined by equations :eq:`khodge-psi` and
    :eq:`khodge-p`, then the duals are defined by :math:`khodge-dual`. This allows for
    the resulting system to be sparser, at the cost of having to obtain the primal values
    in post-processing.

    .. math::
        :label: khodge-psi

        \boldsymbol{\Psi}^{(k)} = \begin{bmatrix} \psi_0 (\vec{\xi}) & \cdots & \psi_n
        (\vec{\xi}) \end{bmatrix}


    .. math::
        :label: khodge-p

        \vec{p}^{(k)} = \begin{bmatrix} p^0 \\ \vdots \\ p^n \end{bmatrix}


    .. math::
        :label: khodge-dual

        \begin{align}
            \tilde{\boldsymbol{\Psi}}^{(n - k)} = \boldsymbol{\Psi}^{(k)}
            \left(\mathbb{M}^{(k)}\right)^{-1} &&
            \vec{\tilde{p}}^{(n - k)} = \mathbb{M}^{(k)} \vec{p}^{(k)}
        \end{align}

    Parameters
    ----------
    form : KForm
        Form which the Hodge should be applied to. Note that applying the Hodge twice
        is the identity operation.
    """

    base_form: KForm

    def __init__(self, form: KForm) -> None:
        super().__init__("~" + form.label, form.order.dual)
        object.__setattr__(self, "base_form", form)

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        return not self.base_form.is_primal

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.order.dual

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return self.base_form.is_weight

    @property
    def core_form(self) -> KWeight | KFormUnknown:
        """Most basic form, be it unknown or weight."""
        return self.base_form.core_form

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return self.base_form.is_linear


@dataclass(init=False, frozen=True, eq=False)
class KFormDerivative(KForm):
    r"""Exterior derivative of a form.

    An exterior derivative maps a differential k-form into a (k + 1) form:

    .. math::

        \mathrm{d}: p^{(k)} \in \Lambda^{(k)}(\mathcal{M}) \leftarrow q^{(k + 1)} \in
        \Lambda^{(k + 1)}(\mathcal{M})

    This operation is expressed in terms of a so called incidence matrix
    :math:`\mathbb{E}^{(k, k + 1)}`, which maps degrees of freedom from basis of k-forms
    to those of (k + 1)-forms

    Note that applying the operator :math:`\mathrm{d}` twice will always result in a
    form which is zero everywhere:

    .. math::

        \mathrm{d}\left( \mathrm{d} p^{(k)} \right) = 0

    Parameters
    ----------
    form : KForm
        The form of which the derivative is to be taken.
    """

    form: KForm

    def __init__(self, form: KForm) -> None:
        object.__setattr__(self, "form", form)
        super().__init__("d" + form.label, UnknownFormOrder(form.order.value + 1))

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        return self.form.is_primal

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return self.form.is_weight

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Order in primal basis."""
        if self.form.is_primal:
            return self.order
        else:
            return UnknownFormOrder(self.form.primal_order.value - 1)

    @property
    def core_form(self) -> KWeight | KFormUnknown:
        """Most basic form, be it unknown or weight."""
        return self.form.core_form

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return self.form.is_linear


@dataclass(frozen=True, eq=False)
class KInteriorProduct(KForm):
    """Represents an interior product of a K-form with a tangent vector field."""

    form: KForm
    vector_field: Function2D

    def __post_init__(self) -> None:
        """Enforce the conditions for allowing interior product."""
        # The form can not be a zero-form
        if self.form.order == UnknownFormOrder.FORM_ORDER_0:
            raise ValueError("Interior product can not be applied to a 0-form.")

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal or not."""
        return self.form.is_primal

    @property
    def is_weight(self) -> bool:
        """Check if it is a weight form."""
        return self.form.is_weight

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Return the order of the primal."""
        if self.form.is_primal:
            return UnknownFormOrder(self.form.primal_order.value - 1)
        return UnknownFormOrder(self.form.primal_order.value + 1)

    @property
    def core_form(self) -> KWeight | KFormUnknown:
        """Most basic form, be it unknown or weight."""
        return self.form.core_form

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return self.form.is_linear


@dataclass(frozen=True, eq=False)
class KInteriorProductNonlinear(KForm):
    """Represents an interior product of a K-form with a lowered 1-form.

    In two dimentions there are at total of four different types of interior products:

    - primal 1-form
    - primal 2-form
    - dual 1-form
    - dual 2-form

    These all correspond to a different operation:

    - primal 1-form: scalar cross product
    - primal 2-form: multiplication with a vector field
    - dual 1-form: dot product
    - dual 2-form: vector cross product
    """

    form: KFormUnknown | KHodge
    form_field: KFormUnknown

    def __post_init__(self) -> None:
        """Enforce the conditions for allowing interior product."""
        # The form cannot be a zero-form
        if not (
            type(self.form) is KFormUnknown
            or (type(self.form) is KHodge and type(self.form.base_form) is KFormUnknown)
        ):
            raise TypeError(
                "Form with which the interior product is taken can be only an unknown or"
                f" its Hodge (instead it was {type(self.form)})."
            )

        if type(self.form_field) is not KFormUnknown:
            raise TypeError(
                "Form field must be an unknown 1-form (instead it was"
                f" {type(self.form_field)})."
            )

        if self.form.order == UnknownFormOrder.FORM_ORDER_0:
            raise ValueError("Interior product can not be applied to a 0-form.")
        if self.form_field.order != UnknownFormOrder.FORM_ORDER_1:
            raise ValueError(
                "Interior product requires the other form to be a"
                f" 1-form, it was instead a {self.form_field.order.value - 1}-form."
            )

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal or not."""
        return self.form.is_primal

    @property
    def is_weight(self) -> bool:
        """Check if it is a weight form."""
        return self.form.is_weight

    @property
    def primal_order(self) -> UnknownFormOrder:
        """Return the order of the primal."""
        if self.form.is_primal:
            return UnknownFormOrder(self.form.primal_order.value - 1)
        return UnknownFormOrder(self.form.primal_order.value + 1)

    @property
    def core_form(self) -> KWeight | KFormUnknown:
        """Most basic form, be it unknown or weight."""
        return self.form.core_form

    @property
    def is_linear(self) -> bool:
        """Check if the form is linear."""
        return False


@dataclass(frozen=True, eq=False)
class TermEvaluatable(Term):
    """Terms which can be evaluated as blocks of the system matrix.

    This is a base class for all terms which can be evaluated as blocks of the system.
    """

    weight: KWeight

    def __post_init__(self) -> None:
        """Check that the weight is indeed a weight."""
        if not self.weight.is_weight:
            raise TypeError(f"The weight form {self.weight} is not actually a weight.")

    def __add__(self, other: TermEvaluatable, /) -> KSum:
        """Add the term to another."""
        if isinstance(other, TermEvaluatable):
            return KSum((1.0, self), (1.0, other))
        return NotImplemented

    def __radd__(self, other: TermEvaluatable, /) -> KSum:
        """Add the term to another."""
        return self.__add__(other)

    def __sub__(self, other: TermEvaluatable, /) -> KSum:
        """Subtract the term from another."""
        if isinstance(other, TermEvaluatable):
            return KSum((1.0, self), (-1.0, other))
        return NotImplemented

    def __rsub__(self, other: TermEvaluatable, /) -> KSum:
        """Subtract the combination to another."""
        if isinstance(other, TermEvaluatable):
            return KSum((1.0, other), (-1.0, self))
        return NotImplemented

    def __mul__(self, other: float | int, /) -> KSum:
        """Multiply by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KSum((v, self))

    def __rmul__(self, other: float | int, /) -> KSum:
        """Multiply by a constant."""
        return self.__mul__(other)

    def __truediv__(self, other: float | int, /) -> KSum:
        """Divide by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KSum((1 / v, self))

    def __neg__(self) -> KSum:
        """Negate the term."""
        return KSum((-1, self))

    @overload
    def __eq__(self, other: TermEvaluatable | None | Literal[0], /) -> KEquation: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(self, other: TermEvaluatable | None | Literal[0], /) -> KEquation | bool:
        """Check equality or form an equation."""
        if isinstance(other, TermEvaluatable):
            return KEquation(KSum((1.0, self)), KSum((1.0, other)))

        try:
            if other is None or float(other) == 0:
                return KEquation(
                    KSum((1.0, self)),
                    KSum((1.0, KElementProjection("0", self.weight, None))),
                )
        except Exception:
            pass

        return self is other

    @property
    def unknowns(self) -> tuple[KFormUnknown, ...]:
        """Return all unknowns in the term."""
        raise NotImplementedError

    @property
    def vector_fields(self) -> tuple[Function2D | KFormUnknown, ...]:
        """Return all vector fields."""
        raise NotImplementedError


def _extract_unknowns(form: KForm) -> list[KFormUnknown]:
    """Extract unknown forms from the form, otherwise raises type error."""
    if type(form) is KFormUnknown:
        return [form]

    # if type(form) is KTimeDerivative:
    #     return [form.base_form]

    if type(form) is KFormDerivative:
        return _extract_unknowns(form.form)

    if type(form) is KHodge:
        return _extract_unknowns(form.base_form)

    if type(form) is KInteriorProduct:
        return _extract_unknowns(form.form)

    if type(form) is KInteriorProductNonlinear:
        return _extract_unknowns(form.form) + [
            form.form_field,
        ]

    raise TypeError(f"Unknown forms can not be extracted from the form {form}.")


@dataclass(init=False, frozen=True, eq=False)
class KInnerProduct(TermEvaluatable):
    r"""Inner product of a primal and dual form.

    An inner product must be taken with primal and dual forms of the same k-order.
    The discrete version of an inner product of two k-forms is expressed as a
    discrete inner product on the mass matrix:

    .. math::

        \left< p^{(k)}, q^{(k)} \right> = \int_{\mathcal{K}} p^{(k)} \wedge \star q^{(k)}
        = \vec{p}^T \mathbb{M}^k \vec{q}
    """

    unknown_form: KForm
    weight_form: KForm

    def __init__(self, a: KForm, b: KForm, /) -> None:
        if a.is_weight == b.is_weight:
            raise TypeError(
                "Inner product can only be taken between a weight and an unknown k-form."
            )
        if a.is_weight:
            weight = a
            unknown = b
        else:
            weight = b
            unknown = a
        w_order = weight.primal_order
        u_order = unknown.primal_order
        if w_order != u_order:
            raise ValueError(
                f"The K forms are not of the same (primal) order ({w_order} vs {u_order})"
            )

        object.__setattr__(self, "unknown_form", unknown)
        object.__setattr__(self, "weight_form", weight)
        w = weight.core_form
        assert type(w) is KWeight
        super().__init__(f"<{weight.label}, {unknown.label}>", w)

    @property
    def unknowns(self) -> tuple[KFormUnknown, ...]:
        """Return all unknowns in the sum."""
        return tuple(_extract_unknowns(self.unknown_form))


@dataclass(init=False, frozen=True, eq=False)
class KSum(TermEvaluatable):
    """Linear combination of differential form inner products.

    Parameters
    ----------
    *pairs : tuple of float and KFormInnerProduct
        Coefficients and the inner products.
    """

    pairs: tuple[tuple[float, KExplicit | KInnerProduct], ...]

    def __init__(self, *pairs: tuple[float, TermEvaluatable]) -> None:
        if len(pairs) < 1:
            raise TypeError("Can not create a sum object with no members.")

        weight: KWeight = pairs[0][1].weight
        new_pairs: list[tuple[float, KExplicit | KInnerProduct]] = list()
        for coeff, term in pairs:
            if weight != term.weight:
                raise ValueError("Can not sum terms with varying weight forms")

            if type(term) is KSum:
                new_pairs.extend([(coeff * c, t) for c, t in term.pairs])

            else:
                if not isinstance(term, KExplicit) and type(term) is not KInnerProduct:
                    raise TypeError(
                        "Terms can only be sums, explicit, or inner products."
                    )

                new_pairs.append((coeff, term))
        del pairs

        object.__setattr__(self, "pairs", tuple(new_pairs))
        label = "(" + "+".join(ip.label for _, ip in self.pairs) + ")"
        super().__init__(label, weight)

    @property
    def unknowns(self) -> tuple[KFormUnknown, ...]:
        """Return all unknowns in the sum."""
        out: set[KFormUnknown] = set()

        for _, p in self.pairs:
            out |= set(p.unknowns)

        return tuple(out)

    @property
    def vector_fields(self) -> tuple[Function2D | KFormUnknown, ...]:
        """Return all vector fields in the sum."""
        out: set[Function2D | KFormUnknown] = set()

        for _, p in self.pairs:
            out |= set(p.vector_fields)

        return tuple(out)

    @property
    def explicit_terms(self) -> tuple[tuple[float, KExplicit], ...]:
        """Get all explicit terms."""
        return tuple((k, p) for k, p in self.pairs if isinstance(p, KExplicit))

    @property
    def implicit_terms(self) -> tuple[tuple[float, TermEvaluatable], ...]:
        """Get all implicit terms."""
        return tuple((k, p) for k, p in self.pairs if not isinstance(p, KExplicit))

    def split_terms_linear_nonlinear(self) -> tuple[KSum | None, KSum | None]:
        """Split the sum into linear implicit and non-linear implicit terms.

        Returns
        -------
        KSum
            All linear terms. If there are no linear implicit terms, it is None instead.

        KSum
            All non-linear terms. If there are no non-linear implicit terms, it is None
            instead.
        """
        linear: list[tuple[float, KInnerProduct]] = list()
        nonlin: list[tuple[float, KInnerProduct]] = list()

        for c, v in self.pairs:
            if isinstance(v, KExplicit):
                continue

            assert type(v) is KInnerProduct
            if v.unknown_form.is_linear and v.weight_form.is_linear:
                linear.append((c, v))
            else:
                nonlin.append((c, v))

        return (
            KSum(*linear) if len(linear) else None,
            KSum(*nonlin) if len(nonlin) else None,
        )


@dataclass(frozen=True)
class KExplicit(TermEvaluatable):
    """Base class for explicit terms.

    This type just implements some common functionality.
    """

    weight: KWeight
    func: Callable | None = None

    @property
    def unknowns(self) -> tuple[KFormUnknown, ...]:
        """Return all unknowns (there are none)."""
        return tuple()

    @property
    def vector_fields(self) -> tuple[Function2D | KFormUnknown, ...]:
        """Return all vector fields (there are none)."""
        return tuple()


@dataclass(frozen=True)
class KElementProjection(KExplicit):
    r"""Element integral of the function with the basis.

    This is used to form the right side of the systems of equations coming from a forcing
    function.

    Parameters
    ----------
    weight : KWeight
        Weight form used.
    func : tuple[str, Callable], optional
        The function to use, specified by a name and the callable to use. If it is not
        specified or given as ``None``, then :math:`f = 0`.
    """


@dataclass(frozen=True)
class KBoundaryProjection(KExplicit):
    r"""Boundary integral of a forcing.

    This is intended to be used to define boundary conditions. Given that
    the function to be projected is denoted by :math:`f` and the weight function
    is denoted by :math:`w`, this term represents the integral

    .. math::

        \int_{\partial \Omega} f^{(k)} \wedge \star w^{(k + 1)}

    Such terms typically arise from weak boundary conditions.
    """


@dataclass(frozen=True)
class KEquation:
    """Equation of differential forms and weights, consisting of a left and a right side.

    The equation represents an equation where all the implicit terms are on the left side
    and all explicit ones are on the right side.

    Parameters
    ----------
    left : KSum or KInnerProduct
        Term representing the implicit part of the equation with all the unknown forms.
    right : KFormProjection
        The form representing the explicit part of the equation.
    """

    left: KSum
    right: KSum

    def __post_init__(self) -> None:
        """Check that terms are done properly."""
        if len(self.left.explicit_terms):
            raise ValueError(
                "Explicit terms may not appear on the left side of the equation."
            )
        if self.left.weight != self.right.weight:
            raise ValueError(
                "Left and right side of the equation must use the exact same weight"
                " function."
            )

    @property
    def weight(self) -> KWeight:
        """Return the weight used by both sides."""
        return self.left.weight
