"""Dealing with K-forms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt

from interplib._mimetic import Manifold

VectorFieldFunction = Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class Term:
    """Represents a term in the k-form expressions.

    This type contains the most basic functionality and is mainly intended to help with
    type hints.
    """

    manifold: Manifold
    label: str

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return self.label


@dataclass(frozen=True)
class KForm(Term):
    """Differential K form.

    It is described by and order and identifier, that is used to print it.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    """

    order: int
    label: str

    def __post_init__(self) -> None:
        """Check that the order of the form is not too high."""
        if self.manifold.dimension < self.order:
            raise ValueError(
                f"Can not create a {self.order}-form on a manifold of dimension"
                f" {self.manifold.dimension}"
            )

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order})"

    @overload
    def __mul__(self, other: KForm, /) -> KInnerProduct: ...
    @overload
    def __mul__(self, other: VectorFieldFunction, /) -> KInteriorProduct: ...
    def __mul__(
        self, other: KForm | VectorFieldFunction, /
    ) -> KInnerProduct | KInteriorProduct:
        """Inner product with a weight."""
        if isinstance(other, KForm):
            return KInnerProduct(other, self)
        if callable(other):
            return KInteriorProduct(
                self.manifold,
                f"i_{{{other.__name__}}}({self.label})",
                self.order - 1,
                self,
                other,
            )
        return NotImplemented

    @overload
    def __rmul__(self, other: KForm, /) -> KInnerProduct: ...
    @overload
    def __rmul__(self, other: VectorFieldFunction, /) -> KInteriorProduct: ...
    def __rmul__(
        self, other: KForm | VectorFieldFunction, /
    ) -> KInnerProduct | KInteriorProduct:
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
    def primal_order(self) -> int:
        """Order in primal basis."""
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

    @property
    def weight(self) -> KWeight:
        """Create a weight based on this form."""
        return KWeight(self.manifold, self.label, self.order, self)

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return False

    @property
    def primal_order(self) -> int:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.manifold.dimension - self.order

    @property
    def is_primal(self) -> bool:
        """Check if form is primal or dual."""
        return not self.dual


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
    :math:`\vec{p}^{(k)}`, which are defined by :eq:`khodge-psi` and :eq:`khodge-p`,
    then the duals are defined by :math:`khodge-dual`. This allows for the resulting
    system to be sparser, at the cost of having to obtain the primal values in
    post-processing.

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
        super().__init__(
            form.manifold, "~" + form.label, form.manifold.dimension - form.order
        )
        object.__setattr__(self, "base_form", form)

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        return not self.base_form.is_primal

    @property
    def primal_order(self) -> int:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.manifold.dimension - self.order

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return self.base_form.is_weight


@dataclass(frozen=True, eq=False)
class KWeight(KForm):
    """Differential K form represented with the dual basis.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    base_form : KForm
        Form, which the weight is based on.
    """

    base_form: KForm

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order}*)"

    @property
    def derivative(self) -> KFormDerivative:
        """Derivative of the form."""
        return KFormDerivative(self)

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
            return KElementProjection(self, other)
        if other == 0:
            return KElementProjection(self, None)
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
            return KBoundaryProjection(self, other)
        return NotImplemented

    def __matmul__(self, other: Callable | Literal[0], /) -> KElementProjection:
        """Create projection for the right hand side."""
        if isinstance(other, int) and other == 0:
            return KElementProjection(self, None)
        if callable(other):
            return KElementProjection(self, other)
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
    def primal_order(self) -> int:
        """Order of the mass matrix which needs to be used."""
        if self.is_primal:
            return self.order
        return self.manifold.dimension - self.order

    @property
    def is_primal(self) -> bool:
        """Check if form is primal or dual."""
        return self.base_form.is_primal


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
        super().__init__(form.manifold, "d" + form.label, form.order + 1)

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        return self.form.is_primal

    @property
    def is_weight(self) -> bool:
        """Check if the form is a weight."""
        return self.form.is_weight

    @property
    def primal_order(self) -> int:
        """Order in primal basis."""
        if self.form.is_primal:
            return self.order
        else:
            return self.form.primal_order - 1


@dataclass(frozen=True, eq=False)
class KInteriorProduct(KForm):
    """Represents an interior product of a K-form with a tangent vector field."""

    form: KForm
    vector_field: VectorFieldFunction

    def __post_init__(self) -> None:
        """Enforce the conditions for allowing interior product."""
        # The form can not be a zero-form
        if self.form.order == 0:
            raise ValueError("Interior product can not be applied to a 0-form.")

        if not self.form.is_primal:
            raise ValueError("Can not apply interior product to a dual form.")

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal or not."""
        return False

    @property
    def is_weight(self) -> bool:
        """Check if it is a weight form."""
        return self.form.is_weight

    @property
    def primal_order(self) -> int:
        """Return the order of the primal."""
        return self.form.order - 1


@dataclass(init=False, frozen=True, eq=False)
class KInnerProduct(Term):
    r"""Inner product of a primal and dual form.

    An inner product must be taken with a primal and dual forms of the same k-order.
    The discrete version of an inner product of two k-forms is expressed as a
    discrete inner product on the mass matrix:

    .. math::

        \left< p^{(k)}, q^{(k)} \right> = \int\limts_{\mathcal{K}} p^{(k)} q^{(k)} =
        \vec{p}^T \mathbb{M}^k \vec{q}
    """

    weight: KForm
    function: KForm

    def __init__(self, a: KForm, b: KForm, /) -> None:
        if a.is_weight == b.is_weight:
            raise TypeError(
                "Inner product can only be taken between a weight and an unknown k-form."
            )
        if a.is_weight:
            weight = a
            function = b
        else:
            weight = b
            function = a
        wg_order = weight.primal_order
        fn_order = function.primal_order
        if wg_order != fn_order:
            raise ValueError(
                f"The K forms are not of the same order ({wg_order} vs {fn_order})"
            )
        if weight.manifold is not function.manifold:
            raise ValueError(
                "Inner product can only be taken between differential forms defined on "
                "the same manifold."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "function", function)
        super().__init__(weight.manifold, f"<{weight.label}, {function.label}>")

    def __add__(self, other: KInnerProduct | KSum, /) -> KSum:
        """Add the inner products together."""
        if isinstance(other, KInnerProduct):
            return KSum((1.0, self), (1.0, other))
        if isinstance(other, KSum):
            return KSum((1.0, self), *other.pairs)
        return NotImplemented

    def __sub__(self, other: KInnerProduct | KSum, /) -> KSum:
        """Subtract the inner products."""
        if isinstance(other, KInnerProduct):
            return KSum((1.0, self), (-1.0, other))
        if isinstance(other, KSum):
            return KSum((1.0, self), *((-c, f) for c, f in other.pairs))
        return NotImplemented

    def __mul__(self, other: float, /) -> KSum:
        """Multiply with a constant."""
        try:
            v = float(other)
            return KSum((v, self))
        except Exception:
            return NotImplemented

    def __rmul__(self, other: float, /) -> KSum:
        """Multiply with a constant."""
        return self.__mul__(other)

    def __neg__(self) -> KSum:
        """Negate the inner product."""
        return KSum((-1.0, self))

    @overload
    def __eq__(self, other: KProjection | KProjectionCombination, /) -> KEquaton: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(self, other: KProjection | KProjectionCombination, /) -> KEquaton | bool:
        """Check equality or form an equation."""
        if isinstance(other, KProjection):
            return KEquaton(self, KProjectionCombination(other.weight, (1.0, other)))
        if isinstance(other, KProjectionCombination):
            return KEquaton(self, other)
        return self is other


@dataclass(init=False, frozen=True, eq=False)
class KSum(Term):
    """Linear combination of differential form inner products.

    Parameters
    ----------
    *pairs : tuple of float and KFormInnerProduct
        Coefficients and the inner products.
    """

    # Check if order is even needed
    pairs: tuple[tuple[float, KInnerProduct], ...]

    def __init__(self, *pairs: tuple[float, KInnerProduct]) -> None:
        manifold: Manifold | None = None
        for _, ip in pairs:
            if manifold is None:
                manifold = ip.manifold
            elif manifold is not ip.manifold:
                raise ValueError("Can not sum inner products from different manifolds.")
        if len(pairs) < 1:
            raise TypeError("Can not create a sum object with no members.")
        assert manifold is not None
        object.__setattr__(self, "pairs", pairs)
        label = "(" + "+".join(ip.label for _, ip in self.pairs) + ")"
        super().__init__(manifold, label)

    def __add__(self, other: KSum | KInnerProduct, /) -> KSum:
        """Add two sums together."""
        if isinstance(other, KSum):
            return KSum(*self.pairs, *other.pairs)
        if isinstance(other, KInnerProduct):
            return KSum(*self.pairs, (1.0, other))
        return NotImplemented

    def __mul__(self, other: float, /) -> KSum:
        """Multiply with a constant."""
        try:
            v = float(other)
            return KSum(*tuple((v * c, ip) for c, ip in self.pairs))
        except Exception:
            return NotImplemented

    @overload
    def __eq__(
        self, other: KProjection | KProjectionCombination | int, /
    ) -> KEquaton: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(
        self, other: KProjection | KProjectionCombination | int, /
    ) -> KEquaton | bool:
        """Check equality or form an equation."""
        if isinstance(other, KProjection):
            return KEquaton(self, KProjectionCombination(other.weight, (1.0, other)))
        if isinstance(other, KProjectionCombination):
            return KEquaton(self, other)
        try:
            if float(other) == 0:
                _, ws, _, _ = _extract_forms(self.pairs[0][1].weight)
                w = tuple(w for w in ws)[0]
                return KEquaton(
                    self,
                    KProjectionCombination(
                        w,
                        (1.0, KElementProjection(w, None)),
                    ),
                )
        except Exception:
            pass
        return self is other


@dataclass(frozen=True)
class KProjection:
    """Base class for projections used on the right hand side of the equations."""

    weight: KWeight
    func: Callable | None = None

    def __add__(self, other: KProjection, /) -> KProjectionCombination:
        """Add the term to another."""
        if isinstance(other, KProjection):
            return KProjectionCombination(self.weight, (1.0, self), (1.0, other))
        return NotImplemented

    def __radd__(self, other: KProjection, /) -> KProjectionCombination:
        """Add the term to another."""
        return self.__add__(other)

    def __sub__(self, other: KProjection, /) -> KProjectionCombination:
        """Subtract the term from another."""
        if isinstance(other, KProjection):
            return KProjectionCombination(self.weight, (1.0, self), (-1.0, other))
        return NotImplemented

    def __rsub__(self, other: KProjection, /) -> KProjectionCombination:
        """Subtract the combination to another."""
        if isinstance(other, KProjection):
            return KProjectionCombination(self.weight, (1.0, other), (-1.0, self))
        return NotImplemented

    def __mul__(self, other: float | int, /) -> KProjectionCombination:
        """Multiply the projection by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KProjectionCombination(self.weight, (v, self))

    def __rmul__(self, other: float | int, /) -> KProjectionCombination:
        """Multiply the projection by a constant."""
        return self.__mul__(other)

    def __div__(self, other: float | int, /) -> KProjectionCombination:
        """Divide the projection by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KProjectionCombination(self.weight, (1 / v, self))

    def __neg__(self) -> KProjectionCombination:
        """Negate the combination."""
        return KProjectionCombination(self.weight, (-1, self))


@dataclass(frozen=True)
class KElementProjection(KProjection):
    r"""Element integral of the function with the basis.

    This is used to form the right side of the systems of equations coming from a forcing
    function.

    Parameters
    ----------
    weight : KWeight
        Weight form used.
    func : tuple[str, Callable], optional
        The function to use, specified by a name and the callable to use. If it not
        specified or given as ``None``, then :math:`f = 0`.
    """


@dataclass(frozen=True)
class KBoundaryProjection(KProjection):
    r"""Boundary integral of a forcing.

    This is intended to be used to define boundary conditions. Given that
    the function to be projected is denoted by :math:`f` and the weight function
    is denoted by :math:`w`, this term represents the integral

    .. math::

        \int_{\partial \Omega} f w {d \vec{\Gamma}}

    Such terms typically arise from weak boundary conditions.
    """


@dataclass(frozen=True, init=False)
class KProjectionCombination:
    """Combination of boundary and element projections."""

    pairs: tuple[tuple[float, KProjection]]
    weight: KWeight

    def __init__(self, weight: KWeight, *pairs: tuple[float, KProjection]) -> None:
        object.__setattr__(self, "pairs", tuple((float(a), b) for a, b in pairs))
        object.__setattr__(self, "weight", weight)
        for _, f in self.pairs:
            if self.weight != f.weight:
                raise ValueError(
                    "Can not combine projections with different weight functions"
                    f" (namely {self.weight} and {f.weight})."
                )

    def __add__(
        self, other: KProjectionCombination | KProjection, /
    ) -> KProjectionCombination:
        """Add the combination to another."""
        if isinstance(other, KProjectionCombination):
            return KProjectionCombination(self.weight, *self.pairs, *other.pairs)
        if isinstance(other, (KProjection)):
            return KProjectionCombination(self.weight, *self.pairs, (1.0, other))
        return NotImplemented

    def __radd__(
        self, other: KProjectionCombination | KProjection, /
    ) -> KProjectionCombination:
        """Add the combination to another."""
        return self.__add__(other)

    def __sub__(
        self, other: KProjectionCombination | KProjection, /
    ) -> KProjectionCombination:
        """Subtract the combination to another."""
        if isinstance(other, KProjectionCombination):
            return KProjectionCombination(
                self.weight, *self.pairs, *((-a, b) for a, b in other.pairs)
            )
        if isinstance(other, (KProjection)):
            return KProjectionCombination(self.weight, *self.pairs, (-1.0, other))
        return NotImplemented

    def __rsub__(
        self, other: KProjectionCombination | KProjection, /
    ) -> KProjectionCombination:
        """Subtract the combination to another."""
        return (self.__sub__(other)).__neg__()

    def __mul__(self, other: float | int, /) -> KProjectionCombination:
        """Multiply the projection by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KProjectionCombination(self.weight, *((v * a, b) for a, b in self.pairs))

    def __rmul__(self, other: float | int, /) -> KProjectionCombination:
        """Multiply the projection by a constant."""
        return self.__mul__(other)

    def __div__(self, other: float | int, /) -> KProjectionCombination:
        """Divide the projection by a constant."""
        try:
            v = float(other)
        except Exception:
            return NotImplemented
        return KProjectionCombination(self.weight, *((a / v, b) for a, b in self.pairs))

    def __neg__(self) -> KProjectionCombination:
        """Negate the combination."""
        return KProjectionCombination(self.weight, *((-a, b) for a, b in self.pairs))

    def __str__(self) -> str:
        """Convert into a print-friendly string."""
        s = ""
        for a, b in self.pairs:
            fn = b.func
            if fn is None or a == 0:
                continue
            else:
                name = fn.__name__

            pre = ""

            if len(s):
                if a < 0:
                    pre = " - "
                else:
                    pre = " + "
            else:
                if a < 0:
                    pre = "-"
                else:
                    pre = ""

            s += pre + f"{abs(a):g} * {name}"
        if not s:
            return "0"
        return s


def _extract_forms(
    form: Term,
) -> tuple[set[KFormUnknown], set[KWeight], list[KForm], set[VectorFieldFunction]]:
    """Extract all unknown and weight forms, which make up the current form.

    Parameters
    ----------
    form : Term
        Form which is to be extracted.

    Returns
    -------
    set of KForm
        All unique forms occurring within the term.
    set of KFormWeight
        All unique weight forms occurring within the term.
    list of KForm
        List of weak form terms.
    set of VectorFieldFunction
        Set of interior product functions.
    """
    if type(form) is KFormDerivative:
        return _extract_forms(form.form)
    if type(form) is KSum:
        set_f: set[KFormUnknown] = set()
        set_w: set[KWeight] = set()
        weak_all: list[KForm] = []
        vffs: set[VectorFieldFunction] = set()
        for _, ip in form.pairs:
            f, w, weak, fn = _extract_forms(ip)
            set_f |= f
            set_w |= w
            vffs |= fn
            weak_all += weak
        return (set_f, set_w, weak_all, vffs)

    if type(form) is KInnerProduct:
        f1 = _extract_forms(form.weight)
        f2 = _extract_forms(form.function)
        weak = f1[2] + f2[2]
        assert len(f1[2]) == 0 and len(f2[2]) == 0
        if type(form.weight) is KFormDerivative:
            weak.append(form.function)
        return (f1[0] | f2[0], f1[1] | f2[1], weak, f1[3] | f2[3])

    if type(form) is KHodge:
        return _extract_forms(form.base_form)

    if type(form) is KFormUnknown:
        return ({form}, set(), [], set())

    if type(form) is KWeight:
        return (set(), {form}, [], set())

    if type(form) is KInteriorProduct:
        f1 = _extract_forms(form.form)
        return (f1[0], f1[1], f1[2], f1[3] | {form.vector_field})

    raise TypeError(f"Invalid type {type(form)}")


@dataclass(init=False, frozen=True)
class KEquaton:
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

    left: KSum | KInnerProduct
    right: KProjectionCombination
    variables: tuple[KForm, ...]
    weak_forms: tuple[KForm]
    vector_fields: set[VectorFieldFunction]

    def __init__(self, left: KSum | KInnerProduct, right: KProjectionCombination) -> None:
        p, d, w, vf = _extract_forms(left)
        if d != {right.weight}:
            raise ValueError("Left and right side do not use the same weight.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "variables", tuple(k for k in p))
        object.__setattr__(self, "weak_forms", tuple(w))
        object.__setattr__(self, "vector_fields", vf)


def _parse_form(form: Term) -> dict[Term, str | None]:
    """Extract the string representations of the forms, as well as their dual weight.

    Parameters
    ----------
    form : Term
        Form, which is to be extracted.

    Returns
    -------
    dict of KForm -> (str or None)
        Mapping of forms in the equation and string representation of the operations
        performed on them.
    """
    if type(form) is KSum:
        left: dict[Term, str | None] = {}
        for c, p in form.pairs:
            right = _parse_form(p)
            if c != 1.0:
                for k in right:
                    vk = right[k]
                    if vk is not None:
                        vk = f"{abs(c):g} * {vk}"
                        right[k] = vk
                    else:
                        right[k] = f"{abs(c):g}"

            for k in right:
                if k in left:
                    vl = left[k]
                    vr = right[k]
                    if vl is not None and vr is not None:
                        if c > 0:
                            left[k] = f"({left[k]} + {right[k]})"
                        else:
                            left[k] = f"({left[k]} - {right[k]})"
                    else:
                        left[k] = vl if vl is not None else vr
                else:
                    vr = right[k]
                    if c >= 0 or vr is None:
                        left[k] = vr
                    else:
                        left[k] = "-" + vr

        return left
    if type(form) is KInnerProduct:
        if isinstance(form.function, KHodge):
            primal = _parse_form(form.function.base_form)
        else:
            primal = _parse_form(form.function)
        dual = _parse_form(form.weight)
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            if form.function.is_primal:
                if form.weight.is_primal:
                    mtx_name = f"M({form.weight.order})"
                else:
                    mtx_name = ""
            else:
                if form.weight.is_primal:
                    mtx_name = ""
                else:
                    mtx_name = f"M({form.weight.primal_order})^{{-1}}"

            primal[k] = (
                (f"({vd})^T @ " if vd is not None else "")
                + mtx_name
                + (f" @ {vp}" if vp is not None else "")
            )
        primal[dv] = None
        return primal

    if type(form) is KFormDerivative:
        res = _parse_form(form.form)
        if form.form.is_primal:
            mtx_name = f"E({form.order}, {form.order - 1})"
        else:
            mtx_name = f"E({form.primal_order + 1}, {form.primal_order})^T"
        for k in res:
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")
        return res

    if type(form) is KInteriorProduct:
        res = _parse_form(form.form)
        mtx_name = f"M({form.order}, {form.form.order}; {form.vector_field.__name__})"
        for k in res:
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")
        return res

    if type(form) is KHodge:
        res = _parse_form(form.base_form)
        for k in res:
            if form.base_form.is_primal:
                mtx_name = f"M({form.primal_order})"
            else:
                mtx_name = f"M({form.primal_order})^{{-1}}"
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")
        return res

    if type(form) is KFormUnknown:
        return {form: None}
    if type(form) is KWeight:
        return {form: None}
    raise TypeError("Unknown type")


@overload
def _form_size_2d(p: npt.NDArray[np.integer], k: int) -> npt.NDArray[np.integer]: ...


@overload
def _form_size_2d(p: int, k: int) -> int: ...


def _form_size_2d(
    p: npt.NDArray[np.integer] | int, k: int
) -> npt.NDArray[np.integer] | int:
    """Compute number of degrees of freedom a form gets on an element with order."""
    if k == 0:
        return (p + 1) * (p + 1)
    if k == 1:
        return 2 * p * (p + 1)
    if k == 2:
        return p * p
    assert False


class KFormSystem:
    """System of equations of differential forms, which are optionally sorted.

    This is a collection of equations, which fully describe a problem to be solved for
    the degrees of freedom of differential forms.

    Parameters
    ----------
    *equations : KFormEquation
        Equations which are to be used.
    sorting_primal : (KFormPrimal) -> Any, optional
        Callable passed to the :func:`sorted` builtin to sort the primal forms. This
        corresponds to sorting the columns of the system matrix.
    sorting_dual : (KFormDual) -> Any, optional
        Callable passed to the :func:`sorted` builtin to sort the dual forms. This
        corresponds to sorting the rows of the system matrix.
    """

    unknown_forms: tuple[KFormUnknown, ...]
    dual_forms: tuple[KWeight, ...]
    equations: tuple[KEquaton, ...]
    weak_forms: set[KForm]
    vector_fields: tuple[VectorFieldFunction, ...]

    def __init__(
        self,
        *equations: KEquaton,
        sorting: Callable[[KForm], Any] | None = None,
    ) -> None:
        unknowns: set[KFormUnknown] = set()
        weights: list[KWeight] = []
        weak: set[KForm] = set()
        equation_list: list[KEquaton] = []
        vfs: set[VectorFieldFunction] = set()
        for ie, equation in enumerate(equations):
            p, d, w, vf = _extract_forms(equation.left)
            vfs |= vf
            weak |= set(w)
            unknowns |= p
            if len(d) != 1:
                raise ValueError(f"Equation {ie} has more that one dual weight.")
            dual = d.pop()
            if dual != equation.right.weight:
                raise ValueError(
                    f"The dual forms of the left and right sides of the equation {ie} "
                    f"don't match ({dual} vs {equation.right.weight})."
                )
            if dual in weights:
                raise ValueError(
                    f"Dual form is not unique to the equation {ie}, as it already appears"
                    f" in equation {weights.index(dual)}."
                )
            weights.append(dual)
            equation_list.append(equation)
        self.weak_forms = weak
        if sorting is not None:
            self.unknown_forms = tuple(sorted(unknowns, key=sorting))
        else:
            self.unknown_forms = tuple(unknowns)

        self.dual_forms = tuple(
            weights[self.unknown_forms.index(d.base_form)] for d in weights
        )
        self.equations = tuple(
            equation_list[self.unknown_forms.index(d.base_form)] for d in weights
        )
        self.vector_fields = tuple(vec_field for vec_field in vfs)

    @overload
    def shape_1d(
        self, order: npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]: ...

    @overload
    def shape_1d(self, order: int) -> tuple[int, int]: ...

    def shape_1d(
        self, order: npt.NDArray[np.integer] | int
    ) -> tuple[npt.NDArray[np.integer] | int, npt.NDArray[np.integer] | int]:
        """Return the shape of the system for the 1D case.

        Parameters
        ----------
        order : int
            Order of 1D polynomial basis.

        Returns
        -------
        int
            Number of rows of the system.
        int
            Number of columns of the system.
        """
        height = sum(order + 1 - d.order for d in self.dual_forms)
        width = sum(order + 1 - d.order for d in self.unknown_forms)
        return (height, width)

    @overload
    def offsets_1d(self, order: int) -> tuple[tuple[int, ...], tuple[int, ...]]: ...

    @overload
    def offsets_1d(
        self, order: npt.NDArray[np.integer]
    ) -> tuple[
        tuple[npt.NDArray[np.integer], ...], tuple[npt.NDArray[np.integer], ...]
    ]: ...

    def offsets_1d(
        self, order: int | npt.NDArray[np.integer]
    ) -> tuple[
        tuple[int | npt.NDArray[np.integer], ...],
        tuple[int | npt.NDArray[np.integer], ...],
    ]:
        """Compute offsets of different forms and equations.

        Parameters
        ----------
        order : int
            Order of 1D polynomial basis.

        Returns
        -------
        tuple[int, ...]
            Offsets of different equations in rows.
        tuple[int, ...]
            Offsets of different form degrees of freedom in columns.
        """
        offset_forms = (np.zeros_like(order),) + tuple(
            accumulate(order + 1 - d.order for d in self.unknown_forms)
        )
        offset_equations = (np.zeros_like(order),) + tuple(
            accumulate(order + 1 - d.order for d in self.dual_forms)
        )
        return offset_equations, offset_forms

    @overload
    def shape_2d(
        self, order: npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]: ...

    @overload
    def shape_2d(self, order: int) -> tuple[int, int]: ...

    def shape_2d(
        self, order: npt.NDArray[np.integer] | int
    ) -> tuple[npt.NDArray[np.integer] | int, npt.NDArray[np.integer] | int]:
        """Return the shape of the system for the 2D case.

        Parameters
        ----------
        order : int
            Order of 2D polynomial basis.

        Returns
        -------
        int
            Number of rows of the system.
        int
            Number of columns of the system.
        """
        height = sum(_form_size_2d(order, d.order) for d in self.dual_forms)
        width = sum(_form_size_2d(order, d.order) for d in self.unknown_forms)
        return (height, width)

    @overload
    def offsets_2d(self, order: int) -> tuple[tuple[int, ...], tuple[int, ...]]: ...

    @overload
    def offsets_2d(
        self, order: npt.NDArray[np.integer]
    ) -> tuple[
        tuple[npt.NDArray[np.integer], ...], tuple[npt.NDArray[np.integer], ...]
    ]: ...

    def offsets_2d(
        self, order: int | npt.NDArray[np.integer]
    ) -> tuple[
        tuple[int | npt.NDArray[np.integer], ...],
        tuple[int | npt.NDArray[np.integer], ...],
    ]:
        """Compute offsets of different forms and equations.

        Parameters
        ----------
        order : int
            Order of 2D polynomial basis.

        Returns
        -------
        tuple[int, ...]
            Offsets of different equations in rows.
        tuple[int, ...]
            Offsets of different form degrees of freedom in columns.
        """
        offset_forms = (np.zeros_like(order),) + tuple(
            accumulate(_form_size_2d(order, d.order) for d in self.unknown_forms)
        )
        offset_equations = (np.zeros_like(order),) + tuple(
            accumulate(_form_size_2d(order, d.order) for d in self.dual_forms)
        )
        return offset_equations, offset_forms

    def __str__(self) -> str:
        """Create a printable representation of the object."""
        duals: list[KWeight] = []
        out_mat: list[list[str]] = []
        rhs: list[str] = []
        for ie, eq in enumerate(self.equations):
            p = _parse_form(eq.left)
            d = [k for k in p if isinstance(k, KWeight)]
            if len(d) != 1:
                raise ValueError(
                    f"Equation {ie} does not have a single weight (weights were {d})."
                )
            o: dict[KForm, str] = {}
            for k in p:
                if not isinstance(k, KForm):
                    continue
                v = p[k]
                if v is None:
                    o[k] = "I"
                else:
                    o[k] = v
            duals.extend(d)
            out_list: list[str] = []
            for k in self.unknown_forms:
                if k in o:
                    out_list.append(o[k])
                else:
                    out_list.append("0")
            out_mat.append(out_list)
            rhs.append(str(eq.right))

        out_dual = [str(d) for d in self.dual_forms]
        out_v = [str(iv) for iv in self.unknown_forms]

        w_mat = max(max(len(s) for s in row) for row in out_mat)
        w_v = max(len(s) for s in out_v)
        w_d = max(len(s) for s in out_dual)
        w_r = max(len(s) for s in rhs)

        s = ""
        for ie in range(len(self.equations)):
            row = out_mat[ie]
            row_str = " | ".join(rs.rjust(w_mat) for rs in row)
            s += (
                f"[{out_dual[ie].rjust(w_d)}]"
                + ("    (" if ie != 0 else "^T  (")
                + f"[{row_str}]  [{out_v[ie].rjust(w_v)}] - [{rhs[ie].rjust(w_r)}]) ="
                + " [0]\n"
            )
        return s[:-1]  # strip the trailing new line.
