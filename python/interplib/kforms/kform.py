"""Dealing with K-forms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre

from interplib._mimetic import Manifold
from interplib.mimetic.mimetic1d import Element1D


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
class KFormBase(Term):
    """Base class for K forms, which implements the common methods.

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


@dataclass(frozen=True)
class KForm(KFormBase):
    """Differential K form.

    It is described by and order and identifier, that is used to print it.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    """

    def __mul__(self, other: KWeight, /) -> KInnerProduct:
        """Inner product with a weight."""
        if isinstance(other, KWeight):
            return KInnerProduct(other, self)
        return NotImplemented

    def __rmul__(self, other: KWeight, /) -> KInnerProduct:
        """Inner product with a weight."""
        if isinstance(other, KWeight):
            return KInnerProduct(other, self)
        return NotImplemented

    def __invert__(self) -> KHodge:
        """Hodge the k-form."""
        return KHodge(self)

    @property
    def derivative(self) -> KFormDerivative:
        """Derivative of the form."""
        return KFormDerivative(self)

    @property
    def weight(self) -> KWeight:
        """Create a weight based on this form."""
        return KWeight(self.manifold, self.label, self.order, self)

    @property
    def is_primal(self) -> bool:
        """Check if the form is primal."""
        return True

    @property
    def primal_order(self) -> int:
        """Order in primal basis."""
        return self.order


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


@dataclass(frozen=True, eq=False)
class KWeight(KFormBase):
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
    def derivative(self) -> KWeightDerivative:
        """Derivative of the form."""
        return KWeightDerivative(self)

    def __mul__(self, other: Callable | Literal[0], /) -> KFormProjection:
        """Create projection for the right hand side."""
        if isinstance(other, int) and other == 0:
            return KFormProjection(self, None)
        if callable(other):
            return KFormProjection(self, other)
        return NotImplemented


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


@dataclass(init=False, frozen=True, eq=False)
class KWeightDerivative(KWeight):
    r"""Exterior derivative of a weight form.

    This is the operation represented by :class:`KFormDerivative`, but applied to a weight
    function. This means that it is represented by a transpose of the incidence matrix.

    Parameters
    ----------
    form : KFormWeight
        The weight form of which the derivative is to be taken.
    """

    form: KWeight

    def __init__(self, form: KWeight) -> None:
        object.__setattr__(self, "form", form)
        super().__init__(
            form.manifold, "d" + form.label, form.order + 1, base_form=form.base_form
        )


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

    weight: KWeight
    function: KForm

    # Check if order is even needed
    def __init__(self, weight: KWeight, function: KForm) -> None:
        wg_order = weight.order
        fn_order = function.order
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

    @overload
    def __eq__(self, other: KFormProjection, /) -> KEquaton: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(self, other: KFormProjection, /) -> KEquaton | bool:
        """Check equality or form an equation."""
        if isinstance(other, KFormProjection):
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
    def __eq__(self, other: KFormProjection, /) -> KEquaton: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(self, other: KFormProjection, /) -> KEquaton | bool:
        """Check equality or form an equation."""
        if isinstance(other, KFormProjection):
            return KEquaton(self, other)
        return self is other


@dataclass(frozen=True)
class KFormProjection:
    r"""Weigh a form with a dual form and implicilty project it.

    This is used to form the right side of the systems of equations. It represents
    the projection of a given  function :math:`f: \mathbb{R}^n -> \mathbb{R}` onto
    a set of primal basis to create a k-form, which is then used to compute the inner
    product with the specified dual form.

    Parameters
    ----------
    dual : KFormWeight
        Dual form used as a weight.
    func : tuple[str, Callable], optional
        The function to use, specified by a name and the callable to use. If it not
        specified or given as ``None``, then :math:`f = 0`.
    """

    weight: KWeight
    func: Callable | None = None


def _extract_forms(form: Term) -> tuple[set[KForm], set[KWeight], list[KForm]]:
    """Extract all primal and dual forms, which make up the current form.

    Parameters
    ----------
    form : Term
        Form which is to be extracted.

    Returns
    -------
    set of KForm
        All unique forms occurring within the form.
    set of KFormWeight
        All unique weight forms occurring within the form.
    list of KForm
        List of weak form terms.
    """
    if type(form) is KFormDerivative:
        return ({form.form}, set(), [])
    if type(form) is KWeightDerivative:
        return (set(), {form.form}, [])
    if type(form) is KSum:
        set_f: set[KForm] = set()
        set_w: set[KWeight] = set()
        weak_all: list[KForm] = []
        for _, ip in form.pairs:
            f, w, weak = _extract_forms(ip)
            set_f |= f
            set_w |= w
            weak_all += weak
        return (set_f, set_w, weak_all)
    if type(form) is KInnerProduct:
        f1 = _extract_forms(form.weight)
        f2 = _extract_forms(form.function)
        weak = f1[2] + f2[2]
        assert len(f1[2]) == 0 and len(f2[2]) == 0
        if type(form.weight) is KWeightDerivative:
            weak.append(form.function)
        return (f1[0] | f2[0], f1[1] | f2[1], weak)
    if type(form) is KHodge:
        return _extract_forms(form.base_form)
    if type(form) is KForm:
        return ({form}, set(), [])
    if type(form) is KWeight:
        return (set(), {form}, [])
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
    right: KFormProjection
    variables: tuple[KForm, ...]
    weak_forms: tuple[KForm]

    def __init__(self, left: KSum | KInnerProduct, right: KFormProjection) -> None:
        p, d, w = _extract_forms(left)
        if d != {right.weight}:
            raise ValueError("Left and right side do not use the same weight.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "variables", tuple(k for k in p))
        object.__setattr__(self, "weak_forms", tuple(w))


_cached_roots_legendre = cache(roots_legendre)


def _extract_rhs_1d(
    right: KFormProjection, element: Element1D
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element1D
        The element on which the projection is evaluated on.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    fn = right.func
    p = element.order
    if fn is None:
        if right.weight.order == 1:
            return np.zeros(p)
        elif right.weight.order == 0:
            return np.zeros(p + 1)
        else:
            assert False
    else:
        out_vec: npt.NDArray[np.float64]
        mass: npt.NDArray[np.floating]
        real_nodes = np.linspace(element.xleft, element.xright, p + 1)
        # TODO: find a nice way to do this
        if right.weight.order == 1:
            xi, w = _cached_roots_legendre(2 * p)
            out_vec = np.empty(p)
            func = fn
            for i in range(p):
                # out_vec[i] =  quad(fn[1], real_nodes[i], real_nodes[i + 1])[0]
                dx = real_nodes[i + 1] - real_nodes[i]
                out_vec[i] = np.dot(w, func(dx * (xi + 1) / 2 + real_nodes[i])) * (dx) / 2
            mass = element.mass_edge
        elif right.weight.order == 0:
            out_vec = np.empty(p + 1)
            out_vec[:] = fn(real_nodes)
            mass = element.mass_node
        return np.astype(mass @ np.astype(out_vec, np.float64), np.float64)


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
                    left[k] = right[k]
        return left
    if type(form) is KInnerProduct:
        primal = _parse_form(form.function)
        dual = _parse_form(form.weight)
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            primal[k] = (
                (f"({vd})^T @ " if vd is not None else "")
                + f"M({form.weight.order}, {form.function.primal_order})"
                + (f" @ {vp}" if vp is not None else "")
            )
        primal[dv] = None
        return primal
    if type(form) is KFormDerivative:
        res = _parse_form(form.form)
        for k in res:
            res[k] = f"E({form.order}, {form.order - 1})" + (
                f" @ {res[k]}" if res[k] is not None else ""
            )
        return res
    if type(form) is KWeightDerivative:
        res = _parse_form(form.form)
        for k in res:
            res[k] = f"E({form.order}, {form.order - 1})" + (
                f" @ {res[k]}" if res[k] is not None else ""
            )
        return res
    if type(form) is KHodge:
        res = _parse_form(form.base_form)
        for k in res:
            res[k] = f"M({form.primal_order})" + (
                f" @ {res[k]}" if res[k] is not None else ""
            )
        return res

    if type(form) is KForm:
        return {form: None}
    if type(form) is KWeight:
        return {form: None}
    raise TypeError("Unknown type")


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

    primal_forms: tuple[KForm, ...]
    dual_forms: tuple[KWeight, ...]
    equations: tuple[KEquaton, ...]
    weak_forms: set[KForm]

    def __init__(
        self,
        *equations: KEquaton,
        sorting: Callable[[KForm], Any] | None = None,
    ) -> None:
        primals: set[KForm] = set()
        duals: list[KWeight] = []
        weak: set[KForm] = set()
        equation_list: list[KEquaton] = []
        for ie, equation in enumerate(equations):
            p, d, w = _extract_forms(equation.left)
            weak |= set(w)
            primals |= p
            if len(d) != 1:
                raise ValueError(f"Equation {ie} has more that one dual weight.")
            dual = d.pop()
            if dual != equation.right.weight:
                raise ValueError(
                    f"The dual forms of the left and right sides of the equation {ie} "
                    f"don't match ({dual} vs {equation.right.weight})."
                )
            if dual in duals:
                raise ValueError(
                    f"Dual form is not unique to the equation {ie}, as it already appears"
                    f" in equation {duals.index(dual)}."
                )
            duals.append(dual)
            equation_list.append(equation)
        self.weak_forms = weak
        if sorting is not None:
            self.primal_forms = tuple(sorted(primals, key=sorting))
        else:
            self.primal_forms = tuple(primals)

        self.dual_forms = tuple(
            duals[self.primal_forms.index(d.base_form)] for d in duals
        )
        self.equations = tuple(
            equation_list[self.primal_forms.index(d.base_form)] for d in duals
        )

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
        width = sum(order + 1 - d.order for d in self.primal_forms)
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
            accumulate(order + 1 - d.order for d in self.primal_forms)
        )
        offset_equations = (np.zeros_like(order),) + tuple(
            accumulate(order + 1 - d.order for d in self.dual_forms)
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
            for k in self.primal_forms:
                if k in o:
                    out_list.append(o[k])
                else:
                    out_list.append("0")
            out_mat.append(out_list)
            fn = eq.right.func
            rhs.append(fn.__name__ if fn is not None else "0")

        out_dual = [str(d) for d in self.dual_forms]
        out_v = [str(iv) for iv in self.primal_forms]

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
        return s


def _equation_1d(
    form: Term, element: Element1D
) -> dict[Term, npt.NDArray[np.float64] | np.float64]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : Term
        Form to evaluate.

    Returns
    -------
    dict of KForm -> array or float
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``float``, if it should be multiplication with a constant.
    """
    if type(form) is KSum:
        left: dict[Term, npt.NDArray[np.float64] | np.float64] = {}

        for c, ip in form.pairs:
            right = _equation_1d(ip, element)
            if c != 1.0:
                for f in right:
                    right[f] *= c  # type: ignore

            for k in right:
                vr = right[k]
                if k in left:
                    vl = left[k]
                    if vl.ndim == vr.ndim:
                        left[k] = np.asarray(
                            vl + vr, np.float64
                        )  # vl and vr are non-none
                    elif vl.ndim == 0:
                        assert isinstance(vr, np.ndarray)
                        mat = np.eye(vr.shape[0], vr.shape[1]) * vr
                        left[k] = np.astype(mat + vr, np.float64)
                else:
                    left[k] = right[k]  # k is not in left
        return left
    if type(form) is KInnerProduct:
        primal = _equation_1d(form.function, element)
        dual = _equation_1d(form.weight, element)
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            order_p = form.function.primal_order
            order_d = form.weight.order
            mass: npt.NDArray[np.float64]
            if order_p == 0 and order_d == 0:
                mass = element.mass_node  # type: ignore
            elif order_p == 1 and order_d == 1:
                mass = element.mass_edge  # type: ignore
            elif order_p == 1 and order_d == 0:
                mass = element.mass_node_edge  # type: ignore
            elif order_p == 0 and order_d == 1:
                mass = element.mass_node_edge.transpose()  # type: ignore
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 1D mesh."
                )
            if vd.ndim != 0:
                assert isinstance(vd, np.ndarray)
                mass = np.astype(vd.T @ mass, np.float64)
            else:
                assert isinstance(vd, np.float64)
                mass *= vd
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            primal[k] = mass
        return primal
    if type(form) is KFormDerivative:
        res = _equation_1d(form.form, element)
        e = element.incidence_primal_0()
        for k in res:
            rk = res[k]
            if rk.ndim != 0:
                res[k] = np.astype(e @ rk, np.float64)
            else:
                assert isinstance(rk, np.float64)
                res[k] = np.astype(e * rk, np.float64)
        return res

    if type(form) is KWeightDerivative:
        res = _equation_1d(form.form, element)
        e = element.incidence_primal_0()
        for k in res:
            rk = res[k]
            if rk.ndim != 0:
                res[k] = np.astype(e @ rk, np.float64)
            else:
                assert isinstance(rk, np.float64)
                res[k] = np.astype(e * rk, np.float64)
        return res
    if type(form) is KHodge:
        primal = _equation_1d(form.base_form, element)
        prime_order = form.primal_order
        for k in primal:
            if prime_order == 0:
                mass = element.mass_node  # type: ignore
            elif prime_order == 1:
                mass = element.mass_edge  # type: ignore
            else:
                assert False
            vp = primal[k]
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            primal[k] = mass
        return primal
    if type(form) is KForm:
        return {form: np.float64(1.0)}
    if type(form) is KWeight:
        return {form: np.float64(1.0)}
    raise TypeError("Unknown type")


def element_system(
    system: KFormSystem, element: Element1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute element matrix and vector.

    Parameters
    ----------
    system : KFormSystem
        System to discretize.
    element : Element1D
        The element on which the discretization should be performed.

    Returns
    -------
    array
        Element matrix representing the left side of the system.
    array
        Element vector representing the right side of the system
    """
    system_size = system.shape_1d(element.order)
    assert system_size[0] == system_size[1], "System must be square."
    system_matrix = np.zeros(system_size, np.float64)
    system_vector = np.zeros(system_size[0], np.float64)
    offset_equations, offset_forms = system.offsets_1d(element.order)

    for ie, equation in enumerate(system.equations):
        form_matrices = _equation_1d(equation.left, element)
        for form in form_matrices:
            val = form_matrices[form]
            idx = system.primal_forms.index(form)
            assert val is not None
            system_matrix[
                offset_equations[ie] : offset_equations[ie + 1],
                offset_forms[idx] : offset_forms[idx + 1],
            ] = val
        system_vector[offset_equations[ie] : offset_equations[ie + 1]] = _extract_rhs_1d(
            equation.right, element
        )

    return system_matrix, system_vector
