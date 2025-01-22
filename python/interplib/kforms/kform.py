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

from interplib.mimetic.mimetic1d import Element1D


@dataclass(frozen=True)
class KForm:
    """Generic differential K form.

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

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order})"

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        raise NotImplementedError

    @property
    def is_dual(self) -> bool:
        """Check if form is dual."""
        return not self.is_primal

    def __mul__(self, other: KForm) -> KFormInnerProduct:
        """Inner product of two KForms."""
        if isinstance(other, KForm):
            return KFormInnerProduct(self, other)
        return NotImplemented

    def __add__(self, other: KForm) -> KFormSum:
        """Addition of two KForms."""
        if isinstance(other, KForm):
            return KFormSum(self, other)
        return NotImplemented

    @property
    def derivative(self) -> KFormDerivative:
        """Derivative of the form."""
        return KFormDerivative(self)

    @overload
    def __eq__(self, other: KFormProjection, /) -> KFormEquaton: ...

    @overload
    def __eq__(self, other, /) -> bool: ...

    def __eq__(self, other: KFormProjection | object) -> KFormEquaton | bool:
        """When used against KFormProjection form an equation, otherwise compare.

        If ``type(other) is KFormProjection``, the result is
        ``KFormEquaton(self, other)``, otherwise it simply returns ``self is other``.
        """
        if isinstance(other, KFormProjection):
            return KFormEquaton(self, other)
        return self is other

    def equals(self, other: KFormProjection) -> KFormEquaton:
        """Check for equality with another object or form an equation.

        If ``other`` is KFormProjection, the result is an equation. In any other
        case, the result is equal to expression ``self == other`` is equivalent to
        ``self is other``.
        """
        return KFormEquaton(self, other)


@dataclass(frozen=True, eq=False)
class KFormPrimal(KForm):
    """Differential K form represented with the primal basis.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    """

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return True


@dataclass(frozen=True, eq=False)
class KFormDual(KForm):
    """Differential K form represented with the dual basis.

    Parameters
    ----------
    order : int
        Order of the differential form.
    label : str
        Label which is used to print form as a string.
    """

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return False

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order}*)"

    def __matmul__(self, other: tuple[str, Callable] | Literal[0]) -> KFormProjection:
        """Check for equality, or construct the projection."""
        if isinstance(other, int) and other == 0:
            return KFormProjection(self, None)
        name, fn = other
        return KFormProjection(self, (str(name), fn))


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
        super().__init__(form.order + 1, label="d" + form.label)

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return self.form.is_primal


@dataclass(init=False, frozen=True, eq=False)
class KFormInnerProduct(KForm):
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

    def __init__(self, weight: KForm, function: KForm) -> None:
        wg_order = weight.order
        fn_order = function.order
        if wg_order != fn_order:
            raise ValueError(
                f"The K forms are not of the same order ({wg_order} vs {fn_order})"
            )
        if weight.is_primal == function.is_primal:
            raise ValueError(
                "Can not take inner product of two primal or two dual forms."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "function", function)
        super().__init__(0, f"<{weight.label}, {function.label}>")

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return False

    @property
    def is_dual(self) -> bool:
        """Check if form is dual."""
        return False


@dataclass(init=False, frozen=True, eq=False)
class KFormSum(KForm):
    """Sum of two differential forms of the same order.

    Parameters
    ----------
    first : KForm
        First form to add.
    second : KForm
        Second form to add.
    """

    first: KForm
    second: KForm

    def __init__(self, first: KForm, second: KForm, /) -> None:
        order = first.order
        if order != second.order or first.is_primal != second.is_primal:
            raise ValueError(
                "Can only sum forms if both are primal/dual and have the same order."
            )
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        super().__init__(order, f"({first.label} + {second.label})")


@dataclass(frozen=True)
class KFormProjection:
    r"""Weigh a form with a dual form and implicilty project it.

    This is used to form the right side of the systems of equations. It represents
    the projection of a given  function :math:`f: \mathbb{R}^n -> \mathbb{R}` onto
    a set of primal basis to create a k-form, which is then used to compute the inner
    product with the specified dual form.

    Parameters
    ----------
    dual : KFormDual
        Dual form used as a weight.
    func : tuple[str, Callable], optional
        The function to use, specified by a name and the callable to use. If it not
        specified or given as ``None``, then :math:`f = 0`.
    """

    dual: KFormDual
    func: tuple[str, Callable] | None = None


def _extract_forms(form: KForm) -> tuple[set[KFormPrimal], set[KFormDual], list[KForm]]:
    """Extract all primal and dual forms, which make up the current form.

    Parameters
    ----------
    form : KForm
        Form which is to be extracted.

    Returns
    -------
    set of KFormPrimal
        All unique primal forms occurring within the form.
    set of KFormDual
        All unique dual forms occurring within the form.
    list of KForm
        All k-forms for which the weak boundary conditions can be used.
    """
    if isinstance(form, KFormPrimal):
        return ({form}, set(), [])
    if isinstance(form, KFormDual):
        return (set(), {form}, [])
    if isinstance(form, KFormDerivative):
        return _extract_forms(form.form)
    if isinstance(form, KFormSum):
        f1 = _extract_forms(form.first)
        f2 = _extract_forms(form.second)
        return (f1[0] | f2[0], f1[1] | f2[1], f1[2] + f2[2])
    if isinstance(form, KFormInnerProduct):
        f1 = _extract_forms(form.function)
        f2 = _extract_forms(form.weight)
        out_weak: list[KForm] = []
        if isinstance(form.weight, KFormDerivative) and form.weight.is_dual:
            out_weak.append(form.function)
        return (f1[0] | f2[0], f1[1] | f2[1], out_weak)
    raise TypeError(f"Invalid type {type(form)}")


@dataclass(init=False, frozen=True)
class KFormEquaton:
    """Equation of differential forms, consisting of a left and a right side.

    The equation represents an equation where all the implicit terms are on the left side
    and all explicit ones are on the right side.

    Parameters
    ----------
    left : KForm
        The form representing the implicit part of the equation.
    right : KFormProjection
        The form representing the explicit part of the equation.
    """

    left: KForm
    right: KFormProjection
    variables: tuple[KForm, ...]
    weak_forms: tuple[KForm]

    def __init__(self, left: KForm, right: KFormProjection) -> None:
        p, d, w = _extract_forms(left)
        if d != {right.dual}:
            raise ValueError(
                "Left and right side do not use the same dual form as a weight."
            )
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
        if right.dual.order == 1:
            return np.zeros(p)
        elif right.dual.order == 0:
            return np.zeros(p + 1)
        else:
            assert False
    else:
        out_vec: npt.NDArray[np.float64]
        mass: npt.NDArray[np.floating]
        real_nodes = np.linspace(element.xleft, element.xright, p + 1)
        # TODO: find a nice way to do this
        if right.dual.order == 1:
            xi, w = _cached_roots_legendre(2 * p)
            out_vec = np.empty(p)
            func = fn[1]
            for i in range(p):
                # out_vec[i] =  quad(fn[1], real_nodes[i], real_nodes[i + 1])[0]
                dx = real_nodes[i + 1] - real_nodes[i]
                out_vec[i] = np.dot(w, func(dx * (xi + 1) / 2 + real_nodes[i])) * (dx) / 2
            mass = element.mass_edge
        elif right.dual.order == 0:
            out_vec = np.empty(p + 1)
            out_vec[:] = fn[1](real_nodes)
            mass = element.mass_node
        return np.astype(mass @ np.astype(out_vec, np.float64), np.float64)


def _parse_form(form: KForm) -> dict[KForm, str | None]:
    """Extract the string representations of the forms, as well as their dual weight.

    Parameters
    ----------
    form : KForm
        Form, which is to be extracted.

    Returns
    -------
    KFormDual
        The dual form, if any are used.
    dict of KForm -> (str or None)
        Mapping of forms in the equation and string representation of the operations
        performed on them.
    """
    if isinstance(form, KFormSum):
        left = _parse_form(form.first)
        right = _parse_form(form.second)
        for k in right:
            if k in left:
                vl = left[k]
                vr = right[k]
                if vl is not None and vr is not None:
                    left[k] = f"({left[k]} + {right[k]})"
                else:
                    left[k] = vl if vl is not None else vr
            else:
                left[k] = right[k]
        return left
    if isinstance(form, KFormInnerProduct):
        if form.weight.is_dual:
            primal = _parse_form(form.function)
            dual = _parse_form(form.weight)
        else:
            dual = _parse_form(form.function)
            primal = _parse_form(form.weight)
        assert len(dual) == 1
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            primal[k] = (
                (f"({vd})^T @ " if vd is not None else "")
                + f"M({form.function.order})"
                + (f" @ {vp}" if vp is not None else "")
            )
        primal[dv] = None
        return primal
    if isinstance(form, KFormDerivative):
        res = _parse_form(form.form)
        for k in res:
            res[k] = f"E({form.order}, {form.order - 1})" + (
                f" @ {res[k]}" if res[k] is not None else ""
            )
        return res
    if isinstance(form, KFormPrimal):
        return {form: None}
    if isinstance(form, KFormDual):
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

    primal_forms: tuple[KFormPrimal, ...]
    dual_forms: tuple[KFormDual, ...]
    equations: tuple[KFormEquaton, ...]
    weak_forms: set[KForm]

    def __init__(
        self,
        *equations: KFormEquaton,
        sorting_primal: Callable[[KFormPrimal], Any] | None = None,
        sorting_dual: Callable[[KFormDual], Any] | None = None,
    ) -> None:
        primals: set[KFormPrimal] = set()
        duals: list[KFormDual] = []
        weak: set[KForm] = set()
        equation_list: list[KFormEquaton] = []
        for ie, equation in enumerate(equations):
            p, d, w = _extract_forms(equation.left)
            weak |= set(w)
            primals |= p
            if len(d) != 1:
                raise ValueError(f"Equation {ie} has more that one dual weight.")
            dual = d.pop()
            if dual != equation.right.dual:
                raise ValueError(
                    f"The dual forms of the left and right sides of the equation {ie} "
                    f"don't match ({dual} vs {equation.right.dual})."
                )
            if dual in duals:
                raise ValueError(
                    f"Dual form is not unique to the equation {ie}, as it already appears"
                    f" in equation {duals.index(dual)}."
                )
            duals.append(dual)
            equation_list.append(equation)
        self.weak_forms = weak
        if sorting_primal is not None:
            self.primal_forms = tuple(sorted(primals, key=sorting_primal))
        else:
            self.primal_forms = tuple(primals)
        if sorting_dual is not None:
            self.dual_forms = tuple(sorted(duals, key=sorting_dual))
        else:
            self.dual_forms = tuple(duals)
        self.equations = tuple(equation_list[duals.index(d)] for d in self.dual_forms)

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
        duals: list[KFormDual] = []
        out_mat: list[list[str]] = []
        rhs: list[str] = []
        for ie, eq in enumerate(self.equations):
            p = _parse_form(eq.left)
            d = [k for k in p if isinstance(k, KFormDual)]
            if len(d) != 1:
                raise ValueError(
                    f"Equation {ie} does not have a single weight (weights were {d})."
                )
            o: dict[KFormPrimal, str] = {}
            for k in p:
                if not isinstance(k, KFormPrimal):
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
            rhs.append(fn[0] if fn is not None else "0")

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
    form: KForm, element: Element1D
) -> dict[KForm, npt.NDArray[np.float64] | None]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : KForm
        Form to evaluate.

    Returns
    -------
    dict of KForm -> array or None
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``None``, if it should be the identity operation.
    """
    if isinstance(form, KFormSum):
        left = _equation_1d(form.first, element)
        right = _equation_1d(form.second, element)
        for k in right:
            vr = right[k]
            if k in left:
                vl = left[k]
                if vl is not None and vr is not None:
                    left[k] = np.astype(vl + vr, np.float64)  # vl and vr are non-none
                elif vl is None:  # vr is, or is not None
                    left[k] = right[k]
            else:
                left[k] = right[k]  # k is not in left

        return left
    if isinstance(form, KFormInnerProduct):
        if form.weight.is_dual and form.function.is_primal:
            primal = _equation_1d(form.function, element)
            dual = _equation_1d(form.weight, element)
        elif form.weight.is_dual and form.function.is_primal:
            dual = _equation_1d(form.function, element)
            primal = _equation_1d(form.weight, element)
        else:
            raise ValueError(
                "Inner product terms must be such that the one part is dual only while"
                " another is primal only."
            )
        assert len(dual) == 1
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            mass: npt.NDArray[np.float64]
            if form.function.order == 0:
                mass = element.mass_node  # type: ignore
            elif form.function.order == 1:
                mass = element.mass_edge  # type: ignore
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 1D mesh."
                )
            if vd is not None:
                mass = np.astype(vd.T @ mass, np.float64)
            if vp is not None:
                mass = np.astype(mass @ vp, np.float64)
            primal[k] = mass
        return primal
    if isinstance(form, KFormDerivative):
        res = _equation_1d(form.form, element)
        e = element.incidence_primal_0()
        for k in res:
            rk = res[k]
            if rk is not None:
                res[k] = np.astype(e @ rk, np.float64)
            else:
                res[k] = e
        return res
    if isinstance(form, KFormPrimal):
        return {form: None}
    if isinstance(form, KFormDual):
        return {form: None}
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
