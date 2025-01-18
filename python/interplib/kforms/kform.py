"""Dealing with K-forms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from interplib.mimetic.mimetic1d import Element1D


@dataclass(frozen=True)
class KForm:
    """Generic K form."""

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

    def _matrix1d(self, element: Element1D) -> npt.NDArray[np.float64]:
        del element
        raise NotImplementedError


@dataclass(frozen=True)
class KFormPrimal(KForm):
    """K form on the primal basis."""

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return True

    def _matrix1d(self, element: Element1D) -> npt.NDArray[np.float64]:
        return np.eye(element.order + 1 - self.order)


@dataclass(frozen=True)
class KFormDual(KForm):
    """K form on the dual basis."""

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return False

    def __str__(self) -> str:
        """Return print-friendly representation of the object."""
        return f"{self.label}({self.order}*)"

    def _matrix1d(self, element: Element1D) -> npt.NDArray[np.float64]:
        return np.eye(element.order + 1 - self.order)


@dataclass(init=False, frozen=True)
class KFormDerivative(KForm):
    """Exterior derivative of the primal form."""

    form: KForm

    def __init__(self, form: KForm) -> None:
        object.__setattr__(self, "form", form)
        super().__init__(form.order + 1, label="d" + form.label)

    @property
    def is_primal(self) -> bool:
        """Check if form is primal."""
        return self.form.is_primal

    def _matrix1d(self, element: Element1D):
        o = self.form.order
        if o > 0:
            raise ValueError(
                f"Can not take an exterior derivative of {o}-form on a 1D mesh."
            )
        e = element.incidence_primal_0()
        return e @ self.form._matrix1d(element)
        # else:
        #     return (e @ self.form._matrix1d(element)).T


@dataclass(init=False, frozen=True)
class KFormInnerProduct(KForm):
    """Inner product of a primal and dual form."""

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

    def _matrix1d(self, element: Element1D) -> npt.NDArray[np.float64]:
        mass: npt.NDArray[np.float64]
        if self.weight.order == 0:
            mass = element.mass_node
        elif self.weight.order == 1:
            mass = element.mass_edge
        else:
            raise ValueError(
                f"Can not take inner product of {self.weight.order}-forms on a 1D mesh."
            )
        mat_w = self.weight._matrix1d(element)
        mat_f = self.function._matrix1d(element)
        return np.astype(
            mat_w.T @ mass @ mat_f,
            np.float64,
        )


@dataclass(init=False, frozen=True)
class KFormSum(KForm):
    """Sum of two KForms."""

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

    def _matrix1d(self, element: Element1D) -> npt.NDArray[np.float64]:
        return np.concatenate(
            (self.first._matrix1d(element), self.second._matrix1d(element)), axis=1
        )


@dataclass(frozen=True)
class KFormProjection:
    """Weigh a form with a dual form and implicilty project it."""

    dual: KFormDual
    func: tuple[str, Callable] | None = None


def _parse_form(form: KForm) -> tuple[KFormDual | None, dict[KForm, str | None]]:
    if isinstance(form, KFormSum):
        dl, left = _parse_form(form.first)
        dr, right = _parse_form(form.second)
        if dl != dr:
            raise ValueError(f"Sum of terms with differing dual weights {dl} and {dr}")
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
        return (dl, left)
    if isinstance(form, KFormInnerProduct):
        if form.weight.is_dual:
            dp, primal = _parse_form(form.function)
            dd, dual = _parse_form(form.weight)
        else:
            dd, dual = _parse_form(form.function)
            dp, primal = _parse_form(form.weight)
        if dp is not None or dd is None:
            raise ValueError(
                "Inner product terms must be such that the one part is dual only while"
                " another is primal only."
            )
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
        return dd, primal
    if isinstance(form, KFormDerivative):
        d, res = _parse_form(form.form)
        for k in res:
            res[k] = f"E({form.order}, {form.order - 1})" + (
                f" @ {res[k]}" if res[k] is not None else ""
            )
        return d, res
    if isinstance(form, KFormPrimal):
        return None, {form: None}
    if isinstance(form, KFormDual):
        return form, {form: None}
    raise TypeError("Unknown type")


def _evaluate_form_1d(
    form: KForm, element: Element1D
) -> dict[KForm, npt.NDArray[np.float64]]:
    if isinstance(form, KFormSum):
        left = _evaluate_form_1d(form.first, element)
        right = _evaluate_form_1d(form.second, element)
        for k in right:
            if k in left:
                left[k] = np.astype(left[k] + right[k], np.float64)
            else:
                left[k] = right[k]
        return left
    if isinstance(form, KFormInnerProduct):
        if form.weight.is_dual and form.function.is_primal:
            primal = _evaluate_form_1d(form.function, element)
            dual = _evaluate_form_1d(form.weight, element)
        elif form.weight.is_dual and form.function.is_primal:
            dual = _evaluate_form_1d(form.function, element)
            primal = _evaluate_form_1d(form.weight, element)
        else:
            raise ValueError(
                "Inner product terms must be such that the one part is dual only while"
                " another is primal only."
            )
        assert len(dual) == 1
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            mass: npt.NDArray[np.float64]
            if form.function.order == 0:
                mass = element.mass_node
            elif form.function.order == 1:
                mass = element.mass_edge
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 1D mesh."
                )

            primal[k] = np.astype(vd.T @ mass @ primal[k], np.float64)
        return primal
    if isinstance(form, KFormDerivative):
        res = _evaluate_form_1d(form.form, element)
        for k in res:
            res[k] = np.astype(element.incidence_primal_0() @ res[k], np.float64)
        return res
    if isinstance(form, KFormPrimal):
        return {form: np.eye(element.order + 1 - form.order)}
    if isinstance(form, KFormDual):
        return {form: np.eye(element.order + 1 - form.order)}
    raise TypeError("Unknown type")


@dataclass(init=False, frozen=True)
class KFormEquaton:
    """A left and right hand sides of an equation."""

    left: KForm
    right: KFormProjection
    variables: tuple[KForm, ...]

    def __init__(self, left: KForm, right: KFormProjection) -> None:
        d, p = _parse_form(left)
        if d != right.dual:
            raise ValueError(
                "Left and right side do not use the same dual form as a weight."
            )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "variables", tuple(k for k in p))


def _make_equations(equations: Sequence[KFormEquaton]) -> str:
    duals: list[KFormDual] = []
    parsed: list[dict[KForm, str]] = []
    rhs: list[str] = []
    for ie, eq in enumerate(equations):
        d, p = _parse_form(eq.left)
        if d is None:
            raise ValueError(f"Equation {ie} has no weight.")
        o: dict[KForm, str] = {}
        for k in p:
            v = p[k]
            if v is None:
                o[k] = "I"
            else:
                o[k] = v
        duals.append(d)
        parsed.append(o)
        fn = eq.right.func
        rhs.append(fn[0] if fn is not None else "0")

    all_inputs: set[KForm] = set()
    for p_dict in parsed:
        all_inputs |= set(p_dict.keys())

    if len(all_inputs) != len(equations):
        raise ValueError(
            f"There are {len(all_inputs)} unknown forms, but {len(equations)} equations."
        )
    if sum(i.order for i in all_inputs) != sum(d.order for d in duals):
        raise ValueError("The system is not square.")

    out_mat = [[p.get(v, "0") for v in all_inputs] for p in parsed]
    out_dual = [str(d) for d in duals]
    out_v = [str(iv) for iv in all_inputs]

    w_mat = max(max(len(s) for s in row_list) for row_list in out_mat)
    w_v = max(len(s) for s in out_v)
    w_d = max(len(s) for s in out_dual)
    w_r = max(len(s) for s in rhs)

    s = ""
    for ie in range(len(equations)):
        row = out_mat[ie]
        row_str = " | ".join(rs.rjust(w_mat) for rs in row)
        s += (
            f"[{out_dual[ie].rjust(w_d)}]"
            + ("    (" if ie != 0 else "^T  (")
            + f"[{row_str}]  [{out_v[ie].rjust(w_v)}] - [{rhs[ie].rjust(w_r)}]) = [0]\n"
        )
    return s


def _extract_rhs_1d(
    right: KFormProjection, element: Element1D
) -> npt.NDArray[np.float64]:
    fn = right.func
    n = element.order + 1 - right.dual.order
    if fn is None:
        return np.zeros(n)
    else:
        out_vec = np.empty(n)
        mass: npt.NDArray[np.float64]
        # TODO: find a nice way to do this
        if right.dual.order == 1:
            real_nodes = np.linspace(element.xleft, element.xright, n + 1)
            for i in range(n):
                out_vec[i] = quad(fn[1], real_nodes[i], real_nodes[i + 1])[0]
            mass = element.mass_edge
        elif right.dual.order == 0:
            out_vec[:] = fn[1](element.nodes)
            mass = element.mass_node
        return np.astype(mass @ out_vec, np.float64)


def _make_matrix_eqn(
    equations: Sequence[KFormEquaton], element: Element1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lhs: list[dict[KForm, npt.NDArray[np.float64]]] = []
    rhs: list[npt.NDArray[np.float64]] = []
    for ie, eq in enumerate(equations):
        expr = _evaluate_form_1d(eq.left, element)
        lhs.append(expr)
        rhs.append(_extract_rhs_1d(eq.right, element))

    all_inputs: set[KForm] = set()
    for p in lhs:
        all_inputs |= set(p.keys())

    if len(all_inputs) != len(equations):
        raise ValueError(
            f"There are {len(all_inputs)} unknown forms, but {len(equations)} equations."
        )

    input_orders = {form: element.order + 1 - form.order for form in all_inputs}

    in_order = tuple(form for form in all_inputs)
    in_offsets = np.pad(np.cumsum([input_orders[v] for v in in_order]), (1, 0))

    elm_vec = np.concatenate(rhs)
    out_offsets = np.pad(np.cumsum([v.size for v in rhs]), (1, 0))
    elm_mat = np.zeros((elm_vec.size, in_offsets[-1]))
    for ie, equation in enumerate(lhs):
        for form in equation:
            idx = in_order.index(form)
            elm_mat[
                out_offsets[ie] : out_offsets[ie + 1],
                in_offsets[idx] : in_offsets[idx + 1],
            ] += equation[form]

    return elm_mat, elm_vec


def _extract_forms(form: KForm) -> tuple[set[KFormPrimal], set[KFormDual]]:
    if isinstance(form, KFormPrimal):
        return ({form}, set())
    if isinstance(form, KFormDual):
        return (set(), {form})
    if isinstance(form, KFormDerivative):
        return _extract_forms(form.form)
    if isinstance(form, KFormSum):
        f1 = _extract_forms(form.first)
        f2 = _extract_forms(form.second)
        return (f1[0] | f2[0], f1[1] | f2[1])
    if isinstance(form, KFormInnerProduct):
        f1 = _extract_forms(form.function)
        f2 = _extract_forms(form.weight)
        return (f1[0] | f2[0], f1[1] | f2[1])
    raise TypeError(f"Invalid type {type(form)}")


class KFromSystem:
    """A system of equations of differential forms."""

    primal_forms: tuple[KFormPrimal, ...]
    dual_forms: tuple[KFormDual, ...]
    equations: tuple[KFormEquaton, ...]

    def __init__(
        self,
        *equations: KFormEquaton,
        sorting_primal: Callable[[KFormPrimal], Any] | None = None,
        sorting_dual: Callable[[KFormDual], Any] | None = None,
    ) -> None:
        primals: set[KFormPrimal] = set()
        duals: list[KFormDual] = []
        equation_list: list[KFormEquaton] = []
        for ie, equation in enumerate(equations):
            p, d = _extract_forms(equation.left)
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

        if sorting_primal is not None:
            self.primal_forms = tuple(sorted(primals, key=sorting_primal))
        else:
            self.primal_forms = tuple(primals)
        if sorting_dual is not None:
            self.dual_forms = tuple(sorted(duals, key=sorting_dual))
        else:
            self.dual_forms = tuple(duals)
        self.equations = tuple(equation_list[duals.index(d)] for d in self.dual_forms)

    def shape_1d(self, order: int) -> tuple[int, int]:
        """Return the shape of the system for the 1D case."""
        width = sum(order + 1 - d.order for d in self.dual_forms)
        height = sum(order + 1 - d.order for d in self.primal_forms)
        return (height, width)

    def offsets_1d(self, order: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Compute offsets of different forms and equations."""
        offset_forms = (0,) + tuple(
            accumulate(order + 1 - d.order for d in self.primal_forms)
        )
        offset_equations = (0,) + tuple(
            accumulate(order + 1 - d.order for d in self.dual_forms)
        )
        return offset_equations, offset_forms

    def __str__(self) -> str:
        """Create a printable representation of the object."""
        duals: list[KFormDual] = []
        out_mat: list[list[str]] = []
        rhs: list[str] = []
        for ie, eq in enumerate(self.equations):
            d, p = _parse_form(eq.left)
            if d is None:
                raise ValueError(f"Equation {ie} has no weight.")
            o: dict[KForm, str] = {}
            for k in p:
                v = p[k]
                if v is None:
                    o[k] = "I"
                else:
                    o[k] = v
            duals.append(d)
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
                mass = element.mass_node
            elif form.function.order == 1:
                mass = element.mass_edge
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
    system: KFromSystem, element: Element1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute element matrix and vector."""
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


# if __name__ == "__main__":
#     # Weights
#     v = KFormDual(0, "v")
#     q = KFormDual(1, "q")

#     # Functions
#     u = KFormPrimal(0, "u")
#     phi = KFormPrimal(1, "phi")
#     f = KFormPrimal(1, "f")

#     equation_u = KFormSum(
#         KFormInnerProduct(v, u), KFormInnerProduct(KFormDerivative(v), phi)
#     )
#     equation_f = KFormInnerProduct(q, KFormDerivative(u))

#     eq1 = KFormEquaton(equation_u, KFormProjection(v, None))
#     eq2 = KFormEquaton(equation_f, KFormProjection(q, ("f", lambda _: 0)))

#     e = Element1D(2, -1.0, +1.0)

#     equations = (eq1, eq2)

#     print(_make_equations(equations))
