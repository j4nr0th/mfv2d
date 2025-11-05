"""Types for describint the system and differential forms and system."""

from collections.abc import Callable, Iterator
from typing import Any, Self, SupportsIndex

from mfv2d._mfv2d import _ElementFormSpecification
from mfv2d.kform import (
    KEquation,
    KExplicit,
    KForm,
    KFormDerivative,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KInteriorProduct,
    KInteriorProductNonlinear,
    KSum,
    KWeight,
    Term,
    UnknownFormOrder,
)


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


def _form_as_string(form: Term) -> dict[Term, str | None]:
    """Extract the string representations of the forms, as well as their dual weight.

    Parameters
    ----------
    form : Term
        Form, which is to be converted into strings.

    Returns
    -------
    dict of KForm -> (str or None)
        Mapping of forms in the equation and string representation of the operations
        performed on them.
    """
    if type(form) is KSum:
        left: dict[Term, str | None] = {}
        for c, p in form.pairs:
            right = _form_as_string(p)
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
        if isinstance(form.unknown_form, KHodge):
            unknown = _form_as_string(form.unknown_form.base_form)
        else:
            unknown = _form_as_string(form.unknown_form)
        weight = _form_as_string(form.weight_form)
        dv = form.weight
        for k in unknown:
            vw = weight[dv]
            vu = unknown[k]
            if form.unknown_form.is_primal:
                if form.weight.is_primal:
                    mtx_name = f"M({form.weight.order.value - 1})"
                else:
                    mtx_name = ""
            else:
                if form.weight.is_primal:
                    mtx_name = ""
                else:
                    mtx_name = f"M({form.weight.primal_order})^{{-1}}"

            unknown[k] = (
                (f"({vw})^T" if vw is not None else "")
                + (" @ " if vw is not None and mtx_name else "")
                + mtx_name
                + (" @ " if vu is not None and mtx_name else "")
                + (f"{vu}" if vu is not None else "")
            )

        assert dv not in unknown
        del weight
        return unknown

    if type(form) is KFormDerivative:
        res = _form_as_string(form.form)
        if form.form.is_primal:
            mtx_name = f"E({form.order.value - 1}, {form.order.value - 1 - 1})"
        else:
            mtx_name = f"E({form.primal_order + 1}, {form.primal_order})^T"
        for k in res:
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")
        return res

    if type(form) is KInteriorProduct:
        res = _form_as_string(form.form)
        mtx_name = (
            f"M({form.order.value - 1}, {form.form.order.value - 1};"
            f" {form.vector_field.__name__})"
        )
        for k in res:
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")
        return res

    if type(form) is KInteriorProductNonlinear:
        res = _form_as_string(form.form)
        mtx_name = (
            f"M({form.order.value - 1}, {form.form.order.value - 1};"
            f" {form.form_field.label})"
        )
        for k in res:
            res[k] = mtx_name + (f" @ {res[k]}" if res[k] is not None else "")

        assert form.form_field not in res
        if type(form.form) is KHodge:
            res[form.form_field] = (
                f"N({form.order.value - 1}, {form.form.order.value - 1};"
                f" {form.form.base_form.label})"
            )
        else:
            res[form.form_field] = (
                f"N({form.order.value - 1}, {form.form.order.value - 1};"
                f" {form.form.label})"
            )

        return res

    if type(form) is KHodge:
        res = _form_as_string(form.base_form)
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

    if isinstance(form, KExplicit):
        return {form: form.label}

    raise TypeError("Unknown type")


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
        unknowns: set[KFormUnknown] = set()
        weights: list[KWeight] = []
        equation_list: list[KEquation] = []
        for ie, equation in enumerate(equations):
            weight = equation.weight
            if weight in weights:
                raise ValueError(
                    f"Weight form is not unique to the equation {ie}, as it already"
                    f" appears in equation {weights.index(weight)}."
                )
            unknowns |= set(equation.left.unknowns + equation.right.unknowns)
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

    def __str__(self) -> str:
        """Create a printable representation of the object."""
        left_out_mat: list[list[str]] = list()
        right_out_mat: list[list[str]] = list()
        rhs_explicit: list[str] = list()
        for ie, eq in enumerate(self.equations):
            left_strings = _form_as_string(eq.left)
            right_strings = _form_as_string(eq.right)

            left_out_list: list[str] = []
            right_out_list: list[str] = []

            for left_form in self.unknown_forms.iter_forms():
                if left_form in left_strings:
                    v = left_strings[left_form]
                    assert v is not None
                    left_out_list.append(v)
                else:
                    left_out_list.append("0")

            for right_form in self.unknown_forms.iter_forms():
                if right_form in right_strings:
                    v = right_strings[right_form]
                    assert v is not None
                    right_out_list.append(v)
                else:
                    right_out_list.append("0")

            right_explicit = str()
            for form in right_strings:
                if not isinstance(form, KExplicit):
                    continue
                v = right_strings[form]
                assert v is not None
                if len(right_explicit) != 0:
                    right_explicit += " "

                    if v[0] == "-":
                        v = "- " + v[1:]
                    else:
                        right_explicit += "+ "
                right_explicit += v

            left_out_mat.append(left_out_list)
            right_out_mat.append(right_out_list)
            rhs_explicit.append(right_explicit)

        out_weights = [str(w) for w in self.weight_forms]
        out_unknown = [str(u) for u in self.unknown_forms]

        width_mat_left = tuple(
            max(len(row[i]) for row in left_out_mat) for i in range(len(self.equations))
        )
        width_mat_right = tuple(
            max(len(row[i]) for row in right_out_mat) for i in range(len(self.equations))
        )
        w_v = max(len(s) for s in out_unknown)
        w_d = max(len(s) for s in out_weights)
        w_r = max(len(s) for s in rhs_explicit)

        s = ""
        for ie in range(len(self.equations)):
            left_row = left_out_mat[ie]
            left_row_str = " | ".join(
                rs.rjust(w) for w, rs in zip(width_mat_left, left_row, strict=True)
            )

            if ie == (len(self.equations) // 2):
                middle = "="
                op = " + "
            else:
                middle = " "
                op = "   "
            s += (
                f"[{out_weights[ie].rjust(w_d)}]"
                + ("    (" if ie != 0 else "^T  (")
                + f"[{left_row_str}]  [{out_unknown[ie].rjust(w_v)}] "
                + middle
                + f" [{rhs_explicit[ie].rjust(w_r)}])"
            )

            if any(any(e != "0" for e in rrow) for rrow in right_out_mat):
                right_row = right_out_mat[ie]
                right_row_str = " | ".join(
                    rs.rjust(w) for w, rs in zip(width_mat_right, right_row, strict=True)
                )
                s += (
                    op
                    + f"[{out_weights[ie].rjust(w_d)}]"
                    + ("    (" if ie != 0 else "^T  (")
                    + f"[{right_row_str}]  [{out_unknown[ie].rjust(w_v)}] "
                )
            s += "\n"
        return s[:-1]  # strip the trailing new line.
