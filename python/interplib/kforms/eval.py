"""Conversion and evaluation of kforms as a stack machine."""

from collections.abc import Iterable
from dataclasses import dataclass

from interplib.kforms.kform import (
    KFormDerivative,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KSum,
    KWeight,
    Term,
)


@dataclass(frozen=True)
class MatOp:
    """Matrix operations which can be created."""


@dataclass(frozen=True)
class Identity(MatOp):
    """Identity operation."""


@dataclass(frozen=True)
class MassMat(MatOp):
    """Mass matrix multiplication."""

    order: int
    inv: bool


@dataclass(frozen=True)
class Incidence(MatOp):
    """Incidence matrix."""

    begin: int
    dual: int


@dataclass(frozen=True)
class Push(MatOp):
    """Push the matrix on the stack."""


@dataclass(frozen=True)
class MatMul(MatOp):
    """Multiply two matrices."""


@dataclass(frozen=True)
class Scale(MatOp):
    """Scale the matrix."""

    k: float


@dataclass(frozen=True)
class Transpose(MatOp):
    """Transpose of the matrix."""


@dataclass(frozen=True)
class Sum(MatOp):
    """Sum matrices together."""

    count: int


def translate_equation(
    form: Term,
) -> dict[Term, list[MatOp]]:
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
    mass: Identity | MassMat
    if type(form) is KSum:
        out: dict[Term, list[MatOp]] = {}
        accum: dict[Term, list[MatOp]] = {}
        counts: dict[Term, int] = {}
        for c, ip in form.pairs:
            right = translate_equation(ip)
            if c != 1.0:
                for f in right:
                    right[f].append(Scale(c))

            for k in right:
                vr = right[k]
                if k in accum:
                    accum[k].append(Push())
                    accum[k].extend(vr)
                    counts[k] += 1
                else:
                    accum[k] = vr  # k is not in left
                    counts[k] = 0

        for k in accum:
            out[k] = accum[k]
            cnt = counts[k]
            if cnt > 0:
                out[k].append(Sum(cnt))

        return out

    if type(form) is KInnerProduct:
        unknown: dict[Term, list[MatOp]]
        if isinstance(form.function, KHodge):
            unknown = translate_equation(form.function.base_form)
        else:
            unknown = translate_equation(form.function)
        weight = translate_equation(form.weight)
        assert len(weight) == 1
        dv = tuple(v for v in weight.keys())[0]
        for k in unknown:
            vd = weight[dv]
            vp = unknown[k]
            order_p = form.function.primal_order
            order_d = form.weight.primal_order
            assert order_p == order_d

            if order_p == order_d:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = MassMat(order_p, False)
                    else:
                        mass = Identity()
                else:
                    if form.weight.is_primal:
                        mass = Identity()
                    else:
                        mass = MassMat(0, True)
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 2D mesh."
                )

            result = vp + [mass, Push()] + vd + [Transpose(), MatMul()]

            unknown[k] = result
        return unknown

    if type(form) is KFormDerivative:
        res = translate_equation(form.form)
        for k in res:
            vr = res[k]
            vr.append(Incidence(form.form.order, not form.form.is_primal))
        return res

    if type(form) is KHodge:
        unknown = translate_equation(form.base_form)
        prime_order = form.primal_order
        for k in unknown:
            mass = MassMat(prime_order, form.is_primal)
            uv = unknown[k]
            uv.append(mass)
            unknown[k] = uv
        return unknown

    if type(form) is KFormUnknown:
        return {form: [Identity()]}

    if type(form) is KWeight:
        return {form: [Identity()]}

    raise TypeError("Unknown type")


def print_eval_procedure(expr: Iterable[MatOp], /) -> str:
    """Print how the terms would be evaluated."""
    stack: list[tuple[float, str]] = []
    val: tuple[float, str] | None = None
    for op in expr:
        if type(op) is MassMat:
            mat = f"M({op.order})" + ("^{-1}" if op.inv else "")
            if val is not None:
                c, s = val
                val = (c, mat + " @ " + s)
            else:
                val = (1.0, mat)

        elif type(op) is Incidence:
            mat = f"E({op.begin + 1}, {op.begin})" + ("*" if op.dual else "")
            if val is not None:
                c, s = val
                val = (c, mat + " @ " + s)
            else:
                val = (1.0, mat)

        elif type(op) is Push:
            if val is None:
                raise ValueError("Invalid Push operation.")
            stack.append(val)
            val = None

        elif type(op) is Scale:
            if val is None:
                val = (op.k, "I")
            else:
                c, s = val
                val = (c * op.k, s)

        elif type(op) is MatMul:
            k, mat = stack.pop()
            if val is None:
                raise ValueError("Invalid MatMul operation.")

            c, s = val
            val = (c * k, s + " @ " + mat)

        elif type(op) is Transpose:
            if val is None:
                raise ValueError("Invalid Transpose operation.")
            c, s = val
            s = "(" + s + ")^T"
            val = (c, s)

        elif type(op) is Sum:
            n = op.count
            if n <= 0:
                raise ValueError("Sum must be of a non-zero number of matrices.")
            if val is None:
                raise ValueError("Invalid Sum operation.")
            if len(stack) < n:
                raise ValueError(
                    f"Not enough matrices on the stack to Sum ({len(stack)} on stack,"
                    f" but {n} should be summed)."
                )
            c, s = val
            s = ("" if c > 0 else "-") + f"{abs(c):g} {s}"
            for _ in range(n):
                k, mat = stack.pop()
                s += (" + " if k > 0 else " - ") + f"{abs(k):g} {mat}"
            val = (1.0, s)

        elif type(op) is Identity:
            if val is None:
                val = (1.0, "I")

        else:
            raise TypeError("Unknown operation.")

    if len(stack):
        raise ValueError(f"{len(stack)} matrices still on the stack.")
    if val is None:
        return "I"
    c, s = val
    return f"{c:g} ({s})"


def extract_mass_matrices(*ops: MatOp) -> set[MassMat]:
    """Extract mass matrices which will be needed."""
    return set(op for op in filter(lambda op: isinstance(op, MassMat), ops))  # type: ignore
