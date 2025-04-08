"""Conversion and evaluation of kforms as a stack machine."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

from interplib.kforms.kform import (
    KFormDerivative,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KInteriorProduct,
    KSum,
    KWeight,
    Term,
    VectorFieldFunction,
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


@dataclass(frozen=True)
class InterProd(MatOp):
    """Compute interior product."""

    starting_order: int
    field_index: int


def simplify_expression(*operations: MatOp) -> list[MatOp]:
    """Simplify expressions as much as possible."""
    ops = list(operations)
    nops = len(ops)
    initial_nops = 0
    while initial_nops != nops:
        i = 0
        initial_nops = int(nops)
        r: Identity | Scale
        while i < nops:
            if nops - i >= 2 and (
                type(ops[i]) is MassMat and type(ops[i + 1]) is MassMat
            ):
                # Mass matrix and its inverse result in identity
                m1 = ops[i]
                m2 = ops[i + 1]
                assert type(m1) is MassMat and type(m2) is MassMat
                if m1.order == m2.order and m1.inv != m2.inv:
                    del ops[i + 1]
                    ops[i] = Identity()
                    nops -= 1
                    continue

            if nops - i >= 2 and (
                type(ops[i]) is Transpose and type(ops[i + 1]) is Transpose
            ):
                # Double transpose does nothing
                del ops[i + 1]
                del ops[i]
                nops -= 2
                continue

            if nops - i >= 2 and (
                type(ops[i]) is Identity
                and (
                    type(ops[i + 1]) is MassMat
                    or type(ops[i + 1]) is Incidence
                    or type(ops[i + 1]) is Push
                    or type(ops[i + 1]) is Scale
                )
            ):
                # Identity does nothing to these
                del ops[i]
                nops -= 1
                continue

            if nops - i >= 2 and (
                type(ops[i]) is Identity and (type(ops[i + 1]) is Transpose)
            ):
                # Transpose of identity does nothing
                del ops[i + 1]
                nops -= 1
                continue

            if nops - i >= 3 and (
                type(ops[i]) is Push
                and type(ops[i + 1]) is Scale
                and type(ops[i + 2]) is Transpose
            ):
                # Transpose of a fresh scale does nothing
                del ops[i + 2]
                nops -= 1
                continue

            if nops - i >= 2 and (
                (type(ops[i]) is Scale or type(ops[i]) is Identity)
                and (type(ops[i + 1]) is Scale or type(ops[i + 1]) is Identity)
            ):
                # Merge identities/scaling
                v1 = ops[i]
                v2 = ops[i + 1]

                if type(v1) is Identity:
                    if type(v2) is Identity:
                        r = Identity()
                    else:
                        assert type(v2) is Scale
                        r = v2
                else:
                    assert type(v1) is Scale
                    if type(v2) is Identity:
                        r = v1
                    else:
                        assert type(v2) is Scale
                        r = Scale(v1.k + v2.k)

                del ops[i + 1]
                ops[i] = r
                nops -= 1
                continue

            if nops - i >= 1 and (type(ops[i]) is Sum):
                # Sum of zero stack values is no-op
                sop = ops[i]
                assert type(sop) is Sum
                if sop.count == 0:
                    del ops[i]
                    nops -= 1
                    continue

            if nops - i >= 3 and (
                type(ops[i]) is Push
                and type(ops[i + 1]) is Identity
                and type(ops[i + 2]) is MatMul
            ):
                # Multiplication by identity is no-op
                del ops[i + 2]
                del ops[i + 1]
                del ops[i]
                nops -= 3
                continue

            if nops - i >= 3 and (
                type(ops[i]) is Push
                and type(ops[i + 1]) is Scale
                and type(ops[i + 2]) is MatMul
            ):
                # MatMul by Scale-only matrix is same as scaling itself
                s = ops[i + 1]
                del ops[i + 2]
                del ops[i + 1]
                ops[i] = s
                nops -= 2
                continue

            if nops - i >= 5 and (
                type(ops[i]) is Push
                and (type(ops[i + 1]) is Scale or type(ops[i + 1]) is Identity)
                and type(ops[i + 2]) is Push
                and (type(ops[i + 3]) is Scale or type(ops[i + 3]) is Identity)
                and type(ops[i + 4]) is Sum
            ):
                # Sum of two pure identity/scale matrices can be pre-computed
                v1 = ops[i + 1]
                v2 = ops[i + 3]

                if type(v1) is Identity:
                    if type(v2) is Identity:
                        r = Scale(2)
                    else:
                        assert type(v2) is Scale
                        r = Scale(v2.k + 1)
                else:
                    assert type(v1) is Scale
                    if type(v2) is Identity:
                        r = Scale(v1.k + 1)
                    else:
                        assert type(v2) is Scale
                        r = Scale(v1.k + v2.k)
                ops[i + 1] = r
                sop = ops[i + 4]
                assert type(sop) is Sum
                ops[i + 4] = Sum(sop.count - 1)
                del ops[i + 3]
                del ops[i + 2]
                nops -= 2
                continue

            else:
                i += 1

    return ops


def translate_equation(
    form: Term, vec_fields: Sequence[VectorFieldFunction], simplify: bool = True
) -> dict[Term, list[MatOp]]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : Term
        Form to evaluate.
    simplify : bool, default: True
        Simplify the expressions at the top level

    Returns
    -------
    dict of KForm -> array or float
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``float``, if it should be multiplication with a constant.
    """
    v = _translate_equation(form, vec_fields)
    if simplify:
        for k in v:
            v[k] = simplify_expression(*v[k])
    return v


def _translate_equation(
    form: Term, vec_fields: Sequence[VectorFieldFunction]
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
            right = _translate_equation(ip, vec_fields)
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
        # if isinstance(form.function, KHodge):
        #     unknown = _translate_equation(form.function.base_form)
        # else:
        unknown = _translate_equation(form.function, vec_fields)
        weight = _translate_equation(form.weight, vec_fields)
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
        res = _translate_equation(form.form, vec_fields)
        for k in res:
            vr = res[k]
            vr.append(Incidence(form.form.order, not form.form.is_primal))
        return res

    if type(form) is KHodge:
        unknown = _translate_equation(form.base_form, vec_fields)
        prime_order = form.primal_order
        for k in unknown:
            mass = MassMat(prime_order, not form.base_form.is_primal)
            uv = unknown[k]
            uv.append(mass)
            unknown[k] = uv
        return unknown

    if type(form) is KFormUnknown:
        return {form: [Identity()]}

    if type(form) is KWeight:
        return {form: [Identity()]}

    if type(form) is KInteriorProduct:
        res = _translate_equation(form.form, vec_fields)
        for k in res:
            vr = res[k]
            vr.append(InterProd(form.form.order, vec_fields.index(form.vector_field)))
        return res
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

        elif type(op) is InterProd:
            mat = f"M({op.starting_order - 1}, {op.starting_order}; {op.field_index})"
            if val is None:
                val = (1.0, mat)
            c, s = val
            val = (c, mat + " @ " + s)

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


class MatOpCode(IntEnum):
    """Operation codes."""

    INVALID = 0
    IDENTITY = 1
    MASS = 2
    INCIDENCE = 3
    PUSH = 4
    MATMUL = 5
    SCALE = 6
    TRANSPOSE = 7
    SUM = 8
    INTERPROD = 9


def _ctranslate(*ops: MatOp) -> list[MatOpCode | int | float]:
    """Translate the operations into C-friendly values."""
    out: list[MatOpCode | int | float] = list()
    for op in ops:
        if type(op) is Identity:
            out.append(MatOpCode.IDENTITY)
        elif type(op) is MassMat:
            out.append(MatOpCode.MASS)
            out.append(op.order)
            out.append(int(op.inv))
        elif type(op) is Incidence:
            out.append(MatOpCode.INCIDENCE)
            out.append(op.begin)
            out.append(int(op.dual))
        elif type(op) is Push:
            out.append(MatOpCode.PUSH)
        elif type(op) is Scale:
            out.append(MatOpCode.SCALE)
            out.append(op.k)
        elif type(op) is Transpose:
            out.append(MatOpCode.TRANSPOSE)
        elif type(op) is Sum:
            out.append(MatOpCode.SUM)
            out.append(op.count)
        elif type(op) is MatMul:
            out.append(MatOpCode.MATMUL)
        elif type(op) is InterProd:
            out.append(MatOpCode.INTERPROD)
            out.append(op.starting_order)
            out.append(op.field_index)
        else:
            raise TypeError("Unknown instruction")

    return out
