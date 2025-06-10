"""Conversion and evaluation of kforms as a stack machine."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from mfv2d.kform import (
    Function2D,
    KFormDerivative,
    KFormSystem,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KInteriorProduct,
    KInteriorProductNonlinear,
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
class Sum(MatOp):
    """Sum matrices together."""

    count: int


@dataclass(frozen=True)
class InterProd(MatOp):
    """Compute interior product."""

    starting_order: int
    field_index: int
    dual: bool
    adjoint: bool


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
            if (
                nops - i >= 2
                and type(ops[i]) is Identity
                and (type(ops[i + 1]) is not Sum and type(ops[i + 1]) is not Push)
                and (i != 0 or type(ops[i - 1]) is not Push)
            ):
                del ops[i]
                nops -= 1
                continue

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
                type(ops[i]) is Identity
                and (
                    type(ops[i + 1]) is MassMat
                    or type(ops[i + 1]) is Incidence
                    or type(ops[i + 1]) is Push
                    or type(ops[i + 1]) is Scale
                    or type(ops[i + 1]) is InterProd
                )
            ):
                # Identity does nothing to these
                del ops[i]
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
    form: Term,
    vec_fields: Sequence[Function2D | KFormUnknown],
    newton: bool,
    simplify: bool,
) -> dict[Term, list[MatOp]]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : Term
        Form to evaluate.
    vec_fields : Sequence of Function2D or KFormUnknown
        Sequence to use when determining the index of vector fields passed to evaluation
        function.
    newton : bool
        When ``True``, non-linear terms will yield terms two terms used to determine the
        derivative. Otherwise, only the terms to evaluate the value are returned.
    simplify : bool, default: True
        Simplify the expressions at the top level

    Returns
    -------
    dict of KForm -> array or float
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``float``, if it should be multiplication with a constant.
    """
    v = _translate_equation(
        form=form, vec_fields=vec_fields, newton=newton, transpose=False
    )
    if simplify:
        for k in v:
            v[k] = simplify_expression(*v[k])

    # We no longer use the Transpose instruction :)
    return v


def _translate_equation(
    form: Term,
    vec_fields: Sequence[Function2D | KFormUnknown],
    newton: bool,
    transpose: bool,
) -> dict[Term, list[MatOp]]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : Term
        Form to evaluate.
    vec_fields : Sequence of Function2D or KFormUnknown
        Sequence to use when determining the index of vector fields passed to evaluation
        function.
    newton : bool
        When ``True``, non-linear terms will yield terms two terms used to determine the
        derivative. Otherwise, only the terms to evaluate the value are returned.
    transpose : bool
        Should instructions be transposed.

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
            right = _translate_equation(
                form=ip, vec_fields=vec_fields, newton=newton, transpose=transpose
            )
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
        unknown = _translate_equation(
            form=form.unknown_form, vec_fields=vec_fields, newton=newton, transpose=False
        )
        weight = _translate_equation(
            form=form.weight_form, vec_fields=vec_fields, newton=newton, transpose=True
        )
        assert len(weight) == 1
        dv = tuple(v for v in weight.keys())[0]
        for k in unknown:
            vd = weight[dv]
            vp = unknown[k]
            order_p = form.unknown_form.primal_order
            order_d = form.weight_form.primal_order
            assert order_p == order_d

            if order_p == order_d:
                if form.unknown_form.is_primal:
                    if form.weight_form.is_primal:
                        mass = MassMat(order_p, False)
                    else:
                        mass = Identity()
                else:
                    if form.weight_form.is_primal:
                        mass = Identity()
                    else:
                        mass = MassMat(0, True)
            else:
                raise ValueError(
                    f"Order {form.unknown_form.order} can't be used on a 2D mesh."
                )
            if not transpose:
                result = vp + [mass] + vd
            else:
                result = vd + [mass] + vp
            unknown[k] = result
        return unknown

    if type(form) is KFormDerivative:
        res = _translate_equation(form.form, vec_fields, newton, transpose=transpose)
        for k in res:
            vr = res[k]
            if not transpose:
                vr.append(Incidence(form.form.order, not form.form.is_primal))
            else:
                res[k] = [
                    Incidence(1 - form.form.order, form.form.is_primal),
                    Scale(-1),
                ] + vr
        return res

    if type(form) is KHodge:
        unknown = _translate_equation(
            form=form.base_form, vec_fields=vec_fields, newton=newton, transpose=transpose
        )
        prime_order = form.primal_order
        for k in unknown:
            mass = MassMat(prime_order, not form.base_form.is_primal)
            uv = unknown[k]
            if not transpose:
                uv.append(mass)
            else:
                uv = [mass] + uv
            unknown[k] = uv
        return unknown

    if type(form) is KFormUnknown:
        return {form: [Identity()]}

    if type(form) is KWeight:
        return {form: [Identity()]}

    if (type(form) is KInteriorProduct) or (type(form) is KInteriorProductNonlinear):
        if transpose:
            ValueError("Can not transpose interior product instructions (yet?).")

        res = _translate_equation(
            form=form.form, vec_fields=vec_fields, newton=newton, transpose=transpose
        )

        if type(form) is KInteriorProduct:
            idx = vec_fields.index(form.vector_field)

        if type(form) is KInteriorProductNonlinear:
            idx = vec_fields.index(form.form_field)

        prod_instruct: list[MatOp] = [
            InterProd(form.form.order, idx, not form.form.is_primal, False)
        ]

        if not form.form.is_primal:
            prod_instruct = (
                [MassMat(form.primal_order - 1, True)]
                + prod_instruct
                + [MassMat(form.primal_order, True)]
            )

        for k in res:
            res[k].extend(prod_instruct)

        if type(form) is KInteriorProductNonlinear and newton:
            other_form = form.form_field
            if other_form in res:
                raise ValueError(
                    "Can not translate equation which would require sum with adjoint "
                    "non-linear product."
                )
            if type(form.form) is KHodge:
                base = form.form.base_form
            else:
                base = form.form

            qidx = vec_fields.index(base)
            extra_op: list[MatOp] = [
                InterProd(form.form.order, qidx, not form.form.is_primal, True)
            ]
            if not form.form.is_primal:
                extra_op += [MassMat(form.primal_order, True)]
            res[other_form] = extra_op

        return res

    raise TypeError("Unknown type")


class MassMatrixRequired(IntFlag):
    """Flags used to indicate which mass matrices need to be computed."""

    MASS_NODE = 1 << 0
    MASS_EDGE = 1 << 1
    MASS_SURF = 1 << 2
    MASS_INDE = 1 << 3
    MASS_IEDG = 1 << 4
    MASS_ISRF = 1 << 5


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
            if not op.dual:
                mat = f"M({op.starting_order}, {op.starting_order - 1}; {op.field_index})"
            else:
                mat = (
                    f"M*({op.starting_order}, {op.starting_order - 1}; {op.field_index})"
                )
            if val is None:
                val = (1.0, mat)
            else:
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
    SUM = 7
    INTERPROD = 8


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
        elif type(op) is Sum:
            out.append(MatOpCode.SUM)
            out.append(op.count)
        elif type(op) is MatMul:
            out.append(MatOpCode.MATMUL)
        elif type(op) is InterProd:
            out.append(MatOpCode.INTERPROD)
            out.append(op.starting_order)
            out.append(op.field_index)
            out.append(op.dual)
            out.append(op.adjoint)
        else:
            raise TypeError("Unknown instruction")

    return out


def translate_system(
    system: KFormSystem,
    vector_fields: Sequence[Function2D | KFormUnknown],
    newton: bool,
) -> tuple[tuple[tuple[MatOpCode | float | int, ...] | None, ...], ...]:
    """Create the two dimensional instruction array for the C code to execute."""
    bytecodes = [
        translate_equation(eq.left, vector_fields, newton=newton, simplify=True)
        for eq in system.equations
    ]

    codes: list[tuple[None | tuple[MatOpCode | float | int, ...], ...]] = list()
    for bite in bytecodes:
        row: list[tuple[MatOpCode | float | int, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(tuple(_ctranslate(*bite[f])))
            else:
                row.append(None)

        codes.append(tuple(row))
    return tuple(codes)


_CompiledCodeMatrix = Sequence[Sequence[Sequence[MatOpCode | int | float] | None]]


@dataclass(frozen=True, init=False)
class CompiledSystem:
    """System of equations compiled."""

    lhs_full: _CompiledCodeMatrix
    rhs_codes: _CompiledCodeMatrix | None
    linear_codes: _CompiledCodeMatrix
    nonlin_codes: _CompiledCodeMatrix | None

    @staticmethod
    def _compile_system_part(
        system: KFormSystem, expr: KSum | None, newton: bool
    ) -> tuple[tuple[MatOpCode | int | float, ...] | None, ...]:
        """Compile an expression and fit it into the row of the expression matrix."""
        if expr is None:
            return (None,) * len(system.unknown_forms)

        bytecode = translate_equation(expr, system.vector_fields, newton, True)

        return tuple(
            (tuple(_ctranslate(*bytecode[form])) if form in bytecode else None)
            for form in system.unknown_forms
        )

    def __init__(self, system: KFormSystem) -> None:
        # Split the system into the explicit, linear implicit, and non-linear implicit
        # explicit: list[tuple[tuple[float, KExplicit], ...] | None] = list()
        implicit_rhs: list[KSum | None] = list()
        linear_lhs: list[KSum | None] = list()
        nonlin_lhs: list[KSum | None] = list()

        # Loop over equations
        for equation in system.equations:
            assert not equation.left.explicit_terms
            # rhs_expl = equation.right.explicit_terms
            # explicit.append(rhs_expl if rhs_expl else None)
            rhs_impl = equation.right.implicit_terms
            implicit_rhs.append(KSum(*rhs_impl) if rhs_impl else None)
            linear, nonlinear = equation.left.split_terms_linear_nonlinear()
            linear_lhs.append(linear)
            nonlin_lhs.append(nonlinear)

        rhs_codes = tuple(
            CompiledSystem._compile_system_part(system, expr, newton=False)
            for expr in implicit_rhs
        )

        object.__setattr__(
            self,
            "rhs_codes",
            rhs_codes
            if any(any(e is not None for e in row) for row in rhs_codes)
            else None,
        )

        linear_codes = tuple(
            CompiledSystem._compile_system_part(system, expr, newton=False)
            for expr in linear_lhs
        )

        object.__setattr__(self, "linear_codes", linear_codes)

        nonlinear_codes = tuple(
            CompiledSystem._compile_system_part(system, expr, newton=True)
            for expr in nonlin_lhs
        )
        object.__setattr__(
            self,
            "nonlin_codes",
            nonlinear_codes
            if any(any(e is not None for e in row) for row in nonlinear_codes)
            else None,
        )
        object.__setattr__(
            self,
            "lhs_full",
            tuple(
                CompiledSystem._compile_system_part(system, eq.left, newton=False)
                for eq in system.equations
            ),
        )
