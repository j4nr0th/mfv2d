"""Conversion and evaluation of kforms as a stack machine.

This module is concerned with converting expressions from the representation of the
:mod:`mfv2d.kform` module into a sequence of instructions which can be executed on a
stack machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

from mfv2d.kform import (
    Function2D,
    KBoundaryProjection,
    KElementProjection,
    KForm,
    KFormDerivative,
    KFormUnknown,
    KInnerProduct,
    KInteriorProduct,
    KInteriorProductLowered,
    KSum,
    KWeight,
    UnknownFormOrder,
    extract_base_form,
)
from mfv2d.system import KFormSystem


@dataclass(frozen=True)
class MatOp:
    """Matrix operations which can be created.

    This is just a base class for all other matrix operations.
    """


@dataclass(frozen=True)
class Identity(MatOp):
    """Identity operation.

    Do nothing to the matrix.
    """


@dataclass(frozen=True)
class MassMat(MatOp):
    """Mass matrix multiplication.

    Multiply by either mass matrix or its inverse.

    Parameters
    ----------
    order : UnknownFormOrder
        Order of the k-form for the mass matrix.

    inv : bool
        Should the matrix be inverted.
    """

    order: UnknownFormOrder
    inv: bool


@dataclass(frozen=True)
class Incidence(MatOp):
    """Incidence matrix.

    Specifies application of an incidence matrix.

    Parameters
    ----------
    begin : UnknownFormOrder
        Order of the k-form for the incidence matrix from which to apply it.

    transpose : bool
        Should the incidence matrix be transposed.
    """

    begin: UnknownFormOrder
    transpose: int


@dataclass(frozen=True)
class Push(MatOp):
    """Push the matrix on the stack.

    Used for matrix multiplication and summation.
    """


@dataclass(frozen=True)
class Scale(MatOp):
    """Scale the matrix.

    Mutliply the entire matrix with a constant.

    Parameters
    ----------
    k : float
        Value of the constant by which to scale the matrix.
    """

    k: float


@dataclass(frozen=True)
class Sum(MatOp):
    """Sum matrices together.

    Sum the top ``count`` matrices on the stack with the current matrix.

    Parameters
    ----------
    count : int
        Number of matrices to sum to the current matrix. As such must be greater
        than zero.
    """

    count: int


@dataclass(frozen=True)
class InterProd(MatOp):
    """Compute interior product.

    This is the most complicated operation.

    Parameters
    ----------
    starting_order : UnknownFormOrder
        Order of the k-form to which the interior product should be applied.

    field : str or Function2D
        Index of the vector/scalar field from which the values of are taken.

    transpose : bool
        Should the matrix be transposed.

    adjoint : bool
        Should the adjoint interior product be applied, which is used by
        the Newton-Raphson solver.
    """

    starting_order: UnknownFormOrder
    field: str | Function2D
    transpose: bool


def simplify_expression(*operations: MatOp) -> list[MatOp]:
    """Simplify expressions as much as possible.

    Tries to merge operations that can be combined, and removes/simplifies
    as much as possible.

    Parameters
    ----------
    *operations : MatOp
        Sequence of operations to simplify.

    Returns
    -------
    list of MatOp
        Simplified sequence of operations, which should be equivalent to the
        input.
    """
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
                # Identity on anything except on sum or push is a no-op
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
                        r = Scale(v1.k * v2.k)

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

            if i > 1 and (type(ops[i]) is Identity and type(ops[i - 1]) is not Push):
                # Identity following anything but push is a no-op
                del ops[i]
                nops -= 1

            else:
                i += 1

    return ops


def translate_implicit_ksum(ks: KSum) -> dict[KFormUnknown, list[MatOp]]:
    """Compute matrix operations on different unknown blocks.

    Parameter
    ---------
    ks : KSum
        Sum to translate.

    Returns
    -------
    dict of (KFormUnknown, list[MatOp])
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``float``, if it should be multiplication with a constant.
    """
    # Get instructions for each InnerProduct
    instructions: dict[KFormUnknown, list[list[MatOp]]] = dict()
    for k, ip in ks.pairs:
        if type(ip) is not KInnerProduct:
            raise TypeError("Can only translate implicit terms.")

        ops: list[MatOp] = _translate_inner_prod(ip)
        if k != 1.0:
            if len(ops) > 1:
                ops += [Scale(k)]
            else:
                assert ops[0] == Identity()
                ops = [Scale(k)]
        base = extract_base_form(ip.unknown_form)
        assert type(base) is KFormUnknown
        if base not in instructions:
            instructions[base] = list()
        instructions[base].append(ops)

    # Merge instructions as needed
    v: dict[KFormUnknown, list[MatOp]] = dict()
    for form in instructions:
        op_list = instructions[form]
        assert form not in v
        merged: list[MatOp] = op_list[0]
        cnt = len(op_list)
        for i in range(cnt - 1):
            new_ops = op_list[i + 1]
            merged.append(Push())
            merged.extend(new_ops)
        if cnt > 1:
            merged.append(Sum(cnt - 1))
        v[form] = simplify_expression(*merged)

    return v


def explicit_ksum_as_string(ks: KSum) -> str:
    """Make the explicit terms in the KSum into a string.

    Parameter
    ---------
    ks : KSum
        Sum to translate.

    Returns
    -------
    str
        String of explicit terms.
    """
    res = str()

    for k, ip in ks.pairs:
        out = str()
        match ip:
            case KInnerProduct():
                # Not explicit
                continue

            case KElementProjection() as ep:
                if ep.func is None:
                    continue
                out = "E" + ep.label

            case KBoundaryProjection() as bp:
                if bp.func is None:
                    continue
                out = "B" + bp.label

        if k != 1.0:
            out = f"{abs(k):g} * {out}"
        if k < 0:
            out = "- " + out
        else:
            out = "+ " + out
        res = res + " " + out

    return res.strip()


def _translate_form(form: KForm) -> list[MatOp]:
    """Translate form into a sequence of matrix operations to be applied of form DoFs."""
    match form:
        case KFormUnknown():
            return [Identity()]
        case KWeight():
            return [Identity()]
        case KFormDerivative() as fd:
            return _translate_form(fd.form) + [Incidence(fd.form.order, False)]
        case KInteriorProduct() as ip:
            return _translate_form(ip.form) + [
                InterProd(ip.form.order, ip.vector_field, False),
                MassMat(ip.order, True),
            ]
        case KInteriorProductLowered() as ipl:
            return _translate_form(ipl.form) + [
                InterProd(ipl.form.order, ipl.form_field.label, False),
                MassMat(ipl.order, True),
            ]
        case _:
            raise TypeError(f"Unknown form type {type(form)}")


def _translate_inner_prod(inner: KInnerProduct) -> list[MatOp]:
    """Translate inner product."""
    unknown_ops = _translate_form(inner.unknown_form)
    weight_ops = _translate_form(inner.weight_form)

    # Add mass matrix
    unknown_ops.append(MassMat(inner.unknown_form.order, False))

    # Add transposed weight ops to unknown ops.
    for op in reversed(weight_ops):
        match op:
            case Identity():
                # Ignore
                pass
            case Incidence() as inc:
                unknown_ops.append(Incidence(inc.begin, not inc.transpose))
            case MassMat() | Scale():
                # Symmetric, so no transpose
                unknown_ops.append(op)
            case InterProd() as ip:
                unknown_ops.append(
                    InterProd(ip.starting_order, ip.field, not ip.transpose)
                )
            case _:
                raise TypeError("Unexpected type for an inner product instructions.")

    if len(unknown_ops) > 1:
        # Skip the first identity
        return unknown_ops[1:]
    return unknown_ops


class MatOpCode(IntEnum):
    """Operation codes.

    Notes
    -----
    These values must be kept in sync with the ``matrix_op_t`` enum in the C code,
    since that is how Python and C communicate with each other.
    """

    INVALID = 0
    IDENTITY = 1
    MASS = 2
    INCIDENCE = 3
    PUSH = 4
    SCALE = 5
    SUM = 6
    INTERPROD = 7


_TranslatedInstruction = tuple[MatOpCode | int | float | Function2D | str, ...]
_TranslatedBlock = Sequence[_TranslatedInstruction]
_TranslatedRow = Sequence[_TranslatedBlock | None]
_TranslatedSystem2D = Sequence[_TranslatedRow]


def translate_to_c_instructions(*ops: MatOp) -> _TranslatedBlock:
    """Translate the operations into C-compatible values.

    This translation is done since the C code can't handle arbitrary Python objects
    and instead only deals with integers (or int enums) and floats.

    Parameters
    ----------
    *ops : MatOp
        Operations to translate.

    Returns
    -------
    list of MatOpCode | int | float
        List of translated operations.
    """
    out: list[_TranslatedInstruction] = list()
    for op in ops:
        translated: _TranslatedInstruction
        match op:
            case Identity():
                translated = (MatOpCode.IDENTITY,)

            case MassMat():
                translated = (MatOpCode.MASS, op.order, op.inv)

            case Incidence():
                translated = (MatOpCode.INCIDENCE, op.begin, op.transpose)

            case Push():
                translated = (MatOpCode.PUSH,)

            case Scale():
                translated = (MatOpCode.SCALE, op.k)

            case Sum():
                translated = (MatOpCode.SUM, op.count)

            case InterProd():
                translated = (
                    MatOpCode.INTERPROD,
                    op.starting_order,
                    op.field,
                    op.transpose,
                )

            case _:
                raise TypeError(f"Unknown instruction type {type(op).__name__}.")

        out.append(translated)

    return tuple(out)


def translate_system(system: KFormSystem) -> _TranslatedSystem2D:
    """Create the two-dimensional instruction array for the C code to execute."""
    bytecodes = [translate_implicit_ksum(eq.left) for eq in system.equations]

    codes: list[_TranslatedRow] = list()
    for bite in bytecodes:
        row: list[_TranslatedBlock | None] = list()
        for f in system.unknown_forms.iter_forms():
            if f in bite:
                row.append(translate_to_c_instructions(*bite[f]))
            else:
                row.append(None)

        codes.append(tuple(row))
    return tuple(codes)


@dataclass(frozen=True, init=False)
class CompiledSystem:
    """System of equations compiled.

    This is a convenience class which first compiles the system and splits
    it into explicit, linear implicit, and non-linear implicit equations, which
    are then further used by different parts of the solver.

    Parameters
    ----------
    system : KFormSystem
        System to compile.
    """

    lhs_codes: _TranslatedSystem2D
    """All left-hand side codes of the equations. When evaluated, this will
        produce the full left side of the equation."""
    rhs_codes: _TranslatedSystem2D | None
    """If not ``None``, contains the right-hand side codes of the equations."""
    linear_codes: _TranslatedSystem2D
    """All left-hand side codes of the equations, which are linear."""
    nonlin_codes: _TranslatedSystem2D | None
    """If not ``None``, contains the non-linear codes that can be used
        for Newton-Raphson solver."""

    @staticmethod
    def _compile_system_part(system: KFormSystem, expr: KSum | None) -> _TranslatedRow:
        """Compile an expression and fit it into the row of the expression matrix."""
        if expr is None:
            return (None,) * len(system.unknown_forms)

        bytecodes = translate_implicit_ksum(expr)

        row: list[_TranslatedBlock | None] = list()
        for f in system.unknown_forms.iter_forms():
            if f in bytecodes:
                row.append(translate_to_c_instructions(*bytecodes[f]))
            else:
                row.append(None)

        return tuple(row)

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
            CompiledSystem._compile_system_part(system, expr) for expr in implicit_rhs
        )

        object.__setattr__(
            self,
            "rhs_codes",
            rhs_codes
            if any(any(e is not None for e in row) for row in rhs_codes)
            else None,
        )

        linear_codes = tuple(
            CompiledSystem._compile_system_part(system, expr) for expr in linear_lhs
        )

        object.__setattr__(self, "linear_codes", linear_codes)

        nonlinear_codes = tuple(
            CompiledSystem._compile_system_part(system, expr) for expr in nonlin_lhs
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
            "lhs_codes",
            tuple(
                CompiledSystem._compile_system_part(system, eq.left)
                for eq in system.equations
            ),
        )


def _translate_codes_to_str(*ops: MatOp) -> str:
    """Translate operation codes to a string."""
    out: list[str] = list()

    for op in reversed(ops):
        match op:
            case Identity():
                out += "I"
            case MassMat() as mm:
                base = f"M({mm.order.value - 1})"
                if mm.inv:
                    out.append(f"({base})^{{-1}}")
                else:
                    out.append(base)
            case Incidence() as inc:
                base = f"E({inc.begin.value}, {inc.begin.value - 1})"
                if inc.transpose:
                    out.append(f"({base})^T")
                else:
                    out.append(base)
            case InterProd() as ip:
                base = (
                    f"P({ip.starting_order.value - 2}, {ip.starting_order.value - 1},"
                    f" {ip.field if type(ip.field) is str else ip.field.__name__})"  # type: ignore[union-attr]
                )
                if ip.transpose:
                    out.append(f"({base})^T")
                else:
                    out.append(base)
            case Scale() as sc:
                out.append(str(sc.k))
            case _:
                raise TypeError(f"Unsupported instruction type {type(op)}.")

    return " ".join(out)


def _translate_expr_to_str(*ops: MatOp) -> str:
    """Translate operations which may include Sum and Push into a string."""
    s = ops[-1]
    if type(s) is not Sum:
        return _translate_codes_to_str(*ops)

    out = str()
    begin = 0
    c = 0
    for i, op in enumerate(ops):
        if type(op) is Push:
            subsection = _translate_codes_to_str(*ops[begin:i])
            out += f"+ ({subsection})"
            begin = i + 1
            c += 1

    assert c == s.count

    # Do it for the last one, which is not ended by Push
    subsection = _translate_codes_to_str(*ops[begin:-1])
    out += f" + ({subsection})"

    return out.strip()


def system_as_string(system: KFormSystem, /) -> str:
    """Create the string representation of the system."""
    left_bytecodes = [translate_implicit_ksum(eq.left) for eq in system.equations]
    left_rows = bytecode_matrix_as_rows(system, left_bytecodes)

    right_bytecodes = [
        (
            translate_implicit_ksum(KSum(*eq.right.implicit_terms))
            if eq.right.implicit_terms
            else dict()
        )
        for eq in system.equations
    ]
    right_rows = bytecode_matrix_as_rows(system, right_bytecodes)

    # Add brackets and unknowns to
    unknowns = [str(w.base_form) for w in system.weight_forms]
    uw = max([len(u) for u in unknowns])
    unknowns = [u.ljust(uw) for u in unknowns]
    left_rows = [f"[{row}] [{u}]" for u, row in zip(unknowns, left_rows, strict=True)]
    right_rows = [f"[{row}] [{u}]" for u, row in zip(unknowns, right_rows, strict=True)]

    explicit_rows = [explicit_ksum_as_string(eq.right) for eq in system.equations]
    ew = 0
    n = len(explicit_rows)
    for i in range(n):
        ew = max(ew, len(explicit_rows[i]))
    for i in range(n):
        if len(explicit_rows[i]) == 0:
            explicit_rows[i] = "+ 0"
        explicit_rows[i] = "[" + explicit_rows[i].ljust(ew) + "]"

    full_rows = [
        l_row
        + (" = " if row == n // 2 else "   ")
        + r_exp
        + (" + " if row == n // 2 else "   ")
        + r_row
        for row, (l_row, r_row, r_exp) in enumerate(
            zip(left_rows, right_rows, explicit_rows, strict=True)
        )
    ]

    return "\n".join(full_rows)


def bytecode_matrix_as_rows(
    system: KFormSystem, bytecodes: Sequence[Mapping[KFormUnknown, list[MatOp]]]
) -> list[str]:
    """Extract expression rows."""
    expression_matrix = [
        [
            (_translate_expr_to_str(*codes[form]) if form in codes else "0")
            for form in system.unknown_forms.iter_forms()
        ]
        for codes in bytecodes
    ]
    n = len(expression_matrix)
    for col in range(n):
        col_w = 1
        for row in range(n):
            w = len(expression_matrix[row][col])
            col_w = max(w, col_w)
        for row in range(n):
            expression_matrix[row][col] = expression_matrix[row][col].ljust(col_w)

    left_rows = [" | ".join(expr_row) for expr_row in expression_matrix]
    return left_rows


if __name__ == "__main__":
    vor = KFormUnknown("vor", UnknownFormOrder.FORM_ORDER_0)
    vel = KFormUnknown("vel", UnknownFormOrder.FORM_ORDER_1)
    pre = KFormUnknown("pre", UnknownFormOrder.FORM_ORDER_2)

    w_vor = vor.weight
    w_vel = vel.weight
    w_pre = pre.weight

    def vor_src(x, y):
        """Test vorticit function."""
        return 0 * x * y

    import numpy as np

    def vel_src(x, y):
        """Test velocity function."""
        return np.stack((0 * x * y, 0 * x * y), axis=-1)

    def pre_src(x, y):
        """Test pressure function."""
        return 0 * x * y

    system = KFormSystem(
        (w_vor @ vor) + (w_vor.derivative @ vel) == w_vor ^ vor_src,
        (w_vel @ vor.derivative) + (w_vel.derivative @ pre)
        == ((vel * w_vel) @ vor) + ((vel_src * w_vel) @ vor),
        (w_pre @ vel.derivative) == w_pre @ pre_src,
    )

    print(system_as_string(system))
