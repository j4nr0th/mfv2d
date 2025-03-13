"""JAX-friendly evaluation of element matrices."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from interplib.kforms.eval import (
    Identity,
    Incidence,
    MassMat,
    MatMul,
    MatOp,
    Push,
    Scale,
    Sum,
    Transpose,
)
from interplib.mimetic.mimetic2d import BasisCache


# @jax.jit
def compute_jacobian(
    bl: jax.Array,
    br: jax.Array,
    tr: jax.Array,
    tl: jax.Array,
    xi: npt.NDArray,
    eta: npt.NDArray,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array], jax.Array]:
    r"""Compute the jacobian terms.

    Parameters
    ----------
    bl : (float, float)
        Coordinates of the bottom left corner.
    br : (float, float)
        Coordinates of the bottom right corner.
    tr : (float, float)
        Coordinates of the top right corner.
    tl : (float, float)
        Coordinates of the top left corner.
    xi : (N, M) array or (N, 1) array
        Array of first computational coordinate where to evaluate the Jacobian.
    eta : (N, M) array or (1, M) array
        Array of second computational coordinate where to evaluate the Jacobian.

    Returns
    -------
    (N, M, 5) array
        Array of Jacobian terms at the specified positions. The five entries
        are the following:

        - :math:`\frac{\partial x}{\partial \xi}`
        - :math:`\frac{\partial y}{\partial \xi}`
        - :math:`\frac{\partial x}{\partial \eta}`
        - :math:`\frac{\partial y}{\partial \eta}`
        - determinant of the Jacobian
    """
    dx_dxi = ((br[0] - bl[0]) * (1 - eta) + (tr[0] - tl[0]) * (1 + eta)) / 4
    dx_deta = ((tl[0] - bl[0]) * (1 - xi) + (tr[0] - br[0]) * (1 + xi)) / 4
    dy_dxi = ((br[1] - bl[1]) * (1 - eta) + (tr[1] - tl[1]) * (1 + eta)) / 4
    dy_deta = ((tl[1] - bl[1]) * (1 - xi) + (tr[1] - br[1]) * (1 + xi)) / 4
    det = dx_dxi * dy_deta - dx_deta * dy_dxi
    return ((dx_dxi, dy_dxi), (dx_deta, dy_deta), det)


class MatrixType(IntEnum):
    """Current type of the matrix."""

    MATRIX_TYPE_INVALID = 0
    MATRIX_TYPE_IDENTITY = 1
    MATRIX_TYPE_INCIDENCE = 2
    MATRIX_TYPE_FULL = 3


@dataclass(frozen=True)
class MatrixBase:
    """Basic computational matrix."""

    coeff: float


@dataclass(frozen=True)
class MatrixIncidence(MatrixBase):
    """Incidence matrix."""

    start_order: int
    dual: bool


@dataclass(frozen=True)
class MatrixFull(MatrixBase):
    """Full matrix."""

    data: jax.Array


AnyMatrix = MatrixBase | MatrixFull | MatrixIncidence


def full_e10(order: int) -> jax.Array:
    """Full E10 matrix."""
    n_nodes = order + 1
    n_lines = order
    e = jnp.zeros(((n_nodes * n_lines + n_lines * n_nodes), (n_nodes * n_nodes)))

    for row in range(n_nodes):
        for col in range(n_lines):
            e = e.at[row * n_lines + col, n_nodes * row + col].set(+1)
            e = e.at[row * n_lines + col, n_nodes * row + col + 1].set(-1)

    for row in range(n_lines):
        for col in range(n_nodes):
            e = e.at[n_nodes * n_lines + row * n_nodes + col, n_nodes * row + col].set(-1)
            e = e.at[
                n_nodes * n_lines + row * n_nodes + col, n_nodes * (row + 1) + col
            ].set(+1)

    return e


def full_e21(order: int) -> jax.Array:
    """Full E21 matrix."""
    n_nodes = order + 1
    n_lines = order
    e = jnp.zeros(((n_lines * n_lines), (n_nodes * n_lines + n_lines * n_nodes)))

    for row in range(n_lines):
        for col in range(n_lines):
            e = e.at[row * n_lines + col, n_lines * row + col].set(+1)
            e = e.at[row * n_lines + col, n_lines * (row + 1) + col].set(-1)
            e = e.at[row * n_lines + col, n_nodes * n_lines + n_nodes * row + col].set(+1)
            e = e.at[
                row * n_lines + col, n_nodes * n_lines + n_nodes * row + col + 1
            ].set(-1)

    return e


def apply_e10(order: int, other: jax.Array) -> jax.Array:
    """Apply the E10 matrix to the given input.

    Calling this function is equivalent to left multiplying by E10.
    """
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((2 * order * (order + 1), other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        for row in range(n_nodes):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_nodes * row + col
                col_e2 = n_nodes * row + col + 1
                out = out.at[row_e, i_col].set(
                    other[col_e1, i_col] - other[col_e2, i_col]
                )

        for row in range(n_lines):
            for col in range(n_nodes):
                row_e = n_nodes * n_lines + row * n_nodes + col
                col_e1 = n_nodes * (row + 1) + col
                col_e2 = n_nodes * row + col
                out = out.at[row_e, i_col].set(
                    other[col_e1, i_col] - other[col_e2, i_col]
                )

    return out


def apply_e10_t(order: int, other: jax.Array) -> jax.Array:
    """Apply the E10 transpose matrix to the given input.

    Calling this function is equivalent to left multiplying by E10 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros(((order + 1) ** 2, other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        # Nodes with lines on their right
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col
                col_e1 = n_lines * row + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] + other[col_e1, i_col])

        # Nodes with lines on their left
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col + 1
                col_e1 = n_lines * row + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] - other[col_e1, i_col])

        # Nodes with lines on their top
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = row * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] - other[col_e1, i_col])

        # Nodes with lines on their bottom
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = (row + 1) * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] + other[col_e1, i_col])

    return out


def apply_e21(order: int, other: jax.Array) -> jax.Array:
    """Apply the E21 matrix to the given input.

    Calling this function is equivalent to left multiplying by E21.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((order**2, other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_lines * row + col  # +
                col_e2 = n_lines * (row + 1) + col  # -
                col_e3 = n_nodes * n_lines + n_nodes * row + col  # +
                col_e4 = n_nodes * n_lines + n_nodes * row + col + 1  # -
                out = out.at[row_e, i_col].set(
                    other[col_e1, i_col]
                    - other[col_e2, i_col]
                    + other[col_e3, i_col]
                    - other[col_e4, i_col]
                )

    return out


def apply_e21_t(order: int, other: jax.Array) -> jax.Array:
    """Apply the E21 transposed matrix to the given input.

    Calling this function is equivalent to left multiplying by E21 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros(((2 * order * (order + 1)), other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        # Lines with surfaces on the top
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = row * n_lines + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] + other[col_e1, i_col])

        # Lines with surfaces on the bottom
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (row + 1) * n_lines + col
                col_e1 = row * n_lines + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] - other[col_e1, i_col])

        # Lines with surfaces on the left
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col
                col_e1 = row * n_lines + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] + other[col_e1, i_col])

        # Lines with surfaces on the right
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col + 1
                col_e1 = row * n_lines + col
                out = out.at[row_e, i_col].set(out[row_e, i_col] - other[col_e1, i_col])

    return out


def incidence_matrix_as_full(order: int, m: MatrixIncidence) -> MatrixFull:
    """Compute the full incidence matrix."""
    k = m.coeff
    if m.start_order == 0:
        if m.dual:
            v = full_e21(order).T
            k *= -1
        else:
            v = full_e10(order)
    elif m.start_order == 1:
        if m.dual:
            v = full_e10(order).T
            k *= -1
        else:
            v = full_e21(order)
    else:
        raise ValueError("Order of the start forms can be only 1 or 2.")

    return MatrixFull(k, v)


def apply_incidence_left(
    order: int, incidence: MatrixIncidence, m: AnyMatrix
) -> AnyMatrix:
    """Apply the incidence matrix to the input.

    Equivalent to:

    .. math::

        E A = B
    """
    k = m.coeff * incidence.coeff
    if type(m) is MatrixBase:
        return MatrixIncidence(k, incidence.start_order, incidence.dual)
    if type(m) is MatrixIncidence:
        tmp = incidence_matrix_as_full(order, m)
        v = tmp.data
    elif type(m) is MatrixFull:
        v = m.data
    else:
        raise TypeError(f"Unknown matrix type ({type(m)}).")

    if incidence.start_order == 0:
        if not incidence.dual:
            v = apply_e10(order, v)

        else:
            v = apply_e21_t(order, v)
            k *= -1.0

    elif incidence.start_order == 1:
        if not incidence.dual:
            v = apply_e21(order, v)

        else:
            v = apply_e10_t(order, v)
            k *= -1.0

    else:
        raise ValueError("Order of the start forms can be only 1 or 2")

    return MatrixFull(k, v)


def apply_e10_r(order: int, other: jax.Array) -> jax.Array:
    """Apply the right E10 matrix to the given input.

    Calling this function is equivalent to right multiplying by E10.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((other.shape[0], (order + 1) ** 2), np.float64)

    for i_row in range(other.shape[0]):
        # Nodes with lines on their right
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col
                col_e1 = n_lines * row + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] + other[i_row, col_e1])

        # Nodes with lines on their left
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col + 1
                col_e1 = n_lines * row + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] - other[i_row, col_e1])

        # Nodes with lines on their top
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = row * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] - other[i_row, col_e1])

        # Nodes with lines on their bottom
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = (row + 1) * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] + other[i_row, col_e1])

    return out


def apply_e10_rt(order: int, other: jax.Array) -> jax.Array:
    """Apply the right transposed E10 matrix to the given input.

    Calling this function is equivalent to right multiplying by E10 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((other.shape[0], 2 * order * (order + 1)), np.float64)

    for i_row in range(other.shape[0]):
        for row in range(n_nodes):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_nodes * row + col
                col_e2 = n_nodes * row + col + 1
                out = out.at[i_row, row_e].set(
                    other[i_row, col_e1] - other[i_row, col_e2]
                )

        for row in range(n_lines):
            for col in range(n_nodes):
                row_e = n_nodes * n_lines + row * n_nodes + col
                col_e1 = n_nodes * (row + 1) + col
                col_e2 = n_nodes * row + col
                out = out.at[i_row, row_e].set(
                    other[i_row, col_e1] - other[i_row, col_e2]
                )

    return out


def apply_e21_r(order: int, other: jax.Array) -> jax.Array:
    """Apply the right E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((other.shape[0], (order + 1) * order * 2), np.float64)

    for i_row in range(other.shape[0]):
        # Lines with surfaces on the top
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = row * n_lines + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] + other[i_row, col_e1])

        # Lines with surfaces on the bottom
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (row + 1) * n_lines + col
                col_e1 = row * n_lines + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] - other[i_row, col_e1])

        # Lines with surfaces on the left
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col
                col_e1 = row * n_lines + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] + other[i_row, col_e1])

        # Lines with surfaces on the right
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col + 1
                col_e1 = row * n_lines + col
                out = out.at[i_row, row_e].set(out[i_row, row_e] - other[i_row, col_e1])

    return out


def apply_e21_rt(order: int, other: jax.Array) -> jax.Array:
    """Apply the right transpose E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = jnp.zeros((other.shape[0], order**2), np.float64)

    for i_row in range(other.shape[0]):
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_lines * row + col  # +
                col_e2 = n_lines * (row + 1) + col  # -
                col_e3 = n_nodes * n_lines + n_nodes * row + col  # +
                col_e4 = n_nodes * n_lines + n_nodes * row + col + 1  # -
                out = out.at[i_row, row_e].set(
                    other[i_row, col_e1]
                    - other[i_row, col_e2]
                    + other[i_row, col_e3]
                    - other[i_row, col_e4]
                )

    return out


def apply_incidence_right(
    order: int, incidence: MatrixIncidence, m: AnyMatrix
) -> AnyMatrix:
    """Apply the incidence matrix to the input.

    Equivalent to:

    .. math::

        A E = B
    """
    k = m.coeff * incidence.coeff
    if type(m) is MatrixBase:
        return MatrixIncidence(k, incidence.start_order, incidence.dual)
    if type(m) is MatrixIncidence:
        tmp = incidence_matrix_as_full(order, m)
        v = tmp.data
    elif type(m) is MatrixFull:
        v = m.data
    else:
        raise TypeError(f"Unknown matrix type ({type(m)}).")

    if incidence.start_order == 0:
        if not incidence.dual:
            v = apply_e10_r(order, v)

        else:
            v = apply_e21_rt(order, v)
            k *= -1.0

    elif incidence.start_order == 1:
        if not incidence.dual:
            v = apply_e21_r(order, v)

        else:
            v = apply_e10_rt(order, v)
            k *= -1.0

    else:
        raise ValueError("Order of the start forms can be only 1 or 2")

    return MatrixFull(k, v)


def matmul(order: int, left: AnyMatrix, right: AnyMatrix) -> AnyMatrix:
    """Multiply two matrices."""
    if type(left) is MatrixIncidence:
        return apply_incidence_left(order, left, right)

    if type(left) is MatrixBase:
        return replace(right, coeff=right.coeff * left.coeff)

    if type(left) is MatrixFull:
        if type(right) is MatrixBase:
            return replace(left, coeff=left.coeff * right.coeff)

        if type(right) is MatrixFull:
            return MatrixFull(left.coeff * right.coeff, left.data @ right.data)

        if type(right) is MatrixIncidence:
            return apply_incidence_right(order, right, left)

        raise TypeError(f"Unknown right matrix type ({type(right)}).")

    raise TypeError(f"Unknown left matrix type ({type(left)}).")


def matadd(order: int, left: AnyMatrix, right: AnyMatrix) -> AnyMatrix:
    """Add two matricest together."""
    if type(left) is MatrixBase:
        # Add diagonal
        if type(right) is MatrixBase:
            return MatrixBase(left.coeff + right.coeff)

        if type(right) is MatrixIncidence:
            raise ValueError("Can not a diagonal matrix to an incidence matrix.")

        if type(right) is MatrixFull:
            idx = jnp.arange(right.data.shape[0])
            return MatrixFull(
                right.coeff,
                right.data.at[idx, idx].set(
                    right.data[idx, idx] + left.coeff / right.coeff
                ),
            )

        raise TypeError(f"Unknown right matrix type ({type(right)}).")

    if type(left) is MatrixIncidence:
        if type(right) is MatrixIncidence:
            if right.start_order != left.start_order or right.dual != left.dual:
                raise TypeError(
                    "Can not add two incidence matrices which are not of the same order."
                )

            return MatrixIncidence(left.coeff + right.coeff, left.start_order, left.dual)

        if type(right) is MatrixFull:
            tmp = incidence_matrix_as_full(order, left)
            mv = tmp.data
            mk = tmp.coeff

            return MatrixFull(1.0, mk * mv + right.coeff * right.data)

        if type(right) is MatrixBase:
            raise TypeError("Can not add a diagonal matrix and an incidence matrix.")

        raise TypeError(f"Unknown right matrix type ({type(right)}).")

    if type(left) is MatrixFull:
        if type(right) is MatrixBase:
            idx = jnp.arange(left.data.shape[0])
            return MatrixFull(
                left.coeff,
                left.data.at[idx, idx].set(
                    left.data[idx, idx] + right.coeff / left.coeff
                ),
            )

        if type(right) is MatrixIncidence:
            tmp = incidence_matrix_as_full(order, right)
            mv = tmp.data
            mk = tmp.coeff

            return MatrixFull(1.0, mk * mv + left.coeff * left.data)

        if type(right) is MatrixFull:
            return MatrixFull(1.0, left.coeff * left.data + right.coeff * right.data)

        raise TypeError(f"Unknown right matrix type ({type(right)}).")

    raise TypeError(f"Unknown left matrix type ({type(left)}).")


def evaluate_element_matrix(
    order: int,
    expr: Sequence[MatOp],
    mass_matrices: tuple[jax.Array, jax.Array, jax.Array],
    imass_matrices: tuple[jax.Array, jax.Array, jax.Array],
) -> jax.Array | float:
    """Evaluate the element matrix."""
    stack: list[AnyMatrix] = []
    val: AnyMatrix | None = None
    mat: AnyMatrix

    for op in expr:
        if type(op) is MassMat:
            # mat = # Get the mass matrix
            if not op.inv:
                mvals = mass_matrices
            else:
                mvals = imass_matrices
            mat = MatrixFull(1.0, mvals[op.order])

            if val is not None:
                val = matmul(order, mat, val)
            else:
                val = mat

        elif type(op) is Incidence:
            # mat = f"E({op.begin + 1}, {op.begin})" + ("*" if op.dual else "")
            mat = MatrixIncidence(1.0, op.begin, bool(op.dual))
            if val is not None:
                val = apply_incidence_left(order, mat, val)
            else:
                val = mat

        elif type(op) is Push:
            if val is None:
                raise ValueError("Invalid Push operation.")
            stack.append(val)
            val = None

        elif type(op) is Scale:
            if val is None:
                val = MatrixBase(op.k)
            else:
                val = replace(val, coeff=op.k * val.coeff)

        elif type(op) is MatMul:
            mat = stack.pop()
            if val is None:
                raise ValueError("Invalid MatMul operation.")

            val = matmul(order, val, mat)

        elif type(op) is Transpose:
            if val is None:
                raise ValueError("Invalid Transpose operation.")

            if type(val) is MatrixBase:
                # No Op
                pass
            elif type(val) is MatrixFull:
                # Do the actual transpose
                val = replace(val, data=val.data.T)
            elif type(val) is MatrixIncidence:
                # Make it a dual matrix
                val = MatrixIncidence(-1 * val.coeff, 1 - val.start_order, not val.dual)
            else:
                raise TypeError(f"Unknown matrix type ({type(val)})")

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

            for _ in range(n):
                mat = stack.pop()
                val = matadd(order, val, mat)

        elif type(op) is Identity:
            if val is None:
                val = MatrixBase(1.0)

        else:
            raise TypeError(f"Unknown operation {op}.")

    if len(stack):
        raise ValueError(f"{len(stack)} matrices still on the stack.")
    # TODO: return a full matrix
    if val is None:
        raise ValueError("No current state.")
    if len(stack):
        raise ValueError("Values still on the stack.")

    if type(val) is MatrixBase:
        return val.coeff

    if type(val) is MatrixIncidence:
        val = incidence_matrix_as_full(order, val)

    if type(val) is MatrixFull:
        return val.coeff * val.data

    raise TypeError(f"Unknown matrix type computed ({type(val)}).")


def compute_element_matrices(
    expr_mat: Sequence[Sequence[None | Sequence[MatOp]]],
    order: int,
    form_sizes: Sequence[int],
    bl: jax.Array,
    br: jax.Array,
    tr: jax.Array,
    tl: jax.Array,
    caches: dict[int, BasisCache],
) -> jax.Array:
    """Comnpute element matrices."""
    # Compute mass matrices
    cache = caches[order]
    (j00, j01), (j10, j11), det = compute_jacobian(
        bl, br, tr, tl, cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
    )
    mat_mass_0 = jnp.sum(cache.mass_node_precomp * det, axis=(-2, -1))
    precomp_edge = cache.mass_edge_precomp
    nb = precomp_edge.shape[0] // 2

    khh = j11**2 + j10**2
    kvv = j01**2 + j00**2
    kvh = j01 * j11 + j00 * j10
    khh = khh / det
    kvv = kvv / det
    kvh = kvh / det
    mat_mass_1_00 = jnp.sum(
        precomp_edge[0 * nb : 1 * nb, 0 * nb : 1 * nb, ...] * khh, axis=(-2, -1)
    )
    mat_mass_1_01 = jnp.sum(
        precomp_edge[1 * nb : 2 * nb, 0 * nb : 1 * nb, ...] * kvh, axis=(-2, -1)
    )
    mat_mass_1_11 = jnp.sum(
        precomp_edge[1 * nb : 2 * nb, 1 * nb : 2 * nb, ...] * kvv, axis=(-2, -1)
    )
    mat_mass_1 = jnp.block(
        [[mat_mass_1_00, mat_mass_1_01], [mat_mass_1_01.T, mat_mass_1_11]]
    )
    del mat_mass_1_01, mat_mass_1_00, mat_mass_1_11
    mat_mass_2 = jnp.sum(cache.mass_surf_precomp / det, axis=(-2, -1))

    n_form = len(form_sizes)
    blocks: list[list[jax.Array]] = list()
    for iform in range(n_form):
        row_blocks: list[jax.Array] = list()

        for jform in range(n_form):
            expr = expr_mat[iform][jform]

            if expr is not None:
                mat = evaluate_element_matrix(
                    order,
                    expr,
                    (mat_mass_0, mat_mass_1, mat_mass_2),
                    (
                        jnp.linalg.inv(mat_mass_0),
                        jnp.linalg.inv(mat_mass_1),
                        jnp.linalg.inv(mat_mass_2),
                    ),
                )
                if isinstance(mat, (float, int)):
                    row_blocks.append(
                        jnp.eye(form_sizes[iform], form_sizes[jform], dtype=jnp.float64)
                        * mat
                    )
                else:
                    row_blocks.append(mat)
            else:
                row_blocks.append(
                    jnp.zeros((form_sizes[iform], form_sizes[jform]), jnp.float64)
                )
        blocks.append(row_blocks)

    return jnp.block(blocks)


# if __name__ == "__main__":
#     jax.config.update("jax_enable_x64", True)

#     np.random.seed(0)

#     instructions: tuple[tuple[None | tuple[MatOp, ...], ...], ...] = (
#         ((Identity(),), (Incidence(1, 0), MassMat(2, False), Transpose())),
#         ((Incidence(1, 0), MassMat(2, False)), None),
#     )
#     order = 1

#     caches = {o: BasisCache(o, 2 * o) for o in range(1, 5)}

#     res = compute_element_matrices(
#         instructions,
#         order,
#         [order * (order + 1) * 2, order**2],
#         np.array((0, 0)),
#         np.array((1, 0)),
#         np.array((1, 1)),
#         np.array((0, 1)),
#         caches,
#     )

#     from functools import partial

#     partial_func = partial(compute_element_matrices, caches=caches)
#     compiled_func = partial(
#         jax.jit(partial_func, static_argnames=("expr_mat", "order", "form_sizes")),
#         expr_mat=instructions,
#     )

#     NE = 3
#     blvs = np.full((NE, 2), (0, 0))
#     brvs = np.full((NE, 2), (1, 0))
#     trvs = np.full((NE, 2), (1, 1))
#     tlvs = np.full((NE, 2), (0, 1))
#     orders = np.array((1, 2, 3))

#     for i in range(NE):
#         v = compiled_func(
#             order=orders[i],
#             form_sizes=(orders[i] * (order + 1) * 2, order**2),
#             bl=blvs[i],
#             br=brvs[i],
#             tr=trvs[i],
#             tl=tlvs[i],
#         )
#         print(v)
