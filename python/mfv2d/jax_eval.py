"""JAX-friendly evaluation of element matrices."""

from dataclasses import dataclass, replace
from enum import IntEnum

# from time import perf_counter
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from mfv2d.eval import (
    Identity,
    Incidence,
    MassMat,
    MatMul,
    MatOp,
    Push,
    Scale,
    Sum,
)
from mfv2d.mimetic2d import BasisCache


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


# TODO: use vectorized setting and getting
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


# TODO: use vectorized setting and getting
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
    mat = jnp.reshape(other, (order + 1, order + 1, other.shape[-1]))

    out_h = mat[:, :-1, :] - mat[:, +1:, :]
    out_v = mat[+1:, :, :] - mat[:-1, :, :]
    result = jnp.concatenate(
        (
            jnp.reshape(out_h, (order * (order + 1), other.shape[-1])),
            jnp.reshape(out_v, (order * (order + 1), other.shape[-1])),
        ),
        axis=0,
    )

    return result


def apply_e10_t(order: int, other: jax.Array) -> jax.Array:
    """Apply the E10 transpose matrix to the given input.

    Calling this function is equivalent to left multiplying by E10 transposed.
    """
    assert other.ndim == 2
    mat_h = jnp.reshape(
        other[: order * (order + 1), :], (order + 1, order, other.shape[-1])
    )
    mat_v = jnp.reshape(
        other[order * (order + 1) :, :], (order, order + 1, other.shape[-1])
    )
    result = jnp.zeros(((order + 1), (order + 1), other.shape[-1]), np.float64)

    result = result.at[+1:, :, :].set(result[+1:, :, :] + mat_v)
    result = result.at[:-1, :, :].set(result[:-1, :, :] - mat_v)
    result = result.at[:, :-1, :].set(result[:, :-1, :] + mat_h)
    result = result.at[:, +1:, :].set(result[:, +1:, :] - mat_h)

    result = jnp.reshape(result, ((order + 1) ** 2, other.shape[-1]))

    return result


def apply_e21(order: int, other: jax.Array) -> jax.Array:
    """Apply the E21 matrix to the given input.

    Calling this function is equivalent to left multiplying by E21.
    """
    assert other.ndim == 2
    mat_h = jnp.reshape(
        other[: order * (order + 1), :], (order + 1, order, other.shape[-1])
    )
    mat_v = jnp.reshape(
        other[order * (order + 1) :, :], (order, order + 1, other.shape[-1])
    )
    result = jnp.reshape(
        (mat_h[:-1, :, :] - mat_h[+1:, :, :]) - (mat_v[:, +1:, :] - mat_v[:, :-1, :]),
        (order**2, other.shape[-1]),
    )
    return result


def apply_e21_t(order: int, other: jax.Array) -> jax.Array:
    """Apply the E21 transposed matrix to the given input.

    Calling this function is equivalent to left multiplying by E21 transposed.
    """
    assert other.ndim == 2
    mat = jnp.reshape(other, (order, order, other.shape[-1]))
    out_h = jnp.zeros((order + 1, order, other.shape[-1]))
    out_v = jnp.zeros((order, order + 1, other.shape[-1]))

    out_h = out_h.at[:-1, :, :].set(out_h[:-1, :, :] + mat)
    out_h = out_h.at[+1:, :, :].set(out_h[+1:, :, :] - mat)
    out_v = out_v.at[:, :-1, :].set(out_v[:, :-1, :] + mat)
    out_v = out_v.at[:, +1:, :].set(out_v[:, +1:, :] - mat)

    result = jnp.concatenate(
        (
            jnp.reshape(out_h, (order * (order + 1), other.shape[-1])),
            jnp.reshape(out_v, (order * (order + 1), other.shape[-1])),
        ),
        axis=0,
    )

    return result


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
    mat_h = jnp.reshape(
        other[:, : order * (order + 1)], (other.shape[0], order + 1, order)
    )
    mat_v = jnp.reshape(
        other[:, order * (order + 1) :], (other.shape[0], order, order + 1)
    )
    result = jnp.zeros((other.shape[0], (order + 1), (order + 1)), np.float64)

    result = result.at[:, +1:, :].set(result[:, +1:, :] + mat_v)
    result = result.at[:, :-1, :].set(result[:, :-1, :] - mat_v)
    result = result.at[:, :, :-1].set(result[:, :, :-1] + mat_h)
    result = result.at[:, :, +1:].set(result[:, :, +1:] - mat_h)

    result = jnp.reshape(result, (other.shape[0], (order + 1) ** 2))
    return result


def apply_e10_rt(order: int, other: jax.Array) -> jax.Array:
    """Apply the right transposed E10 matrix to the given input.

    Calling this function is equivalent to right multiplying by E10 transposed.
    """
    assert other.ndim == 2
    mat = jnp.reshape(other, (other.shape[0], order + 1, order + 1))

    out_h = mat[:, :, :-1] - mat[:, :, +1:]
    out_v = mat[:, +1:, :] - mat[:, :-1, :]
    result = jnp.concatenate(
        (
            jnp.reshape(out_h, (other.shape[0], order * (order + 1))),
            jnp.reshape(out_v, (other.shape[0], order * (order + 1))),
        ),
        axis=1,
    )
    return result


def apply_e21_r(order: int, other: jax.Array) -> jax.Array:
    """Apply the right E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21.
    """
    assert other.ndim == 2
    mat = jnp.reshape(other, (other.shape[0], order, order))
    out_h = jnp.zeros((other.shape[0], order + 1, order))
    out_v = jnp.zeros((other.shape[0], order, order + 1))

    out_h = out_h.at[:, :-1, :].set(out_h[:, :-1, :] + mat)
    out_h = out_h.at[:, +1:, :].set(out_h[:, +1:, :] - mat)
    out_v = out_v.at[:, :, :-1].set(out_v[:, :, :-1] + mat)
    out_v = out_v.at[:, :, +1:].set(out_v[:, :, +1:] - mat)

    result = jnp.concatenate(
        (
            jnp.reshape(out_h, (other.shape[0], order * (order + 1))),
            jnp.reshape(out_v, (other.shape[0], order * (order + 1))),
        ),
        axis=1,
    )
    return result


def apply_e21_rt(order: int, other: jax.Array) -> jax.Array:
    """Apply the right transpose E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21 transposed.
    """
    assert other.ndim == 2
    mat_h = jnp.reshape(
        other[:, : order * (order + 1)], (other.shape[0], order + 1, order)
    )
    mat_v = jnp.reshape(
        other[:, order * (order + 1) :], (other.shape[0], order, order + 1)
    )
    result = jnp.reshape(
        (mat_h[:, :-1, :] - mat_h[:, +1:, :]) - (mat_v[:, :, +1:] - mat_v[:, :, :-1]),
        (other.shape[0], order**2),
    )

    return result


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

            # elif type(op) is Transpose:
            #     if val is None:
            #         raise ValueError("Invalid Transpose operation.")

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
    bl: jax.Array,
    br: jax.Array,
    tr: jax.Array,
    tl: jax.Array,
    order: int,
    form_orders: Sequence[int],
    expr_mat: Sequence[Sequence[None | Sequence[MatOp]]],
) -> jax.Array:
    """Comnpute element matrices using JAX.

    By using :func:`functools.partial`, it is possible to fix all arguments except the
    first four. This makes the function able to be re-used for different elements of the
    same order. This partial function can then also be used with `jax.jit` and `jax.vmap`
    in order to vectorize it and JIT compile it.

    Parameters
    ----------
    bl : (float, float)
        Coordinates of the bottom left corner.
    br : (float, float)
        Coordinates of the bottom right corner.
    tl : (float, float)
        Coordinates of the top left corner.
    tr : (float, float)
        Coordinates of the top right corner.
    order : int
        Order of the element.
    form_orders : Sequence[int]
        Orders of the differential forms.
    expr_mat : (M, M) array of None or Sequence of MatOp
        Expression which needs to be evaluated for each block of the element matrix.

    Returns
    -------
    array
        Element matrix for this element.
    """
    # print(f"Evaluating function for order {order}")
    # t0 = perf_counter()
    # Compute mass matrices
    c = BasisCache(order, 2 * order)
    int_nodes_1d = c.int_nodes_1d
    mass_node_precomp = c.mass_node_precomp
    mass_edge_precomp = c.mass_edge_precomp
    mass_surf_precomp = c.mass_surf_precomp
    del c
    (j00, j01), (j10, j11), det = compute_jacobian(
        bl, br, tr, tl, int_nodes_1d[None, :], int_nodes_1d[:, None]
    )
    mat_mass_0 = jnp.sum(mass_node_precomp * det, axis=(-2, -1))
    nb = mass_edge_precomp.shape[0] // 2

    khh = j11**2 + j10**2
    kvv = j01**2 + j00**2
    kvh = j01 * j11 + j00 * j10
    khh = khh / det
    kvv = kvv / det
    kvh = kvh / det
    mat_mass_1_00 = jnp.sum(
        mass_edge_precomp[0 * nb : 1 * nb, 0 * nb : 1 * nb, ...] * khh, axis=(-2, -1)
    )
    mat_mass_1_01 = jnp.sum(
        mass_edge_precomp[1 * nb : 2 * nb, 0 * nb : 1 * nb, ...] * kvh, axis=(-2, -1)
    )
    mat_mass_1_11 = jnp.sum(
        mass_edge_precomp[1 * nb : 2 * nb, 1 * nb : 2 * nb, ...] * kvv, axis=(-2, -1)
    )
    mat_mass_1 = jnp.block(
        [[mat_mass_1_00, mat_mass_1_01], [mat_mass_1_01.T, mat_mass_1_11]]
    )
    del mat_mass_1_01, mat_mass_1_00, mat_mass_1_11
    mat_mass_2 = jnp.sum(mass_surf_precomp / det, axis=(-2, -1))

    form_sz: list[int] = list()
    for f_order in form_orders:
        if f_order == 0:
            form_sz.append((order + 1) ** 2)
        elif f_order == 1:
            form_sz.append((order + 1) * order * 2)
        elif f_order == 2:
            form_sz.append(order**2)
        else:
            raise ValueError(f"Can not have a form of order {f_order}.")

    form_sizes = np.array(form_sz)
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
    # t1 = perf_counter()
    # print(f"Compiling order {order} function took {t1 - t0:g} seconds.")
    return jnp.block(blocks)


class ElementMatrixResults:
    """Type used to manage element matrices computed using JAX."""

    _group_arrays: tuple[jax.Array, ...]
    _group_indices: tuple[npt.NDArray[np.intp], ...]
    _order_map: dict[int, int]
    _orders: npt.NDArray[np.uint32]

    def __init__(
        self,
        group_arrays: Sequence[jax.Array],
        group_indices: Sequence[npt.NDArray[np.intp]],
        order_map: dict[int, int],
        orders: npt.ArrayLike,
    ) -> None:
        self._group_arrays = tuple(group_arrays)
        self._group_indices = tuple(group_indices)
        self._order_map = dict(order_map)
        self._orders = np.array(orders, np.uint32)

    def element_matrix(self, element: int, /) -> jax.Array:
        """Get the element matrix for the specified element."""
        element_order = int(self._orders[element])
        group_index = self._order_map[element_order]
        group_array = self._group_arrays[group_index]
        group_offset = np.flatnonzero(self._group_indices[group_index] == element)
        assert len(group_offset) == 1
        return jnp.reshape(
            group_array[group_offset[0], :, :],
            (group_array.shape[1], group_array.shape[2]),
        )


compiled_element_function = jax.jit(compute_element_matrices, static_argnums=(4, 5, 6))
mapped_element_function = jax.vmap(
    compiled_element_function, in_axes=(0, 0, 0, 0, None, None, None)
)


def compute_element_matrices_3(
    form_orders: Sequence[int],
    expressions: Sequence[Sequence[Sequence[MatOp] | None]],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
) -> ElementMatrixResults:
    """Evaluate element matrices using JAX.

    Parameters
    ----------
    form_orders : Sequence of int
        Orders of differential forms in the element.

    expressions: (N, N) array of None or Sequence of MatOp
        Expressions or None corresponding to the element matrix blocks.

    pos_bl : (N, 2) array
        Positions of the element bottom left corners.

    pos_br : (N, 2) array
        Positions of the element bottom right corners.

    pos_tr : (N, 2) array
        Positions of the element top right corners.

    pos_tl : (N, 2) array
        Positions of the element top left corners.

    element_orders : (N,) array
        Orders of the elements.

    Returns
    -------
    ElementMatrixResults

    """
    unique_orders = np.unique(element_orders)
    # compiled_func = jax.jit(
    #     compute_element_matrices, static_argnums=(4, 5, 6)
    # )
    # fn = jax.vmap(
    #     compiled_func,
    #     in_axes=(0, 0, 0, 0, None, None, None),
    #     out_axes=0,
    # )
    fn = mapped_element_function

    base_indices: dict[int, int] = dict()
    contents: list[npt.NDArray[np.intp]] = list()
    results: list[jax.Array] = list()
    for ifun, order in enumerate(unique_orders):
        mask = np.flatnonzero(element_orders == order)
        contents.append(mask)
        base_indices[int(order)] = int(ifun)
        results.append(
            fn(
                jax.device_put(pos_bl[mask, :]),
                jax.device_put(pos_br[mask, :]),
                jax.device_put(pos_tr[mask, :]),
                jax.device_put(pos_tl[mask, :]),
                order,
                tuple(form_orders),
                expressions,
            )
        )

    return ElementMatrixResults(results, contents, base_indices, element_orders)


# if __name__ == "__main__":
#     jax.config.update("jax_enable_x64", True)

#     np.random.seed(0)

#     instructions: tuple[tuple[None | tuple[MatOp, ...], ...], ...] = (
#         ((MassMat(1, False),), (Incidence(1, 0), MassMat(2, False), Transpose())),
#         ((Incidence(1, 0), MassMat(2, False)), None),
#     )

#     from collections.abc import Callable
#     from functools import partial

#     NE = 10000
#     blvs = np.full((NE, 2), np.array((0, 0)))
#     brvs = np.full((NE, 2), np.array((1, 0)))
#     trvs = np.full((NE, 2), np.array((1, 1)))
#     tlvs = np.full((NE, 2), np.array((0, 1)))
#     np.random.seed(0)
#     orders = np.random.randint(1, 10, NE)
#     unique_orders = np.unique(orders)
#     funcs: dict[int, Callable] = dict()
#     for order in unique_orders:
#         c = BasisCache(order, 2 * order)

#         partial_func = partial(
#             compute_element_matrices,
#             expr_mat=instructions,
#             form_orders=(1, 2),
#             order=order,
#             int_nodes_1d=c.int_nodes_1d,
#             mass_node_precomp=c.mass_node_precomp,
#             mass_edge_precomp=c.mass_edge_precomp,
#             mass_surf_precomp=c.mass_surf_precomp,
#         )

#         compiled_func = jax.jit(partial_func)

#         del c
#         fn = jax.vmap(
#             compiled_func,
#             in_axes=(0, 0, 0, 0),
#             out_axes=0,
#         )

#         funcs[order] = fn

#     res: list[jax.Array] = []

#     # Run the operations to be profiled

#     t0 = perf_counter()
#     for order in unique_orders:
#         pos = np.where(orders == order)[0]
#         t0 = perf_counter()
#         v = funcs[order](
#             jax.device_put(blvs[pos, :]),
#             jax.device_put(brvs[pos, :]),
#             jax.device_put(trvs[pos, :]),
#             jax.device_put(tlvs[pos, :]),
#         )
#         t1 = perf_counter()

#         print(f"Did {len(pos)} elements of order {order} in {t1 - t0:g} seconds.")
#         res.append(v)
#     for v in res:
#         v.block_until_ready()
#     t1 = perf_counter()
#     print(f"{NE} elements took {t1 - t0:g} seconds")
