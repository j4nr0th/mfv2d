"""Code related to solving the system of equations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy import linalg as la
from scipy import sparse as sp

from mfv2d._mfv2d import DenseVector, MatrixCRS, SparseVector, TraceVector
from mfv2d._mfv2d import LinearSystem as CLinearSystem
from mfv2d.mimetic2d import Constraint
from mfv2d.system import ElementFormSpecification


class LinearSystem(CLinearSystem):
    """Class used to represent a linear system with element equations and constraints.

    Parameters
    ----------
    n_elem : int
        Number of elements in the system.

    form_spec : ElementFormSpecification
        Specification of forms in the element system.

    orders : array
        Array of element orders.

    element_matrices : Sequence of arrays
        Sequence of element matrices.

    constraints : Sequence of Constraint
        Sequences of constraints the element systems are subject to.
    """

    def __new__(
        cls,
        n_elem: int,
        form_spec: ElementFormSpecification,
        orders: npt.NDArray[np.uint32],
        element_matrices: Sequence[npt.NDArray[np.float64]],
        constraints: Sequence[Constraint],
    ) -> Self:
        """Create a new LinearSystem."""
        assert n_elem == len(element_matrices)
        assert orders.shape == (n_elem, 2)

        for ie in range(n_elem):
            size = form_spec.total_size(*orders[ie])
            assert element_matrices[ie].shape == (size, size)

        element_constraint_counts = np.zeros(n_elem, np.uintc)
        for con in constraints:
            for ec in con.element_constraints:
                element_constraint_counts[ec.i_e] += 1

        system_blocks = tuple(
            (
                element_matrices[ie],
                MatrixCRS(len(constraints), element_matrices[ie].shape[1]),
            )
            for ie in range(n_elem)
        )

        c_vals: list[float] = list()
        c_elems: list[npt.NDArray[np.uint32]] = list()
        c_lens: list[int] = [0]
        for ic, con in enumerate(constraints):
            present_elements: list[int] = list()
            for ec in con.element_constraints:
                present_elements.append(ec.i_e)
                system = system_blocks[ec.i_e]
                system[1].build_row(
                    ic,
                    SparseVector.from_entries(system[1].shape[1], ec.dofs, ec.coeffs),
                )

            c_elems.append(
                np.array([ec.i_e for ec in con.element_constraints], np.uint32)
            )
            c_vals.append(con.rhs)
            c_lens.append(c_elems[-1].size)

            for ie in (i for i in range(n_elem) if i not in present_elements):
                system_blocks[ie][1].build_row(ic)

        return super().__new__(
            cls,
            system_blocks,
            np.cumsum(c_lens, dtype=np.uint32),
            np.concatenate(c_elems, dtype=np.uint32),
        )

    def apply_full_trace_system(
        self, x: TraceVector, /, out: TraceVector, tmp1: DenseVector, tmp2: DenseVector
    ) -> None:
        """Apply the Schur's trace system.

        Parameters
        ----------
        TraceVector
            Trace vector to which this is applied to.

        out : TraceVector
            Trace vector, which receives output. Can be the same as input.

        tmp1 : DenseVector
            Dense vector used to store intermediate result. Must be unique.

        tmp2 : DenseVector
            Dense vector used to store intermediate result. Must be unique.
        """
        if tmp1 is tmp2:
            raise ValueError("Temporary dense vectors must not be the same.")

        self.apply_trace_transpose(x, tmp1)
        self.apply_diagonal_inverse(tmp1, tmp2)
        self.apply_trace(tmp2, out)

    def combined_system_matrix(self) -> sp.csr_array:
        """Combine the system matrix into a scipy CSR array."""
        diagonal_part = sp.block_diag(self.get_dense_blocks())
        lagrange_block = sp.block_array(
            [
                [
                    sp.csc_array(
                        (
                            constraint_matrix.values,
                            (
                                constraint_matrix.row_indices,
                                constraint_matrix.column_indices,
                            ),
                        ),
                        shape=constraint_matrix.shape,
                    )
                    for constraint_matrix in self.get_constraint_blocks()
                ]
            ]
        )
        return cast(
            sp.csr_array,
            sp.block_array(
                [
                    [diagonal_part, lagrange_block.T],
                    [lagrange_block, None],
                ],
                format="csr",
            ),
        )


@dataclass(frozen=True)
class ConvergenceSettings:
    """Settings used to specify convergence of an iterative solver."""

    maximum_iterations: int = 100
    """Maximum number of iterations to improve the solution."""

    absolute_tolerance: float = 1e-6
    """When the largest update in the solution drops bellow this value,
    consider it converged."""

    relative_tolerance: float = 1e-5
    """When the largest update in the solution drops bellow the largest
    value of solution degrees of freedom scaled by this value, consider
    it converged."""


_Mat = TypeVar("_Mat")
_Vec = TypeVar("_Vec")


def gmres_general(
    mat: _Mat,
    rhs: _Vec,
    initial_guess: _Vec,
    convergence: ConvergenceSettings,
    system_application_function: Callable[[_Mat, _Vec, _Vec], None],
    vec_dot_function: Callable[[_Vec, _Vec], float],
    vec_add_to_function: Callable[[_Vec, _Vec, _Vec, float], None],
    vec_sub_from_scaled_function: Callable[[_Vec, _Vec, _Vec, float], None],
    vec_scale_by_function: Callable[[_Vec, float, _Vec], None],
    vec_copy_function: Callable[[_Vec], _Vec],
) -> tuple[_Vec, float, int]:
    """General implementation of GMRES to use for any data type with operators.

    Returns
    -------
    _Vec
        Computed solution.

    float
        Estimated residual.

    int
        Iterations done.
    """
    m = convergence.maximum_iterations
    g = np.zeros(m, np.float64)
    h = np.zeros(m, np.float64)
    sk = np.zeros(m, np.float64)
    ck = np.zeros(m, np.float64)
    r = np.zeros((m, m), np.float64)
    k = 0

    p_vecs: list[_Vec] = list()

    # Find stopping criterion
    rhs_mag = np.sqrt(vec_dot_function(rhs, rhs))
    if rhs_mag * convergence.relative_tolerance > convergence.absolute_tolerance:
        tol = convergence.absolute_tolerance
    else:
        tol = rhs_mag * convergence.relative_tolerance

    res = vec_copy_function(rhs)
    system_application_function(mat, initial_guess, res)
    vec_sub_from_scaled_function(rhs, res, res, 1.0)
    # First residual
    p = res
    # Get magnitude of the residual
    r_mag = np.sqrt(vec_dot_function(p, p))
    # Normalize the vector
    vec_scale_by_function(p, 1 / r_mag, p)
    # Add it to the current collection of basis and LSQR state
    p_vecs.append(p)
    g[0] = r_mag

    for k in range(1, m):
        # Make a new basis vector
        p = vec_copy_function(p)
        system_application_function(mat, p, p)
        # Make it orthogonal to other basis
        for li in range(k):
            p_old = p_vecs[li]
            pp_dp = vec_dot_function(p, p_old)
            h[li] = pp_dp
            vec_sub_from_scaled_function(p, p_old, p, pp_dp)

        # Get the magnitude and normalize it
        p_mag2 = vec_dot_function(p, p)  # Surprise tool for later
        p_mag = np.sqrt(p_mag2)
        vec_scale_by_function(p, 1 / p_mag, p)
        p_vecs.append(p)

        # Apply previous Givens rotations to the new column
        for i in range(k - 1):
            tmp = ck[i] * h[i] + sk[i] * h[i + 1]
            h[i + 1] = -sk[i] * h[i] + ck[i] * h[i + 1]
            h[i] = tmp

        # Find new Givens rotation
        rho = np.sqrt(p_mag2 + h[k - 1] * h[k - 1])
        c_new = h[k - 1] / rho
        s_new = p_mag / rho
        ck[k - 1] = c_new
        sk[k - 1] = s_new
        h[k - 1] = c_new * h[k - 1] + s_new * p_mag
        r[:k, k - 1] = h[:k]
        g[k] = -s_new * g[k - 1]
        g[k - 1] = c_new * g[k - 1]

        r_mag = np.abs(g[k])
        if r_mag < tol:
            # k += 1
            break

    # Iterations are done, time to solve the LSQR problem
    alpha = la.solve_triangular(r[:k, :k], g[:k])
    sol = vec_copy_function(initial_guess)
    for i in range(k):
        vec_add_to_function(sol, p_vecs[i], sol, alpha[i])
    return sol, r_mag, k


def pcg_general(
    mat: _Mat,
    rhs: _Vec,
    initial_guess: _Vec,
    convergence: ConvergenceSettings,
    system_application_function: Callable[[_Mat, _Vec, _Vec], None],
    precondition_function: Callable[[_Mat, _Vec, _Vec], None],
    vec_dot_function: Callable[[_Vec, _Vec], float],
    vec_add_to_scaled_function: Callable[[_Vec, _Vec, float, _Vec], None],
    vec_sub_from_scaled_function: Callable[[_Vec, _Vec, float, _Vec], None],
    vec_copy_function: Callable[[_Vec], _Vec],
    degen_limit: float = 1e-12,
) -> tuple[_Vec, float, int]:
    """General implementation of GMRES to use for any data type with operators.

    Returns
    -------
    _Vec
        Computed solution.

    float
        Estimated residual.

    int
        Iterations done.
    """
    # Find stopping criterion
    x = vec_copy_function(initial_guess)
    res = vec_copy_function(initial_guess)
    system_application_function(mat, x, res)
    vec_sub_from_scaled_function(rhs, res, 1.0, res)
    p = vec_copy_function(res)
    precondition_function(mat, res, p)
    z = vec_copy_function(p)
    ap = vec_copy_function(rhs)

    res_mag2 = vec_dot_function(rhs, rhs)
    if (
        np.sqrt(res_mag2) * convergence.relative_tolerance
        > convergence.absolute_tolerance
    ):
        tol = convergence.absolute_tolerance
    else:
        tol = np.sqrt(res_mag2) * convergence.relative_tolerance

    rz_dp = vec_dot_function(res, z)

    iter_cnt = 0
    for iter_cnt in range(convergence.maximum_iterations):
        system_application_function(mat, p, ap)
        apa = vec_dot_function(ap, p)
        if (np.log(abs(apa)) - np.log(res_mag2)) < np.log(degen_limit):
            raise RuntimeError("System degenerated (matrix was probably not SPD).")
        alpha = rz_dp / apa
        vec_add_to_scaled_function(x, p, alpha, x)
        vec_sub_from_scaled_function(res, ap, alpha, res)
        res_mag2 = vec_dot_function(res, res)
        if res_mag2 < tol**2:
            break

        precondition_function(mat, res, z)
        new_rz_dp = vec_dot_function(res, z)

        beta = new_rz_dp / rz_dp
        rz_dp = new_rz_dp
        vec_add_to_scaled_function(z, p, beta, p)

    return x, np.sqrt(res_mag2), iter_cnt


def cg_general(
    mat: _Mat,
    rhs: _Vec,
    initial_guess: _Vec,
    convergence: ConvergenceSettings,
    system_application_function: Callable[[_Mat, _Vec], None],
    vec_dot_function: Callable[[_Vec, _Vec], float],
    vec_add_to_scaled_function: Callable[[_Vec, _Vec, float], None],
    vec_sub_from_scaled_function: Callable[[_Vec, _Vec, float], None],
    vec_copy_function: Callable[[_Vec], _Vec],
    vec_set_function: Callable[[_Vec, _Vec], None],
) -> tuple[_Vec, float, int]:
    """General implementation of GMRES to use for any data type with operators.

    Parameters
    ----------
    m : int
        Number of basis to use for the solution.

    mat : _Mat
        System matrix state.

    rhs : _Vec
        Right side of the system.

    convergence : ConvergenceSettings
        Settings to use for convergence.

    system_application_function : (_Mat, _Vec) -> _Vec
        Function that applies the system to the input vector.

    vec_dot_function : (_Vec, _Vec) -> float
        Function that computes the dot product of system vectors.

    vec_add_function : (_Vec, _Vec) -> _Vec
        Function that adds two vectors together.

    vec_sub_function : (_Vec, _Vec) -> _Vec
        Function which subtracts two vectors.

    vec_scale_function : (float, _Vec) -> _Vec
        Function which scales a vector by a constant

    Returns
    -------
    _Vec
        Computed solution.

    float
        Estimated residual.

    int
        Iterations done.
    """
    # Find stopping criterion
    res_mag2 = vec_dot_function(rhs, rhs)
    if (
        np.sqrt(res_mag2) * convergence.relative_tolerance
        > convergence.absolute_tolerance
    ):
        tol = convergence.absolute_tolerance
    else:
        tol = np.sqrt(res_mag2) * convergence.relative_tolerance

    ap = vec_copy_function(rhs)
    p = vec_copy_function(rhs)
    res = vec_copy_function(rhs)
    x = vec_copy_function(initial_guess)

    iter_cnt = 0
    for iter_cnt in range(convergence.maximum_iterations):
        system_application_function(mat, ap)
        apa = vec_dot_function(ap, p)
        alpha = res_mag2 / apa
        vec_add_to_scaled_function(x, p, alpha)
        vec_sub_from_scaled_function(res, ap, alpha)
        new_res_mag2 = vec_dot_function(res, res)
        if new_res_mag2 < tol**2:
            res_mag2 = new_res_mag2
            break
        beta = new_res_mag2 / res_mag2
        res_mag2 = new_res_mag2
        vec_set_function(ap, res)
        vec_add_to_scaled_function(ap, p, beta)
        vec_set_function(p, ap)

    return x, np.sqrt(res_mag2), iter_cnt


def solve_schur_iterative(
    system: LinearSystem,
    rhs: DenseVector,
    constraints: TraceVector,
    convergence: ConvergenceSettings,
) -> tuple[DenseVector, TraceVector, float, int]:
    """Solve the system using Schur's compliment but iteratively."""
    # Compute rhs forcing for the Lagrange multipliers

    ### A^{-1} y
    inv_a_y = DenseVector(system)
    system.apply_diagonal_inverse(rhs, inv_a_y)
    ### N A^{-1} y - phi
    trace_rhs = TraceVector(system)
    system.apply_trace(inv_a_y, trace_rhs)
    TraceVector.subtract(trace_rhs, constraints, trace_rhs, 1.0)

    tmp_d1 = DenseVector(system)
    tmp_d2 = DenseVector(system)

    def wrapped_apply_system(system: LinearSystem, v: TraceVector, /) -> None:
        """Apply the trace system in a wrapped way."""
        system.apply_full_trace_system(v, v, tmp_d1, tmp_d2)

    # Iteratively solve the system for trace lambda
    # trace_lhs, tr, itr_cnt = gmres_general(
    #     system,
    #     trace_rhs,
    #     convergence,
    #     wrapped_apply_system,
    #     TraceVector.dot,
    #     TraceVector.add_to,
    #     TraceVector.subtract_from_scaled,
    #     TraceVector.scale_by,
    #     TraceVector.copy,
    # )

    def _wrap_add(v1: TraceVector, v2: TraceVector, k: float) -> None:
        TraceVector.add(v1, v2, v1, k)

    def _wrap_sub(v1: TraceVector, v2: TraceVector, k: float) -> None:
        TraceVector.subtract(v1, v2, v1, k)

    trace_lhs, tr, itr_cnt = cg_general(
        system,
        trace_rhs,
        TraceVector(system),
        convergence,
        wrapped_apply_system,
        TraceVector.dot,
        _wrap_add,
        _wrap_sub,
        TraceVector.copy,
        TraceVector.set_from,
    )

    # Apply contribution of trace to the system x = A^{-1} y - A^{-1} N^T lambda
    system.apply_trace_transpose(trace_lhs, tmp_d1)
    system.apply_diagonal_inverse(tmp_d1, tmp_d2)
    DenseVector.subtract(inv_a_y, tmp_d2, inv_a_y, 1.0)

    return inv_a_y, trace_lhs, tr, itr_cnt


@dataclass
class FullVector:
    """Class contaning a dense and trace vector."""

    dense: DenseVector
    trace: TraceVector

    def __post_init__(self) -> None:
        """Check both are from the same parent."""
        if self.dense.parent is not self.trace.parent:
            raise ValueError("Both parts must have the same parent.")

    @classmethod
    def make_empty(cls, system: LinearSystem) -> Self:
        """Create an empty vector."""
        return cls(DenseVector(system), TraceVector(system))

    @staticmethod
    def dot(v1: FullVector, v2: FullVector) -> float:
        """Compute dot product of two vectors."""
        return DenseVector.dot(v1.dense, v2.dense) + TraceVector.dot(v1.trace, v2.trace)

    def copy(self) -> FullVector:
        """Create a copy of itself."""
        return FullVector(self.dense.copy(), self.trace.copy())

    def set_from(self, other: FullVector) -> None:
        """Set the vector from another."""
        self.dense.set_from(other.dense)
        self.trace.set_from(other.trace)

    @staticmethod
    def add(v1: FullVector, v2: FullVector, v_out: FullVector, k: float, /) -> None:
        """Add two vectors."""
        DenseVector.add(v1.dense, v2.dense, v_out.dense, k)
        TraceVector.add(v1.trace, v2.trace, v_out.trace, k)

    @staticmethod
    def subtract(v1: FullVector, v2: FullVector, v_out: FullVector, k: float, /) -> None:
        """Subtracts two vectors."""
        DenseVector.subtract(v1.dense, v2.dense, v_out.dense, k)
        TraceVector.subtract(v1.trace, v2.trace, v_out.trace, k)

    @staticmethod
    def scale(v: FullVector, k: float, v_out: FullVector, /) -> None:
        """Scale the vector by a constant."""
        DenseVector.scale(v.dense, k, v_out.dense)
        TraceVector.scale(v.trace, k, v_out.trace)


def solve_gmres_iterative(
    system: LinearSystem,
    rhs: DenseVector,
    constraints: TraceVector,
    convergence: ConvergenceSettings,
) -> tuple[DenseVector, TraceVector, float, int]:
    """Solve the system using GMRES."""
    # Set up the right side

    rhs_full = FullVector(rhs, constraints)

    dense_buffer1 = DenseVector(system)

    def wrapped_apply_system(
        sys: LinearSystem, v_in: FullVector, v_out: FullVector
    ) -> None:
        sys.apply_diagonal(v_in.dense, dense_buffer1)
        sys.apply_trace_transpose(v_in.trace, v_out.dense)
        sys.apply_trace(v_in.dense, v_out.trace)
        DenseVector.add(v_out.dense, dense_buffer1, v_out.dense, 1.0)

    solution, tr, itr_cnt = gmres_general(
        system,
        rhs_full,
        FullVector.make_empty(system),
        convergence,
        wrapped_apply_system,
        FullVector.dot,
        FullVector.add,
        FullVector.subtract,
        FullVector.scale,
        FullVector.copy,
    )

    return solution.dense, solution.trace, tr, itr_cnt


def solve_cg_iterative(
    system: LinearSystem,
    rhs: DenseVector,
    constraints: TraceVector,
    convergence: ConvergenceSettings,
) -> tuple[DenseVector, TraceVector, float, int]:
    """Solve the system using CG."""
    # Set up the right side

    rhs_full = FullVector(rhs, constraints)

    dense_buffer1 = DenseVector(system)
    dense_buffer2 = DenseVector(system)

    def wrapped_apply_system(sys: LinearSystem, v_in: FullVector) -> None:
        sys.apply_diagonal(v_in.dense, dense_buffer1)
        sys.apply_trace_transpose(v_in.trace, dense_buffer2)
        sys.apply_trace(v_in.dense, v_in.trace)
        DenseVector.add(dense_buffer1, dense_buffer2, v_in.dense, 1.0)

    def wrapped_add(v1: FullVector, v2: FullVector, k: float) -> None:
        DenseVector.add(v1.dense, v2.dense, v1.dense, k)
        TraceVector.add(v1.trace, v2.trace, v1.trace, k)

    def wrapped_sub(v1: FullVector, v2: FullVector, k: float) -> None:
        DenseVector.subtract(v1.dense, v2.dense, v1.dense, k)
        TraceVector.subtract(v1.trace, v2.trace, v1.trace, k)

    solution, tr, itr_cnt = cg_general(
        system,
        rhs_full,
        FullVector.make_empty(system),
        convergence,
        wrapped_apply_system,
        FullVector.dot,
        wrapped_add,
        wrapped_sub,
        FullVector.copy,
        FullVector.set_from,
    )

    return solution.dense, solution.trace, tr, itr_cnt


def solve_pcg_iterative(
    system: LinearSystem,
    rhs: DenseVector,
    constraints: TraceVector,
    convergence: ConvergenceSettings,
) -> tuple[DenseVector, TraceVector, float, int]:
    """Solve the system using preconditioned CG."""
    # Set up the right side

    rhs_full = FullVector(rhs, constraints)

    dense_buffer1 = DenseVector(system)
    dense_buffer2 = DenseVector(system)

    def wrapped_apply_system(
        sys: LinearSystem, v_in: FullVector, v_out: FullVector
    ) -> None:
        sys.apply_diagonal(v_in.dense, dense_buffer1)
        sys.apply_trace_transpose(v_in.trace, dense_buffer2)
        sys.apply_trace(v_in.dense, v_out.trace)
        DenseVector.add(dense_buffer1, dense_buffer2, v_out.dense, 1.0)

    def precondition_block_jacobi(
        sys: LinearSystem, v_in: FullVector, v_out: FullVector
    ) -> None:
        # Block Jacobi precondition
        sys.apply_diagonal_inverse(v_in.dense, v_out.dense)
        v_out.trace.set_from(v_in.trace)

    def wrapped_add(v1: FullVector, v2: FullVector, k: float, v_out: FullVector) -> None:
        DenseVector.add(v1.dense, v2.dense, v_out.dense, k)
        TraceVector.add(v1.trace, v2.trace, v_out.trace, k)

    def wrapped_sub(v1: FullVector, v2: FullVector, k: float, v_out: FullVector) -> None:
        DenseVector.subtract(v1.dense, v2.dense, v_out.dense, k)
        TraceVector.subtract(v1.trace, v2.trace, v_out.trace, k)

    solution, tr, itr_cnt = pcg_general(
        system,
        rhs_full,
        FullVector.make_empty(system),
        convergence,
        wrapped_apply_system,
        precondition_block_jacobi,
        FullVector.dot,
        wrapped_add,
        wrapped_sub,
        FullVector.copy,
    )

    return solution.dense, solution.trace, tr, itr_cnt
