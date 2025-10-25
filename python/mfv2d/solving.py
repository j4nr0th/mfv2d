"""Code related to solving the system of equations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import SupportsIndex, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy import linalg as la
from scipy import sparse as sp

from mfv2d._mfv2d import MatrixCRS, SparseVector
from mfv2d.mimetic2d import Constraint
from mfv2d.system import ElementFormSpecification


@dataclass(frozen=True)
class SystemBlock:
    """Block of the system."""

    diagonal_block: npt.NDArray[np.float64]
    constraint_matrix: MatrixCRS
    _inverse_diagonal: tuple | None = None

    def apply_diag_inverse(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]:
        """Apply inverse of the diagonal block."""
        if self._inverse_diagonal is None:
            object.__setattr__(
                self,
                "_inverse_diagonal",
                la.lu_factor(self.diagonal_block, check_finite=False),
            )

        return np.asarray(la.lu_solve(self._inverse_diagonal, x), np.float64, copy=False)


@dataclass(frozen=True)
class DenseVector:
    """Block of a dense vector."""

    parent: LinearSystem
    data: tuple[npt.NDArray[np.float64], ...]

    def __sub__(self, other: DenseVector) -> DenseVector:
        """Take the difference of the dense vectors."""
        if other.parent is not self.parent:
            raise ValueError("Can not subtract two dense vectors with different parents.")

        return DenseVector(
            self.parent,
            tuple(sd - od for sd, od in zip(self.data, other.data, strict=True)),
        )

    def __add__(self, other: DenseVector) -> DenseVector:
        """Add the dense vectors together."""
        if other.parent is not self.parent:
            raise ValueError("Can not add two dense vectors with different parents.")

        return DenseVector(
            self.parent,
            tuple(sd + od for sd, od in zip(self.data, other.data, strict=True)),
        )

    def combinded_system_vector(self) -> npt.NDArray[np.float64]:
        """Combine the blocks into a single vector."""
        return np.concatenate(self.data, dtype=np.float64)


@dataclass(frozen=True)
class TraceVector:
    """Part of the trace vector."""

    parent: LinearSystem
    values: tuple[SparseVector, ...]

    def __sub__(self, other: TraceVector) -> TraceVector:
        """Take the difference of the trace."""
        if other.parent is not self.parent:
            raise ValueError("Can not subtract two trace vectors with different parents.")

        return TraceVector(
            self.parent,
            tuple((ss - so) for ss, so in zip(self.values, other.values, strict=True)),
        )

    def __add__(self, other: TraceVector) -> TraceVector:
        """Add traces."""
        if other.parent is not self.parent:
            raise ValueError("Can not add two trace vectors with different parents.")

        return TraceVector(
            self.parent,
            tuple((ss + so) for ss, so in zip(self.values, other.values, strict=True)),
        )

    def __mul__(self, other: float) -> TraceVector:
        """Add traces."""
        return TraceVector(
            self.parent,
            tuple((ss.__mul__(other)) for ss in self.values),
        )

    def dot(self, other: TraceVector) -> float:
        """Compute dot product with other trace vector."""
        return sum(
            [
                SparseVector.dot(ss, so)
                for ss, so in zip(self.values, other.values, strict=True)
            ]
        )

    def combinded_system_vector(self) -> npt.NDArray[np.float64]:
        """Combine the blocks into a single vector."""
        return SparseVector.merge_to_dense(*self.values, duplicates="last")


@dataclass(frozen=True)
class SystemConstraintRelations:
    """Description of what elements individual constraints are in."""

    associated_element_array: npt.NDArray[np.uint32]
    element_array_offsets: npt.NDArray[np.uint32]

    def get_value_associated_elements(
        self, idx: SupportsIndex, /
    ) -> npt.NDArray[np.uint32]:
        """Get element indices for elements associated with the value at index."""
        i = int(idx)
        return self.associated_element_array[
            self.element_array_offsets[i] : self.element_array_offsets[i + 1]
        ]

    @property
    def count(self) -> int:
        """Number of constraints."""
        return self.element_array_offsets.size - 1


@dataclass(frozen=True)
class LinearSystem:
    """Type containing system blocks."""

    blocks: tuple[SystemBlock, ...]
    constraints: SystemConstraintRelations

    def __init__(
        self,
        n_elem: int,
        form_spec: ElementFormSpecification,
        orders: npt.NDArray[np.uint32],
        element_matrices: Sequence[npt.NDArray[np.float64]],
        constraints: Sequence[Constraint],
    ) -> None:
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
            SystemBlock(
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
                system.constraint_matrix.build_row(
                    ic,
                    SparseVector.from_entries(
                        system.constraint_matrix.shape[1], ec.dofs, ec.coeffs
                    ),
                )

            c_elems.append(
                np.array([ec.i_e for ec in con.element_constraints], np.uint32)
            )
            c_vals.append(con.rhs)
            c_lens.append(c_elems[-1].size)

            for ie in (i for i in range(n_elem) if i not in present_elements):
                system_blocks[ie].constraint_matrix.build_row(ic)

        object.__setattr__(self, "blocks", system_blocks)
        object.__setattr__(
            self,
            "constraints",
            SystemConstraintRelations(np.concatenate(c_elems), np.cumsum(c_lens)),
        )

    def combined_system_matrix(self) -> sp.csr_array:
        """Combine the system matrix into a scipy CSR array."""
        diagonal_part = sp.block_diag([system.diagonal_block for system in self.blocks])
        lagrange_block = sp.block_array(
            [
                [
                    sp.csc_array(
                        (
                            system.constraint_matrix.values,
                            (
                                system.constraint_matrix.row_indices,
                                system.constraint_matrix.column_indices,
                            ),
                        ),
                        shape=system.constraint_matrix.shape,
                    )
                    for system in self.blocks
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

    def create_block_vector(
        self,
        element_vectors: Sequence[npt.ArrayLike] | None = None,
    ) -> DenseVector:
        """Create a vector associated with the system."""
        vecs: list[npt.NDArray[np.float64]]
        if element_vectors is None:
            vecs = [
                np.zeros(element.diagonal_block.shape[0], np.float64)
                for element in self.blocks
            ]

        elif len(element_vectors) != self.n_blocks:
            raise ValueError(
                "Number of element vectors given does not match the number of elements in"
                " the system."
            )

        else:
            vecs = list()
            for i, (v, system) in enumerate(zip(element_vectors, self.blocks)):
                array = np.asarray(v, np.float64, copy=None)
                if array.shape != (system.diagonal_block.shape[0],):
                    raise ValueError(
                        f"Element vector {i} did not have the same length as element "
                        "had degrees of freedom"
                    )
                vecs.append(array)

        return DenseVector(self, tuple(vecs))

    def create_trace_vector(
        self,
        constraint_vals: npt.ArrayLike | None = None,
    ) -> TraceVector:
        """Create a new trace vector associated with the system."""
        if constraint_vals is None:
            cv: list[SparseVector] = list()
            for block in self.blocks:
                indices = block.constraint_matrix.nonempty_rows
                cv.append(
                    SparseVector.from_entries(
                        block.constraint_matrix.shape[0], indices, np.zeros_like(indices)
                    )
                )
        else:
            cons = np.asarray(constraint_vals, np.float64, copy=None)
            if cons.shape != (self.constraints.count,):
                raise ValueError(
                    "Number of constraint values given did not match the number of "
                    "constraints in the system."
                )
            cv = list()
            for block in self.blocks:
                indices = block.constraint_matrix.nonempty_rows
                cv.append(
                    SparseVector.from_entries(
                        block.constraint_matrix.shape[0], indices, cons[indices]
                    )
                )

        return TraceVector(self, tuple(cv))

    def _check_input_output(
        self,
        input_v: DenseVector | TraceVector,
        output_v: DenseVector | TraceVector,
    ) -> None:
        """Check input and output vectors are indeed based on this system."""
        if input_v.parent is not self:
            raise ValueError("Input vector is not based on the system.")
        if output_v.parent is not self:
            raise ValueError("Output vector is not based on the system.")

    def apply_diagonal(
        self, v: DenseVector, /, out: DenseVector | None = None
    ) -> DenseVector:
        """Apply multiplication by the diagonal part."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.data, out.data, strict=True):
            ob[:] = block.diagonal_block @ vb[:]

        return out

    def apply_diagonal_inverse(
        self, v: DenseVector, /, out: DenseVector | None = None
    ) -> DenseVector:
        """Apply multiplication by the diagonal part."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.data, out.data, strict=True):
            ob[:] = block.apply_diag_inverse(vb[:])

        return out

    def apply_trace(self, v: DenseVector, /) -> TraceVector:
        """Apply the trace (constraint) part of the system."""
        dense_merged: list[SparseVector] = list()
        for block, vb in zip(self.blocks, v.data, strict=True):
            trace_output = block.constraint_matrix.multiply_to_sparse(vb)
            dense_merged.append(trace_output)

        return self.create_trace_vector(
            SparseVector.merge_to_dense(*dense_merged, duplicates="sum")
        )

    def apply_transpose_trace(
        self, v: TraceVector, /, out: DenseVector | None = None
    ) -> DenseVector:
        """Apply transpose of the trace to the trace vector."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.values, out.data, strict=True):
            ob[:] = vb @ block.constraint_matrix

        return out

    def apply_trace_system(self, v: TraceVector, /) -> TraceVector:
        """Apply the system for trace variables from Schur's compliment."""
        return self.apply_trace(
            self.apply_diagonal_inverse(
                self.apply_transpose_trace(
                    v,
                ),
            ),
        )

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the system."""
        return len(self.blocks)

    @property
    def total_size(self) -> int:
        """Total size of the system."""
        return self.constraints.count + sum(
            block.diagonal_block.shape[0] for block in self.blocks
        )

    # def precondition(self, other: LinearVector) -> LinearVector:
    #     """Apply the preconditioner to the vector."""
    #     if other.parent is not self:
    #         raise ValueError(
    #             "Can only multiply a vector with a system that's its parent."
    #         )

    #     return LinearVector(
    #         self,
    #         tuple(
    #             VectorBlock(sb.apply_diag_inverse(vb.main_values), vb.trace_values)
    #             for sb, vb in zip(self.blocks, other.blocks, strict=True)
    #         ),
    #     )

    # def reverse_precondition(self, other: LinearVector) -> None:
    #     """Apply the inverse of the preconditioner to the vector."""
    #     if other.parent is not self:
    #         raise ValueError(
    #             "Can only multiply a vector with a system that's its parent."
    #         )
    #     for sb, vb in zip(self.blocks, other.blocks, strict=True):
    #         vb.main_values[:] = sb.diagonal_block @ vb.main_values


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
    convergence: ConvergenceSettings,
    system_application_function: Callable[[_Mat, _Vec], _Vec],
    vec_dot_function: Callable[[_Vec, _Vec], float],
    vec_add_function: Callable[[_Vec, _Vec], _Vec],
    vec_sub_function: Callable[[_Vec, _Vec], _Vec],
    vec_scale_function: Callable[[_Vec, float], _Vec],
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

    # First residual
    p = rhs
    # Get magnitude of the residual
    r_mag = np.sqrt(vec_dot_function(p, p))
    # Normalize the vector
    p = vec_scale_function(p, 1 / r_mag)
    # Add it to the current collection of basis and LSQR state
    p_vecs.append(p)
    g[0] = r_mag

    for k in range(1, m):
        # Make a new basis vector
        p = system_application_function(mat, p)
        # Make it orthogonal to other basis
        for li in range(k):
            p_old = p_vecs[li]
            pp_dp = vec_dot_function(p, p_old)
            h[li] = pp_dp
            p = vec_sub_function(p, vec_scale_function(p_old, pp_dp))

        # Get the magnitude and normalize it
        p_mag2 = vec_dot_function(p, p)  # Surprise tool for later
        p_mag = np.sqrt(p_mag2)
        p = vec_scale_function(p, 1 / p_mag)
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
    for i in range(k):
        p_vecs[i] = vec_scale_function(p_vecs[i], alpha[i])
    sol = p_vecs[0]
    for i in range(1, k):
        sol = vec_add_function(sol, p_vecs[i])
    return sol, r_mag, k


def solve_schur_iterative(
    system: LinearSystem,
    rhs: DenseVector,
    constraints: TraceVector,
    convergence: ConvergenceSettings,
) -> tuple[DenseVector, TraceVector, float]:
    """Solve the system using Schur's compliment but iteratively."""
    # Compute rhs forcing for the Lagrange multipliers

    ### A^{-1} y
    inv_a_y = system.apply_diagonal_inverse(rhs)
    ### N A^{-1} y - phi
    trace_rhs = system.apply_trace(inv_a_y) - constraints

    # Iteratively solve the system for trace lambda
    trace_lhs, tr, _ = gmres_general(
        system,
        trace_rhs,
        convergence,
        LinearSystem.apply_trace_system,
        TraceVector.dot,
        TraceVector.__add__,
        TraceVector.__sub__,
        TraceVector.__mul__,
    )

    # Apply contribution of trace to the system x = A^{-1} y - A^{-1} N^T lambda
    solution = inv_a_y - system.apply_diagonal_inverse(
        system.apply_transpose_trace(trace_lhs)
    )

    return solution, trace_lhs, tr
