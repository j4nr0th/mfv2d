"""Code related to solving the system of equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import SupportsIndex, cast

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


@dataclass(frozen=True)
class VectorBlock:
    """Block of a vector."""

    main_values: npt.NDArray[np.float64]
    trace_values: SparseVector

    @property
    def norm2(self) -> float:
        """Square of the L2 norm."""
        return np.dot(self.main_values, self.main_values) + self.trace_values.norm2


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
class LinearVector:
    """Type containing system vectors."""

    parent: LinearSystem
    blocks: tuple[VectorBlock, ...]

    def _check_other_implemented(self, other) -> bool:
        """Check that operations are implemented between the elements."""
        if type(other) is not LinearVector:
            return False

        if other.parent is not self.parent:
            raise ValueError(
                "Only vectors that belong to same parents can be operated on."
            )

        return True

    def __add__(self, other: LinearVector) -> LinearVector:
        """Add vectors."""
        if not self._check_other_implemented(other):
            return NotImplemented

        return LinearVector(
            self.parent,
            tuple(
                VectorBlock(
                    sv.main_values + ov.main_values, sv.trace_values + ov.trace_values
                )
                for sv, ov in zip(self.blocks, other.blocks, strict=True)
            ),
        )

    def __sub__(self, other: LinearVector) -> LinearVector:
        """Subtract vectors."""
        if not self._check_other_implemented(other):
            return NotImplemented

        return LinearVector(
            self.parent,
            tuple(
                VectorBlock(
                    sv.main_values - ov.main_values, sv.trace_values - ov.trace_values
                )
                for sv, ov in zip(self.blocks, other.blocks, strict=True)
            ),
        )

    def __mul__(self, other: float) -> LinearVector:
        """Multiply the vector."""
        if not (isinstance(other, float) or isinstance(other, int)):
            return NotImplemented
        return LinearVector(
            self.parent,
            tuple(
                VectorBlock(
                    sv.main_values * other,
                    SparseVector.from_entries(
                        sv.trace_values.n,
                        sv.trace_values.indices,
                        other * sv.trace_values.values,
                    ),
                )
                for sv in (self.blocks)
            ),
        )

    def __rmul__(self, other: float) -> LinearVector:
        """Multiply the vector."""
        return self.__mul__(other)

    def __div__(self, other: float) -> LinearVector:
        """Divide the vector."""
        return self * (1 / other)

    def __matmul__(self, other: LinearVector) -> float:
        """Compute dot product of two vectors."""
        if not self._check_other_implemented(other):
            return NotImplemented

        return np.sum(
            [
                sv.trace_values.dot(ov.trace_values)
                + np.dot(sv.main_values, ov.main_values)
                for sv, ov in zip(self.blocks, other.blocks, strict=True)
            ]
        )

    def __rmatmul__(self, other: LinearVector) -> float:
        """Compute dot product of two vectors."""
        return self.__matmul__(other)

    def combinded_system_vector(self) -> npt.NDArray[np.float64]:
        """Combine all blocks together into a single vector."""
        return np.concatenate(
            [
                *(block.main_values for block in self.blocks),
                SparseVector.merge_to_dense(
                    *(block.trace_values for block in self.blocks)
                ),
            ],
            dtype=np.float64,
        )

    @property
    def norm2(self) -> float:
        """Square of the L2 norm."""
        return np.sum([block.norm2 for block in self.blocks])


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

    def create_vector(
        self,
        element_vectors: Sequence[npt.ArrayLike] | None = None,
        constraint_vals: npt.ArrayLike | None = None,
    ) -> LinearVector:
        """Create a new vector associated with the system."""
        vecs: list[npt.NDArray[np.float64]]
        cons: npt.NDArray[np.float64]
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

        return LinearVector(
            self, tuple(VectorBlock(ev, ec) for ev, ec in zip(vecs, cv, strict=True))
        )

    def __matmul__(self, other: LinearVector) -> LinearVector:
        """Multiply by a system related vector."""
        if type(other) is not LinearVector:
            return NotImplemented

        if other.parent is not self:
            raise ValueError(
                "Can only multiply a vector with a system that's its parent."
            )

        element_vecs: list[npt.NDArray[np.float64]] = list()
        element_cons: list[SparseVector] = [
            SparseVector.from_pairs(self.constraints.count)
        ] * self.n_blocks

        for ie in range(self.n_blocks):
            sys = self.blocks[ie]
            val = other.blocks[ie]

            dense_result = (
                sys.diagonal_block @ val.main_values  # Element system part
                + val.trace_values @ sys.constraint_matrix  # Element constraints
            )
            # NOTE: if this always returned a sparse vector, it would be more efficient
            trace_result = sys.constraint_matrix @ val.main_values
            assert type(trace_result) is np.ndarray
            nonzero = np.flatnonzero(trace_result != 0.0)
            sparse_trace = SparseVector.from_entries(
                trace_result.size, nonzero, trace_result[nonzero]
            )
            for idx in sparse_trace.indices:
                elements = self.constraints.get_value_associated_elements(idx)
                for ie in elements:
                    element_cons[ie] += SparseVector.from_pairs(
                        self.constraints.count, (idx, sparse_trace[idx])
                    )

            element_vecs.append(dense_result)

        return LinearVector(
            self,
            tuple(
                VectorBlock(mv, tv)
                for mv, tv in zip(element_vecs, element_cons, strict=True)
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


@dataclass(frozen=True)
class LinearSolverSettings:
    """Settings for the linear solver."""

    atol: float
    rtol: float
    aiter: int
    riter: float


@dataclass(frozen=True)
class LinearSolverOutput:
    """Output of running a linear solver."""

    solution: LinearVector
    residual_estimate: float
    iterations_taken: int
    error_history: npt.NDArray[np.float64]


def solve_bicgstab(
    system: LinearSystem,
    rhs: LinearVector,
    lhs: LinearVector | None = None,
    settings: LinearSolverSettings = LinearSolverSettings(
        atol=1e-10, rtol=1e-9, aiter=1000, riter=0.1
    ),
) -> LinearSolverOutput:
    """Adjust the left vector such that applying the system on it results in the right."""
    if lhs is None:
        lhs = system.create_vector()

    residual = rhs - system @ lhs
    ymag = np.sqrt(rhs.norm2)
    tol = min(settings.atol, settings.rtol * ymag)
    iters = min(settings.aiter, int(settings.riter * system.total_size))

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    rQ = residual
    p = residual
    r = residual

    errors = np.empty(iters)

    iter_cnt = 0
    for iter_cnt in range(iters):
        Ap = system @ p
        rQAp = rQ @ Ap
        alpha = rho / rQAp

        lhs += alpha * p

        s = r - alpha * Ap
        sksk_dp = s.norm2
        err = np.sqrt(sksk_dp)
        if err < tol:
            errors[iter_cnt] = err
            break

        As = system @ s
        sAAs_dp = As.norm2
        sAs_dp = s @ As

        omega = sAs_dp / sAAs_dp

        lhs += omega * s
        r = s - omega * As
        rkrk_dp = r.norm2
        err = np.sqrt(rkrk_dp)
        errors[iter_cnt] = err
        if err < tol:
            break
        rQrk_dp = rQ @ r
        beta = rQrk_dp / rho * alpha / omega
        rho = rQrk_dp
        p = r + beta * (p - omega * Ap)

    return LinearSolverOutput(lhs, errors[iter_cnt], iter_cnt, errors[:iter_cnt])


def solve_schur(system: LinearSystem, rhs: LinearVector) -> LinearVector:
    """Solve the system using Schur's compliment."""
    trace_transform = tuple(
        sb.constraint_matrix @ MatrixCRS.from_dense(la.inv(sb.diagonal_block))
        for sb in system.blocks
    )

    ## Serial part begins
    full_trace = SparseVector.merge_to_dense(
        *(block.trace_values for block in rhs.blocks)
    )

    trace_forcing = (
        SparseVector.merge_to_dense(
            *(
                tt.multiply_to_sparse(vb.main_values)
                for tt, vb in zip(trace_transform, rhs.blocks, strict=True)
            ),
            duplicates="sum",
        )
        - full_trace
    )
    del full_trace

    trace_matrix = np.zeros((trace_forcing.size, trace_forcing.size), np.float64)
    for tt, sb in zip(trace_transform, system.blocks, strict=True):
        (tt @ sb.constraint_matrix.transpose()).add_to_dense(trace_matrix)
    del trace_transform

    trace_solution = la.solve(
        trace_matrix,
        trace_forcing,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=False,
        assume_a="general",
    )
    del trace_matrix, trace_forcing
    ## Serial part ends

    split_trace = [
        SparseVector.from_entries(
            block.trace_values.n,
            block.trace_values.indices,
            trace_solution[block.trace_values.indices],
        )
        for block in rhs.blocks
    ]
    del trace_solution

    return LinearVector(
        system,
        tuple(
            VectorBlock(
                np.astype(
                    np.linalg.solve(
                        sb.diagonal_block,
                        vb.main_values - np.array(st @ sb.constraint_matrix),
                    ),
                    np.float64,
                    copy=False,
                ),
                st,
            )
            for sb, vb, st in zip(system.blocks, rhs.blocks, split_trace, strict=True)
        ),
    )
