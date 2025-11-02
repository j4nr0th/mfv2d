"""Check that solving code works."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import SupportsIndex, cast

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    ElementFemSpace2D,
    MatrixCRS,
    SparseVector,
    compute_element_matrix,
)
from mfv2d._mfv2d import (
    LinearSystem as CLinearSystem,
)
from mfv2d.continuity import connect_elements
from mfv2d.eval import CompiledSystem
from mfv2d.examples import unit_square_mesh
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import Constraint, FemCache
from mfv2d.solve_system import compute_element_rhs
from mfv2d.solving import (
    ConvergenceSettings,
    DenseVector,
    LinearSystem,
    TraceVector,
    solve_pcg_iterative,
    solve_schur_iterative,
)
from mfv2d.system import ElementFormSpecification, KFormSystem
from scipy import linalg as la
from scipy import sparse as sp
from scipy.sparse import linalg as sla


@dataclass(frozen=True)
class OldSystemBlock:
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
class OldDenseVector:
    """Block of a dense vector."""

    parent: OldLinearSystem
    data: tuple[npt.NDArray[np.float64], ...]

    def __sub__(self, other: OldDenseVector) -> OldDenseVector:
        """Take the difference of the dense vectors."""
        if other.parent is not self.parent:
            raise ValueError("Can not subtract two dense vectors with different parents.")

        return OldDenseVector(
            self.parent,
            tuple(sd - od for sd, od in zip(self.data, other.data, strict=True)),
        )

    def __add__(self, other: OldDenseVector) -> OldDenseVector:
        """Add the dense vectors together."""
        if other.parent is not self.parent:
            raise ValueError("Can not add two dense vectors with different parents.")

        return OldDenseVector(
            self.parent,
            tuple(sd + od for sd, od in zip(self.data, other.data, strict=True)),
        )

    def combinded_system_vector(self) -> npt.NDArray[np.float64]:
        """Combine the blocks into a single vector."""
        return np.concatenate(self.data, dtype=np.float64)


@dataclass(frozen=True)
class OldTraceVector:
    """Part of the trace vector."""

    parent: OldLinearSystem
    values: tuple[SparseVector, ...]

    def __sub__(self, other: OldTraceVector) -> OldTraceVector:
        """Take the difference of the trace."""
        if other.parent is not self.parent:
            raise ValueError("Can not subtract two trace vectors with different parents.")

        return OldTraceVector(
            self.parent,
            tuple((ss - so) for ss, so in zip(self.values, other.values, strict=True)),
        )

    def __add__(self, other: OldTraceVector) -> OldTraceVector:
        """Add traces."""
        if other.parent is not self.parent:
            raise ValueError("Can not add two trace vectors with different parents.")

        return OldTraceVector(
            self.parent,
            tuple((ss + so) for ss, so in zip(self.values, other.values, strict=True)),
        )

    def __mul__(self, other: float) -> OldTraceVector:
        """Add traces."""
        return OldTraceVector(
            self.parent,
            tuple((ss.__mul__(other)) for ss in self.values),
        )

    def dot(self, other: OldTraceVector) -> float:
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
class OldSystemConstraintRelations:
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
class OldLinearSystem:
    """Type containing system blocks."""

    blocks: tuple[OldSystemBlock, ...]
    constraints: OldSystemConstraintRelations

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
            OldSystemBlock(
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
            OldSystemConstraintRelations(np.concatenate(c_elems), np.cumsum(c_lens)),
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
    ) -> OldDenseVector:
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

        return OldDenseVector(self, tuple(vecs))

    def create_trace_vector(
        self,
        constraint_vals: npt.ArrayLike | None = None,
    ) -> OldTraceVector:
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

        return OldTraceVector(self, tuple(cv))

    def _check_input_output(
        self,
        input_v: OldDenseVector | OldTraceVector,
        output_v: OldDenseVector | OldTraceVector,
    ) -> None:
        """Check input and output vectors are indeed based on this system."""
        if input_v.parent is not self:
            raise ValueError("Input vector is not based on the system.")
        if output_v.parent is not self:
            raise ValueError("Output vector is not based on the system.")

    def apply_diagonal(
        self, v: OldDenseVector, /, out: OldDenseVector | None = None
    ) -> OldDenseVector:
        """Apply multiplication by the diagonal part."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.data, out.data, strict=True):
            ob[:] = block.diagonal_block @ vb[:]

        return out

    def apply_diagonal_inverse(
        self, v: OldDenseVector, /, out: OldDenseVector | None = None
    ) -> OldDenseVector:
        """Apply multiplication by the diagonal part."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.data, out.data, strict=True):
            ob[:] = block.apply_diag_inverse(vb[:])

        return out

    def apply_trace(self, v: OldDenseVector, /) -> OldTraceVector:
        """Apply the trace (constraint) part of the system."""
        dense_merged: list[SparseVector] = list()
        for block, vb in zip(self.blocks, v.data, strict=True):
            trace_output = block.constraint_matrix.multiply_to_sparse(vb)
            dense_merged.append(trace_output)

        return self.create_trace_vector(
            SparseVector.merge_to_dense(*dense_merged, duplicates="sum")
        )

    def apply_transpose_trace(
        self, v: OldTraceVector, /, out: OldDenseVector | None = None
    ) -> OldDenseVector:
        """Apply transpose of the trace to the trace vector."""
        if out is None:
            out = self.create_block_vector(None)
        self._check_input_output(v, out)

        for block, vb, ob in zip(self.blocks, v.values, out.data, strict=True):
            ob[:] = vb @ block.constraint_matrix

        return out

    def apply_trace_system(self, v: OldTraceVector, /) -> OldTraceVector:
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


def laplace_sample_system_old(
    nh: int, nv: int, order: int
) -> tuple[OldLinearSystem, OldDenseVector, OldTraceVector]:
    """Create the test Laplace system."""
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
    q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)

    v = u.weight
    p = q.weight

    def f_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Exact source used."""
        return -2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    system = KFormSystem(
        p * q + p.derivative * u == 0,
        v * q.derivative == v @ f_exact,
        sorting=lambda f: f.order,
    )

    # Sample mesh is used
    mesh = unit_square_mesh(nh, nv, order)

    compiled = CompiledSystem(system)

    # Create element FEM Spaces
    cache = FemCache(2)

    leaf_indices = mesh.get_leaf_indices()
    fem_spaces = tuple(
        ElementFemSpace2D(
            cache.get_basis2d(*mesh.get_leaf_orders(idx_leaf)),
            np.astype(mesh.get_leaf_corners(idx_leaf), np.float64, copy=False),
        )
        for idx_leaf in leaf_indices
    )

    vectors = [compute_element_rhs(system, space) for space in fem_spaces]
    matrices = [
        compute_element_matrix(system.unknown_forms, compiled.lhs_full, space)
        for space in fem_spaces
    ]
    continuity_constraints = connect_elements(system.unknown_forms, mesh)

    linear_system = OldLinearSystem(
        mesh.leaf_count,
        system.unknown_forms,
        np.array(
            [mesh.get_leaf_orders(idx_leaf) for idx_leaf in leaf_indices], np.uint32
        ),
        matrices,
        continuity_constraints,
    )

    forcing_vec_d = linear_system.create_block_vector(vectors)
    forcing_vec_t = linear_system.create_trace_vector(
        [con.rhs for con in continuity_constraints]
    )

    return linear_system, forcing_vec_d, forcing_vec_t


def laplace_sample_system_new(
    nh: int, nv: int, order: int
) -> tuple[LinearSystem, DenseVector, TraceVector]:
    """Create the test Laplace system."""
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
    q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)

    v = u.weight
    p = q.weight

    def f_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Exact source used."""
        return -2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    system = KFormSystem(
        p * q + p.derivative * u == 0,
        v * q.derivative == v @ f_exact,
        sorting=lambda f: f.order,
    )

    # Sample mesh is used
    mesh = unit_square_mesh(nh, nv, order)

    compiled = CompiledSystem(system)

    # Create element FEM Spaces
    cache = FemCache(2)

    leaf_indices = mesh.get_leaf_indices()
    fem_spaces = tuple(
        ElementFemSpace2D(
            cache.get_basis2d(*mesh.get_leaf_orders(idx_leaf)),
            np.astype(mesh.get_leaf_corners(idx_leaf), np.float64, copy=False),
        )
        for idx_leaf in leaf_indices
    )

    vectors = [compute_element_rhs(system, space) for space in fem_spaces]
    matrices = [
        compute_element_matrix(system.unknown_forms, compiled.lhs_full, space)
        for space in fem_spaces
    ]
    continuity_constraints = connect_elements(system.unknown_forms, mesh)

    linear_system = LinearSystem(
        mesh.leaf_count,
        system.unknown_forms,
        np.array(
            [mesh.get_leaf_orders(idx_leaf) for idx_leaf in leaf_indices], np.uint32
        ),
        matrices,
        continuity_constraints,
    )

    forcing_vec_d = DenseVector(linear_system, *vectors)
    forcing_vec_t = TraceVector(
        linear_system, np.array([con.rhs for con in continuity_constraints], np.float64)
    )

    return linear_system, forcing_vec_d, forcing_vec_t


_TEST_DIMS = (
    (10, 10, 3),
    (3, 4, 4),
    (5, 2, 5),
)


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_c_python_types(nh: int, nv: int, order: int) -> None:
    """Check that C types behave same as their Python counterparts."""
    sys, den, tra = laplace_sample_system_old(nh, nv, order)

    # Check the system itself works as expected
    csys = CLinearSystem(
        tuple((block.diagonal_block, block.constraint_matrix) for block in sys.blocks),
        np.array(sys.constraints.element_array_offsets, np.uint32),
        np.array(sys.constraints.associated_element_array, np.uint32),
    )

    for sblock, cd, cs in zip(
        sys.blocks,
        csys.get_dense_blocks(),
        csys.get_constraint_blocks(),
        strict=True,
    ):
        assert np.all(sblock.diagonal_block == cd)
        sb = sblock.constraint_matrix
        assert np.all(sb.row_indices == cs.row_indices)
        assert np.all(sb.row_offsets == cs.row_offsets)
        assert np.all(sb.values == cs.values)

    # Check empty dense vectors can be created
    cden = DenseVector(csys)
    merge_dense = den.combinded_system_vector()

    assert np.all(np.zeros_like(merge_dense) == cden.as_merged())

    for block, sbl in zip(den.data, cden.as_split(), strict=True):
        assert np.all(np.zeros_like(block) == sbl)

    # Check filled dense vectors can be created
    cden = DenseVector(csys, *den.data)

    assert np.all(merge_dense == cden.as_merged())

    for block, sbl in zip(den.data, cden.as_split(), strict=True):
        assert np.all(block == sbl)

    # Check empty trace vectors can be created
    ctra = TraceVector(csys)
    merge_trace = tra.combinded_system_vector()

    assert np.all(np.zeros_like(merge_trace) == ctra.as_merged())

    for svec1, svec2 in zip(tra.values, ctra.as_split(), strict=True):
        assert np.all(svec1.indices == svec2.indices)
        assert svec1.n == svec2.n

    # Check filled trace vectors can be created
    ctra = TraceVector(csys, merge_trace)

    assert np.all(merge_trace == ctra.as_merged())

    for svec1, svec2 in zip(tra.values, ctra.as_split(), strict=True):
        assert np.all(svec1.indices == svec2.indices)
        assert np.all(svec1.values == svec2.values)
        assert svec1.n == svec2.n


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_c_system_operations(nh: int, nv: int, order: int) -> None:
    """Check that C operators behave same as their Python counterparts."""
    sys, den, tra = laplace_sample_system_old(nh, nv, order)

    # Check the system itself works as expected
    csys = CLinearSystem(
        tuple((block.diagonal_block, block.constraint_matrix) for block in sys.blocks),
        np.array(sys.constraints.element_array_offsets, np.uint32),
        np.array(sys.constraints.associated_element_array, np.uint32),
    )

    cden = DenseVector(csys, *den.data)

    ctra = TraceVector(csys, tra.combinded_system_vector())

    c_d_out = DenseVector(csys)
    c_t_out = TraceVector(csys)

    # Check diagonal
    csys.apply_diagonal(cden, c_d_out)
    p_d_out = sys.apply_diagonal(den)
    assert pytest.approx(c_d_out.as_merged()) == p_d_out.combinded_system_vector()

    # Check diagonal inverse
    csys.apply_diagonal_inverse(cden, c_d_out)
    p_d_out = sys.apply_diagonal_inverse(den)
    assert pytest.approx(c_d_out.as_merged()) == p_d_out.combinded_system_vector()

    # Check trace
    csys.apply_trace(cden, c_t_out)
    p_t_out = sys.apply_trace(den)
    assert pytest.approx(c_t_out.as_merged()) == p_t_out.combinded_system_vector()

    # Check transpose
    csys.apply_trace_transpose(ctra, c_d_out)
    p_d_out = sys.apply_transpose_trace(tra)
    assert pytest.approx(c_d_out.as_merged()) == p_d_out.combinded_system_vector()


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_multiplication(nh: int, nv: int, order: int):
    """Check that system matrix computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    linear_system, forcing_dense, forcing_trace = laplace_sample_system_old(nh, nv, order)

    combined_matrix = linear_system.combined_system_matrix()
    combined_vec = np.concatenate(
        (forcing_dense.combinded_system_vector(), forcing_trace.combinded_system_vector())
    )

    tsc0 = perf_counter()
    pv1 = linear_system.apply_diagonal(
        forcing_dense
    ) + linear_system.apply_transpose_trace(forcing_trace)
    pv2 = linear_system.apply_trace(forcing_dense)
    tsc1 = perf_counter()

    tsp0 = perf_counter()
    u = combined_matrix @ combined_vec
    tsp1 = perf_counter()

    print(
        f"Time taken for Python multiply for {combined_matrix.shape} system is "
        f"{tsc1 - tsc0:g} seconds."
    )

    print(
        f"Time taken by Scipy multiply for {combined_matrix.shape} system is"
        f" {tsp1 - tsp0:g} seconds."
    )

    cpv = np.concatenate((pv1.combinded_system_vector(), pv2.combinded_system_vector()))
    assert pytest.approx(u) == cpv


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_schur(nh: int, nv: int, order: int):
    """Check that Schur's compliment solver computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    linear_system, forcing_dense, forcing_trace = laplace_sample_system_new(nh, nv, order)

    combined_matrix = linear_system.combined_system_matrix()
    combined_vec = np.concatenate((forcing_dense.as_merged(), forcing_trace.as_merged()))

    tsc0 = perf_counter()
    solution_schur = solve_schur_iterative(
        linear_system,
        forcing_dense,
        forcing_trace,
        convergence=ConvergenceSettings(
            maximum_iterations=nh * nv * order,
            absolute_tolerance=1e-14,
            relative_tolerance=1e-13,
        ),
    )
    tsc1 = perf_counter()
    tsp0 = perf_counter()
    solution_scipy = sla.spsolve(combined_matrix, combined_vec)
    tsp1 = perf_counter()

    print(f"System dimensionas are is {combined_matrix.shape}")

    print(
        f"Time taken by Schur solve for {combined_matrix.shape} system is {tsc1 - tsc0:g}"
        " seconds"
    )

    print(
        f"Time taken by Scipy solve for {combined_matrix.shape} system is {tsp1 - tsp0:g}"
        " seconds"
    )

    sol_d, sol_t, residual, iters = solution_schur

    combined_schur = np.concatenate((sol_d.as_merged(), sol_t.as_merged()))
    print("Estimated residual is:", residual)
    print("Number of itersations:", iters)
    print("Max difference: ", np.max(np.abs(solution_scipy - combined_schur)))
    assert pytest.approx(solution_scipy) == combined_schur


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_pcg(nh: int, nv: int, order: int):
    """Check that preconditioned CG solver computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    linear_system, forcing_dense, forcing_trace = laplace_sample_system_new(nh, nv, order)

    combined_matrix = linear_system.combined_system_matrix()
    combined_vec = np.concatenate((forcing_dense.as_merged(), forcing_trace.as_merged()))

    tsc0 = perf_counter()
    solution_pcg = solve_pcg_iterative(
        linear_system,
        forcing_dense,
        forcing_trace,
        convergence=ConvergenceSettings(
            maximum_iterations=nh * nv * order,
            absolute_tolerance=1e-14,
            relative_tolerance=1e-13,
        ),
    )
    tsc1 = perf_counter()
    tsp0 = perf_counter()
    solution_scipy = sla.spsolve(combined_matrix, combined_vec)
    tsp1 = perf_counter()

    print(f"System dimensionas are is {combined_matrix.shape}")

    print(
        f"Time taken by PCG solve for {combined_matrix.shape} system is {tsc1 - tsc0:g}"
        " seconds"
    )

    print(
        f"Time taken by Scipy solve for {combined_matrix.shape} system is {tsp1 - tsp0:g}"
        " seconds"
    )

    sol_d, sol_t, residual, iters = solution_pcg

    combined_schur = np.concatenate((sol_d.as_merged(), sol_t.as_merged()))
    print("Estimated residual is:", residual)
    print("Number of itersations:", iters)
    print("Max difference: ", np.max(np.abs(solution_scipy - combined_schur)))
    assert pytest.approx(solution_scipy, abs=1e-8) == combined_schur


if __name__ == "__main__":
    # for args in _TEST_DIMS:
    #     test_c_python_types(*args)
    # for args in _TEST_DIMS:
    #     test_c_system_operations(*args)

    # test_gmres(10, 100)
    # test_multiplication(10, 10, 6)
    # test_schur(3, 3, 3)
    # for _ in range(5):
    # test_cg(10, 10, 5)
    test_schur(25, 25, 5)
    test_pcg(25, 25, 5)
    # test_schur(3, 4, 4)
    # test_schur(5, 2, 5)
