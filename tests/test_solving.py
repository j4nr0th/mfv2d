"""Check that solving code works."""

from time import perf_counter

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import ElementFemSpace2D, compute_element_matrix
from mfv2d._mfv2d import (
    LinearSystem as CLinearSystem,
)
from mfv2d.continuity import connect_elements
from mfv2d.eval import CompiledSystem
from mfv2d.examples import unit_square_mesh
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import FemCache
from mfv2d.solve_system import compute_element_rhs
from mfv2d.solving import (
    ConvergenceSettings,
    DenseVector,
    LinearSystem,
    TraceVector,
    gmres_general,
    solve_schur_iterative,
)
from mfv2d.system import KFormSystem
from scipy.sparse import linalg as sla


def laplace_sample_system(
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

    forcing_vec_d = linear_system.create_block_vector(vectors)
    forcing_vec_t = linear_system.create_trace_vector(
        [con.rhs for con in continuity_constraints]
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
    sys, den, tra = laplace_sample_system(nh, nv, order)

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
    cden = csys.create_empty_dense_vector()
    merge_dense = den.combinded_system_vector()

    assert np.all(np.zeros_like(merge_dense) == cden.as_merged())

    for block, sbl in zip(den.data, cden.as_split(), strict=True):
        assert np.all(np.zeros_like(block) == sbl)

    # Check filled dense vectors can be created
    cden = csys.create_dense_vector(*den.data)

    assert np.all(merge_dense == cden.as_merged())

    for block, sbl in zip(den.data, cden.as_split(), strict=True):
        assert np.all(block == sbl)

    # Check empty trace vectors can be created
    ctra = csys.create_empty_trace_vector()
    merge_trace = tra.combinded_system_vector()

    assert np.all(np.zeros_like(merge_trace) == ctra.as_merged())

    for svec1, svec2 in zip(tra.values, ctra.as_split(), strict=True):
        assert np.all(svec1.indices == svec2.indices)
        assert svec1.n == svec2.n

    # Check filled trace vectors can be created
    ctra = csys.create_trace_vector(merge_trace)

    assert np.all(merge_trace == ctra.as_merged())

    for svec1, svec2 in zip(tra.values, ctra.as_split(), strict=True):
        assert np.all(svec1.indices == svec2.indices)
        assert np.all(svec1.values == svec2.values)
        assert svec1.n == svec2.n


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_c_operations(nh: int, nv: int, order: int) -> None:
    """Check that C operators behave same as their Python counterparts."""
    sys, den, tra = laplace_sample_system(nh, nv, order)

    # Check the system itself works as expected
    csys = CLinearSystem(
        tuple((block.diagonal_block, block.constraint_matrix) for block in sys.blocks),
        np.array(sys.constraints.element_array_offsets, np.uint32),
        np.array(sys.constraints.associated_element_array, np.uint32),
    )

    cden = csys.create_dense_vector(*den.data)

    ctra = csys.create_trace_vector(tra.combinded_system_vector())

    c_d_out = csys.create_empty_dense_vector()
    c_t_out = csys.create_empty_trace_vector()

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
    linear_system, forcing_dense, forcing_trace = laplace_sample_system(nh, nv, order)

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


@pytest.mark.parametrize(("n", "m"), ((10, 100), (3, 10), (5, 50)))
def test_gmres(n: int, m: int):
    """Check that GMRES solver computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    rng = np.random.default_rng()
    mat = rng.random((n, n))
    lhs = rng.random((n,))
    rhs = mat @ lhs
    solution, residual, k = gmres_general(
        mat,
        rhs,
        ConvergenceSettings(
            maximum_iterations=m, absolute_tolerance=1e-9, relative_tolerance=1e-7
        ),
        np.ndarray.__matmul__,
        np.dot,
        np.ndarray.__add__,
        np.ndarray.__sub__,
        np.ndarray.__mul__,
    )

    assert pytest.approx(solution) == lhs


@pytest.mark.parametrize(("nh", "nv", "order"), _TEST_DIMS)
def test_schur(nh: int, nv: int, order: int):
    """Check that Schur's compliment solver computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    linear_system, forcing_dense, forcing_trace = laplace_sample_system(nh, nv, order)

    combined_matrix = linear_system.combined_system_matrix()
    combined_vec = np.concatenate(
        (forcing_dense.combinded_system_vector(), forcing_trace.combinded_system_vector())
    )

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

    sol_d, sol_t, residual = solution_schur

    combined_schur = np.concatenate(
        (sol_d.combinded_system_vector(), sol_t.combinded_system_vector())
    )
    print("Estimated residual is:", residual)
    print("Max difference: ", np.max(np.abs(solution_scipy - combined_schur)))
    assert pytest.approx(solution_scipy) == combined_schur


if __name__ == "__main__":
    for args in _TEST_DIMS:
        test_c_python_types(*args)
    for args in _TEST_DIMS:
        test_c_operations(*args)

    # test_gmres(10, 100)
    # test_multiplication(10, 10, 6)
    # test_schur(3, 3, 3)
    # test_schur(20, 15, 5)
    # test_schur(3, 4, 4)
    # test_schur(5, 2, 5)
