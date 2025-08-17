"""Check that solving code works."""

from time import perf_counter

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import ElementFemSpace2D, compute_element_matrix
from mfv2d.continuity import connect_elements
from mfv2d.eval import CompiledSystem
from mfv2d.examples import unit_square_mesh
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import FemCache
from mfv2d.solve_system import compute_element_rhs
from mfv2d.solving import LinearSystem, solve_schur
from mfv2d.system import KFormSystem
from scipy.sparse import linalg as sla


@pytest.mark.parametrize(("nh", "nv", "order"), ((10, 10, 3), (3, 4, 4), (5, 2, 5)))
def test_schur(nh: int, nv: int, order: int):
    """Check that Schur's compliment solver computes the same solution as SciPy."""
    # Problem to be solved is based on mixed Laplace
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
    q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)

    v = u.weight
    p = q.weight

    def u_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """Exact solution used."""
        return np.cos(np.pi * x) * np.cos(np.pi * y)

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

    forcing_vec = linear_system.create_vector(
        vectors, [con.rhs for con in continuity_constraints]
    )

    combined_matrix = linear_system.combined_system_matrix()
    combined_vec = forcing_vec.combinded_system_vector()

    tsc0 = perf_counter()
    solution_schur = solve_schur(linear_system, forcing_vec)
    tsc1 = perf_counter()
    tsp0 = perf_counter()
    solution_scipy = sla.spsolve(combined_matrix, combined_vec)
    tsp1 = perf_counter()

    print(
        f"Time taken by Schur solve for {combined_matrix.shape} system is {tsc1 - tsc0:g}"
        " seconds"
    )

    print(
        f"Time taken by Scipy solve for {combined_matrix.shape} system is {tsp1 - tsp0:g}"
        " seconds"
    )

    combined_schur = solution_schur.combinded_system_vector()
    assert pytest.approx(solution_scipy) == combined_schur


if __name__ == "__main__":
    test_schur(10, 10, 3)
