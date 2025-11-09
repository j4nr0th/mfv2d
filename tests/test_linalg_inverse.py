"""Check inverse is not busted."""

import numpy as np
import pytest
from mfv2d._mfv2d import _compute_matrix_inverse, _solve_linear_system


@pytest.mark.parametrize(("n", "m"), ((2, 3), (4, 5), (100, 4)))
def test_inverse(n: int, m: int) -> None:
    """Check that inverses of matrices match what numpy computes."""
    rng = np.random.default_rng(seed=n * m + n + m)
    for _ in range(m):
        mat = rng.uniform(-1, +1, (n, n))
        numpy_inverse = np.linalg.inv(mat)
        my_inverse = _compute_matrix_inverse(mat)
        assert pytest.approx(my_inverse) == numpy_inverse


@pytest.mark.parametrize(("n1", "n2", "m"), ((5, 4, 3), (10, 10, 5), (100, 3, 3)))
def test_solve(n1: int, n2: int, m: int) -> None:
    """Check that system is of equations is solved correclty."""
    rng = np.random.default_rng(seed=n1 * n2 * m + n1 + m + n2)
    for _ in range(m):
        mat = rng.uniform(-1, +1, (n1, n1))
        lhs = rng.uniform(-1, +1, (n1, n2))
        rhs = mat @ lhs
        my_inverse = _solve_linear_system(mat, rhs)
        assert pytest.approx(my_inverse) == lhs
