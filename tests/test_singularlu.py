"""Check that the singular LU decomposition works."""

import numpy as np
import pytest
from interplib.decomp import SingularLU


@pytest.mark.parametrize("n", (1, 2, 3, 4, 10))
def test_nonsingular(n: int) -> None:
    """Check that it correctly solves non-singular systems."""
    np.random.seed(158)
    m = np.random.random_sample((n, n))
    v = np.random.random_sample(n)

    y = m @ v
    assert np.linalg.det(m) != 0
    assert SingularLU(m).solve(y) == pytest.approx(v)


@pytest.mark.parametrize("n", (2, 3, 4, 10))
def test_singular(n: int) -> None:
    """Check that it correctly solves systems with rank n-1."""
    np.random.seed(158)
    m = np.random.random_sample((n - 1, n))
    m = m.T @ m
    v = np.random.random_sample(n)

    y = m @ v
    assert m @ SingularLU(m).solve(y) == pytest.approx(m @ v)


@pytest.mark.parametrize("n", (1, 2, 3, 4, 10))
def test_very_singular(n: int) -> None:
    """Check that it correctly solves systems with rank 1."""
    np.random.seed(158)
    m = np.random.random_sample((1, n))
    m = m.T @ m
    v = np.random.random_sample(n)

    y = m @ v
    assert m @ SingularLU(m).solve(y) == pytest.approx(m @ v)
