"""Check that the sparse vector objects work as intended."""

import numpy as np
import pytest
from interplib._mimetic import SparseVector


@pytest.mark.parametrize(("n", "m"), ((3, 2), (10, 3), (100, 99)))
def test_construction(n: int, m: int) -> None:
    """Check that construction is done correctly."""
    assert m <= n
    np.random.seed(0)
    caught = False
    i = np.astype(np.unique(np.random.randint(0, n, m)), np.uint64)
    x = np.random.random_sample(i.size)

    # Wrong size for x
    try:
        _ = SparseVector.from_entries(n, i, x[:-1])
    except ValueError:
        caught = True
    assert caught
    caught = False

    # Wrong size for i
    try:
        _ = SparseVector.from_entries(n, i[:-1], x)
    except ValueError:
        caught = True
    assert caught
    caught = False

    # Index out of bounds
    try:
        i2 = np.array(i)
        i2[-1] = n
        _ = SparseVector.from_entries(n, i2, x)
    except ValueError:
        caught = True
    assert caught
    caught = False

    # Not sorted
    try:
        i2 = np.array(i)
        i2[0] = n - 1
        _ = SparseVector.from_entries(n, i2, x)
    except ValueError:
        caught = True
    assert caught
    caught = False

    # Correct
    s = SparseVector.from_entries(n, i, x)

    assert np.all(s.indices == i)
    assert np.all(s.values == x)
    assert s.n == n


@pytest.mark.parametrize(("n", "m"), ((3, 2), (10, 3), (100, 99)))
def test_from_array(n: int, m: int) -> None:
    """Check that conversion to and from numpy arrays works."""
    np.random.seed(0)
    a = np.random.random_sample(n)
    where = np.unique(np.random.randint(0, n, n - m))
    a[where] = 0
    i = np.astype(np.flatnonzero(a), np.uint64)
    s = SparseVector.from_entries(n, i, a[i])
    assert np.all(np.array(s) == a)
