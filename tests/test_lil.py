"""Test that the LiLMatrix works well."""

import numpy as np
import pytest
from interplib._mimetic import LiLMatrix, SparseVector


@pytest.mark.parametrize(("n", "m"), ((10, 1), (10, 10), (3, 4), (4, 3)))
def test_construction(n: int, m: int) -> None:
    """Check that it is possible to manually assemble it."""
    np.random.seed(0)
    a = np.random.random_sample((n, m))
    mask = a >= 0.5
    a[mask] = 0
    lil = LiLMatrix(a.shape[0], a.shape[1])
    for i in range(n):
        row = a[i]
        where = np.astype(np.flatnonzero(row), np.uint64)
        s = SparseVector.from_entries(m, where, row[where])
        lil[i] = s

    assert np.all(np.array(lil) == a)
