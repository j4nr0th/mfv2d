"""Test that the LiLMatrix works well."""

import numpy as np
import pytest
from interplib._mimetic import LiLMatrix, SparseVector
from scipy import sparse as sp


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


@pytest.mark.parametrize("n", (2, 3, 20, 100))
def test_decomposition(n: int) -> None:
    """Check that QR decomposition makes sense."""
    np.random.seed(0)
    a = np.random.random_sample((n, n))
    r = LiLMatrix.from_full(a)
    assert np.all(np.array(r) == a)
    q = r.qr_decompose()

    for g in q:
        a = g @ a

    assert pytest.approx(a) == np.array(r)


@pytest.mark.parametrize("n", (2, 3, 5))
def test_block_diag(n: int) -> None:
    """Check that the block diagonal works."""
    np.random.seed(0)
    matrices = list()
    sizes = np.random.randint(1, 5, n)
    for s in sizes:
        matrices.append(np.random.random_sample((s, s)))

    m = LiLMatrix.block_diag(*(LiLMatrix.from_full(mat) for mat in matrices))
    ms = sp.block_diag(matrices)

    assert np.all(ms.toarray() == np.array(m))
