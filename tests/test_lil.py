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


def test_padding() -> None:
    """Check that adding rows and columns works as expected."""
    a00 = np.array([[0, 1, 2, 4], [2, 4, -1, 0], [2, 0, 0, 3]])
    extra_cols = (
        SparseVector.from_entries(3, (1, 2), (-1, +2)),
        SparseVector.from_entries(3, (0, 2), (-4, +4)),
        SparseVector.from_entries(3, (1,), (5,)),
    )
    extra_rows = (
        SparseVector.from_entries(7, (2, 5, 6), (0.1, 0.2, 0.3)),
        SparseVector.from_entries(7, (1, 2, 3), (1, 2, 3)),
    )
    m = LiLMatrix.from_full(a00)
    assert np.all(np.array(m) == a00)
    ac = np.stack([np.array(c) for c in extra_cols], axis=1)
    a01 = np.concatenate((a00, ac), axis=1)
    m.add_columns(*extra_cols)
    assert np.all(np.array(m) == a01)
    ar = np.stack([np.array(r) for r in extra_rows], axis=0)
    a02 = np.concatenate((a01, ar), axis=0)
    m2 = m.add_rows(*extra_rows)
    assert np.all(np.array(m2) == a02)
