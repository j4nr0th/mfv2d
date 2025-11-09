"""CRS matrix implementation test."""

import numpy as np
import pytest
from mfv2d._mfv2d import MatrixCRS, SparseVector


def random_sparse_vec(gen: np.random.Generator, n: int) -> SparseVector:
    """Create a random sparse vector."""
    nvals = gen.integers(0, n) + 1
    indices = np.unique(gen.integers(0, n, nvals, np.uint64))
    values = gen.uniform(-1, +1, indices.size)
    return SparseVector.from_entries(n, indices, values)


def random_matrix(gen: np.random.Generator, rows: int, cols: int) -> MatrixCRS:
    """Generate a random sparse matrix."""
    mat = MatrixCRS(rows, cols)
    for row in range(rows):
        new_row = random_sparse_vec(gen, cols)
        mat.build_row(row, new_row)
        assert mat.built_rows == row + 1
    return mat


_TEST_DIMS = ((3, 5), (5, 6), (20, 21))


@pytest.mark.parametrize(("rows", "cols"), _TEST_DIMS)
def test_building(rows: int, cols: int) -> None:
    """Check that matrix is built correctly."""
    mat = MatrixCRS(rows, cols)
    rng = np.random.default_rng(seed=0)
    row_data: list[SparseVector] = list()
    for row in range(rows):
        new_row = random_sparse_vec(rng, cols)
        row_data.append(new_row)
        mat.build_row(row, new_row)
        assert mat.built_rows == row + 1

    for row in range(rows):
        new_row = mat[row]
        assert row_data[row] == new_row

    # now it works
    mat.toarray()
    mat.column_indices
    mat.row_indices
    mat.row_offsets
    mat.values
    mat.position_pairs

    mat.build_row(rows // 2, row_data[rows // 2])
    assert mat.built_rows == rows // 2 + 1

    # Can not convert to an array
    with pytest.raises(RuntimeError):
        mat.toarray()

    # Can not get properties
    with pytest.raises(RuntimeError):
        mat.column_indices
    with pytest.raises(RuntimeError):
        mat.row_indices
    with pytest.raises(RuntimeError):
        mat.row_offsets
    with pytest.raises(RuntimeError):
        mat.values
    with pytest.raises(RuntimeError):
        mat.position_pairs

    # Can not index
    for row in range(rows):
        with pytest.raises(RuntimeError):
            new_row = mat[row]

    for row in range(rows // 2 + 1, rows):
        mat.build_row(row, row_data[row])
        assert mat.built_rows == row + 1

    # now it works
    mat.toarray()
    mat.column_indices
    mat.row_indices
    mat.row_offsets
    mat.values
    mat.position_pairs


@pytest.mark.parametrize(("rows", "cols"), _TEST_DIMS)
def test_multiplication(rows: int, cols: int) -> None:
    """Check that matrix vector multiplication works correctly."""
    rng = np.random.default_rng((rows * cols & rows) << max(cols, rows))
    mat = random_matrix(rng, rows, cols)

    # Check matrix right multiply
    for _ in range(max(rows, cols)):
        vs = random_sparse_vec(rng, rows)
        sparse = vs @ mat
        dense = np.array(vs) @ mat.toarray()
        assert pytest.approx(np.array(sparse)) == dense

    # Check matrix left multiply
    for _ in range(max(rows, cols)):
        vs = random_sparse_vec(rng, cols)
        sparse = mat @ vs
        dense = mat.toarray() @ np.array(vs)
        assert pytest.approx(np.array(sparse)) == dense

    # Check matrix right multiply
    for _ in range(max(rows, cols)):
        vd = rng.uniform(0, 1, (cols, rows))
        dsparse = vd @ mat
        dense = np.array(vd) @ mat.toarray()
        assert pytest.approx(np.array(dsparse)) == dense

    # Check matrix left multiply
    for _ in range(max(rows, cols)):
        vd = rng.uniform(0, 1, (cols, rows))
        dsparse = mat @ vd
        dense = mat.toarray() @ np.array(vd)
        assert pytest.approx(np.array(dsparse)) == dense


@pytest.mark.parametrize(("rows", "cols"), _TEST_DIMS)
def test_multiplication_crs(rows: int, cols: int) -> None:
    """Check that matrix vector multiplication works correctly."""
    rng = np.random.default_rng((rows * cols & rows) << max(cols, rows))

    for _ in range(max(rows, cols)):
        m1 = random_matrix(rng, rows, cols)
        m2 = random_matrix(rng, cols, rows)
        ms1 = m1 @ m2
        ms2 = m2 @ m1
        md1 = m1.toarray() @ m2.toarray()
        md2 = m2.toarray() @ m1.toarray()
        assert pytest.approx(ms1.toarray()) == md1
        assert pytest.approx(ms2.toarray()) == md2


@pytest.mark.parametrize(("rows", "cols"), _TEST_DIMS)
def test_transpose(rows: int, cols: int) -> None:
    """Check that matrix vector transpose correctly."""
    rng = np.random.default_rng((rows * cols & rows) << max(cols, rows))

    for _ in range(max(rows, cols)):
        m = random_matrix(rng, rows, cols)
        mt = m.transpose()
        assert pytest.approx(m.toarray().T) == mt.toarray()
