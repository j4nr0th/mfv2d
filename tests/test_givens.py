"""Check that Givens rotations work as expected."""

import numpy as np
import pytest
from interplib._mimetic import GivensRotation, SparseVector


def test_givens_manual_1() -> None:
    """Manually apply the rotation and check that it behaves as expected."""
    np.random.seed(0)

    mat = np.random.random_sample((5, 4))
    c = np.cos(0.1)
    s = np.sin(0.1)
    i1 = 2
    i2 = 3

    g = GivensRotation(mat.shape[0], i1, i2, c, s)

    manual = np.array(mat)  # copy
    manual[i1, :] = c * mat[i1, :] + s * mat[i2, :]
    manual[i2, :] = -s * mat[i1, :] + c * mat[i2, :]
    rotated = g @ mat
    assert pytest.approx(manual) == rotated


def test_givens_manual_2() -> None:
    """Manually apply the rotation and check that it behaves as expected."""
    np.random.seed(0)

    mat = np.random.random_sample((5, 4))
    c = np.cos(0.1)
    s = np.sin(0.1)
    i1 = 3
    i2 = 2

    g = GivensRotation(mat.shape[0], i1, i2, c, s)

    manual = np.array(mat)  # copy
    manual[i1, :] = c * mat[i1, :] + s * mat[i2, :]
    manual[i2, :] = -s * mat[i1, :] + c * mat[i2, :]
    rotated = g @ mat
    assert pytest.approx(manual) == rotated


def test_givens_sv_1() -> None:
    """Check that applying Givens to SparseVectors works."""
    g = GivensRotation(5, 1, 3, np.cos(0.3), np.sin(0.3))
    s = SparseVector.from_entries(5, (0, 2, 4), (1.0, 2.0, 3.0))
    assert np.all(g @ np.array(s) == np.array(g @ s))
    assert np.all(np.array(s) == np.array(g @ s))


def test_givens_sv_2() -> None:
    """Check that applying Givens to SparseVectors works."""
    g = GivensRotation(5, 1, 3, np.cos(0.3), np.sin(0.3))
    s = SparseVector.from_entries(5, (0, 1, 4), (1.0, 2.0, 3.0))
    assert np.all(g @ np.array(s) == np.array(g @ s))


def test_givens_sv_3() -> None:
    """Check that applying Givens to SparseVectors works."""
    g = GivensRotation(5, 1, 3, np.cos(0.3), np.sin(0.3))
    s = SparseVector.from_entries(5, (0, 3, 4), (1.0, 2.0, 3.0))
    assert np.all(g @ np.array(s) == np.array(g @ s))


def test_givens_sv_4() -> None:
    """Check that applying Givens to SparseVectors works."""
    g = GivensRotation(5, 3, 1, np.cos(0.3), np.sin(0.3))
    s = SparseVector.from_entries(5, (0, 1, 4), (1.0, 2.0, 3.0))
    assert np.all(g @ np.array(s) == np.array(g @ s))


def test_givens_sv_5() -> None:
    """Check that applying Givens to SparseVectors works."""
    g = GivensRotation(5, 3, 1, np.cos(0.3), np.sin(0.3))
    s = SparseVector.from_entries(5, (0, 3, 4), (1.0, 2.0, 3.0))
    assert np.all(g @ np.array(s) == np.array(g @ s))
