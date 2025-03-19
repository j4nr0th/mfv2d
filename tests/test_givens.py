"""Check that Givens rotations work as expected."""

import numpy as np
import pytest
from interplib._mimetic import GivensRotation


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
