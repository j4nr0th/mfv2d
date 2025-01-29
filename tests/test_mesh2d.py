"""Check that 2D mesh stuff works."""

import numpy as np
import pytest
from interplib.mimetic.mimetic2d import Element2D


def check_e01():
    """Check that incidence matrix works.

    For this, I manually drew the 3rd order mesh on paper, gave each node
    a random value, then computed what the values should be.
    """
    x = np.array(
        (
            (2, 3, 7, 8),
            (4, -2, 3, 2),
            (-1, 1, 1, -1),
            (5, 7, 4, 0),
        )
    ).reshape((-1,))
    y1 = np.array(
        (
            (-1, -4, -1),
            (6, -5, 1),
            (-2, 0, +2),
            (-2, 3, 4),
        )
    ).reshape((-1,))
    y2 = np.array(
        (
            (2, -5, -4, -6),
            (-5, 3, -2, -3),
            (6, 6, 3, 1),
        )
    ).reshape((-1,))
    y = np.concatenate((y1, y2))

    e = Element2D(3, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    e10 = e.incidence_01()

    y_e = e10 @ x
    assert pytest.approx(y_e) == y


def check_e12():
    """Check that incidence matrix works.

    For this, I manually drew the 3rd order mesh on paper, gave each line
    a value, then checked it's correct.
    """
    x1 = np.array(
        (
            (1, 2, -3),
            (0, 0, 1),
            (3, -1, 4),
            (1, 2, 3),
        )
    ).reshape((-1,))
    x2 = np.array(
        (
            (3, 4, -2, -4),
            (2, 4, 5, 1),
            (3, -1, 2, 5),
        )
    ).reshape((-1,))
    y = np.array(
        (
            (0, 8, -2),
            (-5, 0, 1),
            (6, -6, -2),
        )
    ).reshape((-1,))
    x = np.concatenate((x1, x2))

    e = Element2D(3, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    e21 = e.incidence_12()

    y_e = e21 @ x
    assert pytest.approx(y_e) == y


def check_e10_e21_null():
    """Product of E21 @ E10 should be zero."""
    e = Element2D(10, (-1, -1), (+1, -1), (+1, +1), (-1, +1))
    e10 = e.incidence_01()
    e21 = e.incidence_12()
    assert np.all(e21 @ e10 == 0)
