"""Test that the 1D mesh and manifold works as would be expected."""

import numpy as np
from interplib import Mesh1D
from interplib._mimetic import GeoID, Manifold, Manifold1D


def test_manifold1d():
    """Check that Manifold works as expected."""
    caught = False
    try:
        _ = Manifold()
    except TypeError:
        caught = True
    assert caught, "How did you construct the object?"

    caught = False
    try:
        _ = Manifold1D()
    except TypeError:
        caught = True
    assert caught, "How did you construct the object?"

    n = 10
    real_one = Manifold1D.line_mesh(n)
    assert real_one.n_lines == n
    assert real_one.n_points == n + 1
    for i in range(n):
        ln = real_one.get_line(i + 1)
        i_ln = real_one.find_line(ln)
        assert GeoID(i, 0) == i_ln
        ln2 = real_one.get_line(-(i + 1))
        assert ln.begin == ln2.end
        assert ln.end == ln2.begin
        assert ln.begin == GeoID(i, 0)
        assert ln.end == GeoID(i + 1, 0)

    caught = False
    try:
        _ = real_one.get_line(n + 1)
    except IndexError:
        caught = True
    assert caught, "How did you get a line that far off?"


def test_mesh1d():
    """Check that the mesh works correctly."""
    n = 30
    pos = np.linspace(-1, +1, n)

    caught = False
    try:
        _ = Mesh1D(pos, np.random.randint(1, 5, pos.size))
    except ValueError:
        caught = True
    assert caught

    caught = False
    try:
        _ = Mesh1D(pos, np.random.randint(1, 5, pos.size - 2))
    except ValueError:
        caught = True
    assert caught

    orders = np.random.randint(1, 5, pos.size - 1)
    msh = Mesh1D(pos, orders)

    for i in range(n - 1):
        elem = msh.get_element(i)
        assert elem.order == orders[i]
        assert elem.xleft == pos[i]
        assert elem.xright == pos[i + 1]
