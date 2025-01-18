"""Test basic mimetic geometry elements."""

from interplib._mimetic import GeoID, Line, Surface


def test_geo_id():
    """Check that GeoID methods, getters, and setters work."""
    caught = False

    try:
        _ = GeoID(-1)
    except ValueError:
        caught = True

    assert caught, "Construction with index negative value should not work."
    assert GeoID(3) != GeoID(3, True)
    assert GeoID(3) != GeoID(3, 1)
    val = GeoID(5, True)
    assert val.index == 5
    assert val.reversed
    val2 = GeoID(4)
    assert val2.index == 4
    assert not val2.reversed


def test_line():
    """Check that Line methods, getters, and setters work."""
    # Invalid IDs when 0 is used as index
    ln = Line(0, 0)
    assert not ln.begin
    assert not ln.end
    assert Line(1, 3) == Line(1, 3)
    assert Line(1, 3) != Line(1, 2)
    ln1 = Line(3, -3)
    assert ln1.begin == -ln1.end
    id1 = GeoID(0, False)
    id2 = GeoID(2, False)
    assert Line(id1, id2) == Line(id1, id2)
    assert Line(id1, id2) != Line(id2, id2)
    assert Line(id1, id2) != Line(id2, id1)


def test_surface():
    """Check that surface methods, getters, and setters work."""
    # Invalid IDs when 0 is used as index
    s = Surface(0, 0, 0, 0)

    assert not s.left
    assert not s.right
    assert not s.top
    assert not s.bottom

    assert Surface(1, 3, 4, 5) == Surface(1, 3, 4, 5)
    assert Surface(1, 3, 2, 1) != Surface(1, 2, 6, 4)
    ln1 = Surface(3, -2, -3, +2)
    assert ln1.bottom == -ln1.top
    assert ln1.right == -ln1.left
    id1 = GeoID(0, False)
    id2 = GeoID(2, False)
    id3 = GeoID(4, False)
    id4 = GeoID(6, False)
    assert Surface(id1, id2, id3, id4) == Surface(id1, id2, id3, id4)
    assert Surface(id1, id2, id3, id4) != Surface(id2, id2, id2, id2)
    assert Surface(id1, id2, id3, id4) != Surface(id2, id1, id3, id4)
