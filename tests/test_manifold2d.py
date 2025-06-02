"""Check that the 2D manifold works as intended."""

from mfv2d._mfv2d import Line, Manifold2D, Surface


def test_simple():
    """Simplest check.

    Creates a manifold with the following topology:

    ::

        Point arrangement:
        1------4--5
        |      | /
        |      |/
        2------3

        Line arrangement:
        x<-----x<---x
        |  4   ^ 5 /
        |1    3|  /6
        v  2   | /
        x----->x

    The manifold is then checked to map exactly back to what it is initially.
    """
    lines = [
        Line(1, 2),
        Line(2, 3),
        Line(3, 4),
        Line(4, 1),
        Line(5, 4),
        Line(3, 5),
    ]
    surfaces = [
        Surface(1, 2, 3, 4),
        Surface(-3, 6, 5),
    ]
    m = Manifold2D.from_irregular(5, lines, surfaces)

    for i in range(m.n_surfaces):
        s1 = m.get_surface(i + 1)
        s2 = m.get_surface(-(i + 1))
        assert len(s1) == len(s2)
        for i in range(len(s1)):
            assert s1[i] == -s2[i]

    for i, s in enumerate(surfaces):
        assert s == m.get_surface(i + 1)

    for i, ln in enumerate(lines):
        line = m.get_line(i + 1)
        assert ln == line
