"""Stub for the C implemented types and functions related to mimetics."""

class GeoID:
    """A way to identify a geometrical object in a mesh.

    Parameters
    ----------
    index : int
        Index of the geometrical object.
    reversed : bool
        The object's orientation should be reversed.
    """

    index: int
    reversed: bool

class Line:
    """A geometrical object, which connects two points.

    Parameters
    ----------
    begin : GeoID
        ID of the point where the line beings.
    end : GeoID
        ID of the point where the line ends.
    """

    begin: GeoID
    end: GeoID

class Surface:
    """A geometrical object, which is bound by four lines.

    Parameters
    ----------
    bottom : GeoID
        Bottom boundary of the surface.
    right : GeoID
        Right boundary of the surface.
    top : GeoID
        Top boundary of the surface.
    left : GeoID
        Left boundary of the surface.
    """

    bottom: GeoID
    right: GeoID
    top: GeoID
    left: GeoID

class Volume:
    """A geometrical object, which is bound by six surfaces.

    left : GeoID
        Left boundary surface of the surface.
    back : GeoID
        Back boundary surface of the surface.
    bottom : GeoID
        Bottom boundary surface of the surface.
    right : GeoID
        Right boundary surface of the surface.
    front : GeoID
        Front boundary surface of the surface.
    top : GeoID
        Top boundary surface of the surface.
    """

    left: GeoID
    back: GeoID
    bottom: GeoID
    right: GeoID
    front: GeoID
    top: GeoID

class Manifold:
    """A manifold of a finite number of dimensions."""

    dimension: int
    ...

class Manifold1D(Manifold):
    """A 1D manifold."""

    dimension: int = 1
    n_lines: int
    n_points: int

    def get_line(self, index: GeoID, /) -> Line:
        """Get the line of the specified ID."""
        ...

    def find_line(self, line: Line) -> GeoID:
        """Find the ID of the specified line."""
        ...

    ...
