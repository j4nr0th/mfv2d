"""Stub for the C implemented types and functions related to mimetics."""

from typing import Self, final

@final
class GeoID:
    """Type used to identify a geometrical object with an index and orientation.

    Parameters
    ----------
    index : int
        Index of the geometrical object.
    reversed : any, default: False
        The object's orientation should be reversed.
    """

    def __new__(cls, index: int, reverse: object = False) -> Self: ...
    @property
    def index(self) -> int:
        """Index of the object referenced by id."""
        ...
    @property
    def reversed(self) -> bool:
        """Is the orientation of the object reversed."""
        ...

    def __bool__(self) -> bool: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __neg__(self) -> GeoID: ...

@final
class Line:
    """Geometrical object, which connects two points.

    Parameters
    ----------
    begin : GeoID or int
        ID of the point where the line beings.
    end : GeoID or int
        ID of the point where the line ends.
    """

    def __new__(cls, begin: GeoID | int, end: GeoID | int) -> Self: ...
    @property
    def begin(self) -> GeoID:
        """ID of the point where the line beings."""
        ...
    @property
    def end(self) -> GeoID:
        """ID of the point where the line ends."""
        ...

    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Surface:
    """Geometrical object, which is bound by four lines.

    Parameters
    ----------
    bottom : GeoID or int
        Bottom boundary of the surface.
    right : GeoID or int
        Right boundary of the surface.
    top : GeoID or int
        Top boundary of the surface.
    left : GeoID or int
        Left boundary of the surface.
    """

    def __new__(
        cls, bottom: GeoID | int, right: GeoID | int, top: GeoID | int, left: GeoID | int
    ) -> Self: ...
    @property
    def bottom(self) -> GeoID:
        """Bottom boundary of the surface."""
        ...
    @property
    def right(self) -> GeoID:
        """Right boundary of the surface."""
        ...
    @property
    def top(self) -> GeoID:
        """Top boundary of the surface."""
        ...
    @property
    def left(self) -> GeoID:
        """Left boundary of the surface."""
        ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# class Volume:
#     """A geometrical object, which is bound by six surfaces.
#
#     left : GeoID
#         Left boundary surface of the surface.
#     back : GeoID
#         Back boundary surface of the surface.
#     bottom : GeoID
#         Bottom boundary surface of the surface.
#     right : GeoID
#         Right boundary surface of the surface.
#     front : GeoID
#         Front boundary surface of the surface.
#     top : GeoID
#         Top boundary surface of the surface.
#     """
#
#     left: GeoID
#     back: GeoID
#     bottom: GeoID
#     right: GeoID
#     front: GeoID
#     top: GeoID

class Manifold:
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

@final
class Manifold1D(Manifold):
    """One dimensional manifold."""

    @property
    def n_lines(self) -> int:
        """Number of lines in the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the manifold."""
        ...

    def get_line(self, index: GeoID | int, /) -> Line:
        """Get the line of the specified ID."""
        ...

    def find_line(self, line: Line) -> GeoID:
        """Find the ID of the specified line."""
        ...

    @classmethod
    def line_mesh(cls, segments: int, /) -> Manifold1D:
        """Create a new Manifold1D which represents a line.

        Parameters
        ----------
        segments : int
            Number of segments the line is split into. There will be one more point.

        Returns
        -------
        Manifold1D
            Manifold that represents the topology of the line.
        """
        ...

    def compute_dual(self) -> Manifold1D:
        """Compute the dual to the manifold.

        Returns
        -------
        Manifold1D
            The dual to the manifold.
        """
        ...
