"""Stub for the C implemented types and functions related to mimetics."""

from collections.abc import Sequence
from typing import Self, final

import numpy as np
import numpy.typing as npt

from interplib.kforms.eval import MatOpCode

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

    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Surface:
    """Two dimensional geometrical object, which is bound by lines.

    Parameters
    ----------
    *ids : GeoID or int
        Ids of the lines which are the boundary of the surface.
    """

    def __new__(cls, *ids: GeoID | int) -> Self: ...
    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __getitem__(self, idx: int) -> GeoID: ...
    def __len__(self) -> int: ...
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

class Manifold2D(Manifold):
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        ...

    @property
    def n_lines(self) -> int:
        """Number of lines in the mesh."""
        ...

    @property
    def n_surfaces(self) -> int:
        """Number of surfaces in the mesh."""
        ...

    def get_line(self, index: int | GeoID, /) -> Line:
        """Get the line from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the line to get in 1-based indexing or GeoID. If negative, the
           orientation will be reversed.

        Returns
        -------
        Line
           Line object corresponding to the ID.
        """
        ...

    def get_surface(self, index: int | GeoID, /) -> Surface:
        """Get the surface from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the surface to get in 1-based indexing or GeoID. If negative,
           the orientation will be reversed.

        Returns
        -------
        Surface
           Surface object corresponding to the ID.
        """

    @classmethod
    def from_irregular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with non-constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : Sequence of array_like
            Sequence of arrays specifying connectivity of mesh surfaces in 1-based
            indexing, where a negative value means that the line's orientation is
            reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    @classmethod
    def from_regular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : array_like
            Two dimensional array-like object specifying connectivity of mesh
            surfaces in 1-based indexing, where a negative value means that
            the line's orientation is reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    def compute_dual(self) -> Manifold2D:
        """Compute the dual to the manifold.

        A dual of each k-dimensional object in an n-dimensional space is a
        (n-k)-dimensional object. This means that duals of surfaces are points,
        duals of lines are also lines, and that the duals of points are surfaces.

        A dual line connects the dual points which correspond to surfaces which
        the line was a part of. Since the change over a line is computed by
        subtracting the value at the beginning from that at the end, the dual point
        which corresponds to the primal surface where the primal line has a
        positive orientation is the end point of the dual line and conversely the end
        dual point is the one corresponding to the surface which contained the primal
        line in the negative orientation. Since lines may only be contained in a
        single primal surface, they may have an invalid ID as either their beginning or
        their end. This can be used to determine if the line is actually a boundary of
        the manifold.

        A dual surface corresponds to a point and contains dual lines which correspond
        to primal lines, which contained the primal point of which the dual surface is
        the result of. The orientation of dual lines in this dual surface is positive if
        the primal line of which they are duals originated in the primal point in question
        and negative if it was their end point.

        Returns
        -------
        Manifold2D
            Dual manifold.
        """
        ...

    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

_SerializedBasisCache = tuple[
    int,
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]

def compute_element_matrices(
    form_orders: Sequence[int],
    expressions: Sequence[Sequence[Sequence[MatOpCode | int | float] | None]],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
    serialized_caches: Sequence[_SerializedBasisCache],
    thread_stack_size: int = (1 << 24),
) -> tuple[npt.NDArray[np.float64]]:
    """Compuate element matrices based on the given instructions.

    Parameters
    ----------
    form_orders : Sequence of int
        Orders of the unknown differential forms.

    expressions : 2D matrix of (Sequence of (MatOpCode, int, and float) or None)
        Two dimensional matrix of instructions to compute the entry of the element matrix.
        It can be left as None, which means there is no contribution.

    pos_bl : (N, 2) array
        Array of position vectors for the bottom left corners of elements.

    pos_br : (N, 2) array
        Array of position vectors for the bottom right corners of elements.

    pos_tr : (N, 2) array
        Array of position vectors for the top right corners of elements.

    pos_tl : (N, 2) array
        Array of position vectors for the top left corners of elements.

    element_orders : (N,) array
        Array of orders of the elements. There must be an entry for this in
        the ``serialized_caches``.

    serialized_caches : Sequence of _SerializedBasisCache
        All the serialized caches to use for the elements. Only one is allowed
        per element order.

    thread_stack_size : int, default: 2 ** 24
        Default amount of memory allocated to each worker thread for the element they're
        working on.

    Returns
    -------
    tuple of N arrays
        Tuple of element matices.
    """
    ...

def compute_element_matrices_2(
    form_orders: Sequence[int],
    expressions: Sequence[Sequence[Sequence[MatOpCode | int | float] | None]],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
    serialized_caches: Sequence[_SerializedBasisCache],
    thread_stack_size: int = (1 << 24),
) -> tuple[npt.NDArray[np.float64]]:
    """Compuate element matrices based on the given instructions with tail calls.

    Parameters
    ----------
    form_orders : Sequence of int
        Orders of the unknown differential forms.

    expressions : 2D matrix of (Sequence of (MatOpCode, int, and float) or None)
        Two dimensional matrix of instructions to compute the entry of the element matrix.
        It can be left as None, which means there is no contribution.

    pos_bl : (N, 2) array
        Array of position vectors for the bottom left corners of elements.

    pos_br : (N, 2) array
        Array of position vectors for the bottom right corners of elements.

    pos_tr : (N, 2) array
        Array of position vectors for the top right corners of elements.

    pos_tl : (N, 2) array
        Array of position vectors for the top left corners of elements.

    element_orders : (N,) array
        Array of orders of the elements. There must be an entry for this in
        the ``serialized_caches``.

    serialized_caches : Sequence of _SerializedBasisCache
        All the serialized caches to use for the elements. Only one is allowed
        per element order.

    thread_stack_size : int, default: 2 ** 24
        Default amount of memory allocated to each worker thread for the element they're
        working on.

    Returns
    -------
    tuple of N arrays
        Tuple of element matices.
    """
    ...

def element_matrices(
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
    serialized_cache: _SerializedBasisCache,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute the element matrices."""
    ...

def check_bytecode(expression: list[MatOpCode | int | float], /) -> list[int | float]:
    """Convert bytecode to C-values, then back to Python.

    This function is meant for testing.
    """
    ...

def check_incidence(
    x: npt.ArrayLike, /, order: int, in_form: int, transpose: bool, right: bool
) -> npt.NDArray[np.float64]:
    """Apply the incidence matrix to the input matrix.

    This function is meant for testing.
    """
    ...

def continuity(
    primal: Manifold2D,
    dual: Manifold2D,
    form_orders: npt.ArrayLike,
    element_offsets: npt.ArrayLike,
    dof_offsets: npt.ArrayLike,
    element_orders: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Create continuity equation for different forms."""
    ...
