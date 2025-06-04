"""Stub for the C implemented types and functions related to mimetics."""

from collections.abc import Sequence
from typing import Self, final, overload

import numpy as np
import numpy.typing as npt

from mfv2d.eval import MatOpCode, _CompiledCodeMatrix

def lagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials, defined by

    .. math::

       \mathcal{L}^n_i (x) = \prod\limits_{j=1, j \neq i}^{n} \frac{x - x_j}{x_i - x_j},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_1, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial values at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate Lagrange
    polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import lagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = lagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is lagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = lagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def dlagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate derivatives of Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials derivatives, defined by

    .. math::

       \frac{d \mathcal{L}^n_i (x)}{d x} =
       \sum\limits_{j=0,j \neq i}^n \prod\limits_{k=0, k \neq i, k \neq j}^{n}
       \frac{1}{x_i - x_j} \cdot \frac{x - x_k}{x_i - x_k},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_0, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the derivatives of polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial derivatives at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate derivatives of
    Lagrange polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import dlagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = dlagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is dlagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = dlagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def compute_gll(
    order: int, max_iter: int = 10, tol: float = 1e-15
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre-Lobatto integration nodes and weights.

    If you are often re-using these, consider caching them.

    Parameters
    ----------
    order : int
       Order of the scheme. The number of node-weight pairs is one more.
    max_iter : int, default: 10
       Maximum number of iterations used to further refine the values.
    tol : float, default: 1e-15
       Tolerance for stopping the refinement of the nodes.

    Returns
    -------
    array
       Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.
    array
       Array of integration weights which correspond to the nodes.
    """
    ...
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
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]

def compute_element_matrices(
    form_orders: Sequence[int],
    expressions: _CompiledCodeMatrix,
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
    vector_fields: tuple[npt.NDArray[np.float64], ...],
    element_field_offsets: npt.NDArray[np.uint64],
    serialized_caches: Sequence[_SerializedBasisCache],
    thread_stack_size: int = (1 << 24),
) -> tuple[npt.NDArray[np.float64]]:
    """Compute element matrices based on the given instructions with tail calls.

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

    vector_fields : tuple of arrays
        Tuple of vector field values used for interior products. These are compuated at
        integration nodes for each element, then flattened, and packed in a tuple. The
        ordering of the fields in the tuple must match that of the system instructions
        were generated from.

    element_field_offsets : array
        Array of offsets that indicates where the vector fields for each element begins.
        It should contain one more entry than the element count.

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

def compute_element_explicit(
    dofs: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.uint32],
    form_orders: Sequence[int],
    expressions: Sequence[Sequence[Sequence[MatOpCode | int | float] | None]],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
    vector_fields: tuple[npt.NDArray[np.float64], ...],
    element_field_offsets: npt.NDArray[np.uint64],
    serialized_caches: Sequence[_SerializedBasisCache],
    thread_stack_size: int = (1 << 24),
) -> tuple[npt.NDArray[np.float64]]:
    """Compute element equations based on degrees of freedom given.

    Parameters
    ----------
    dofs : array
        Array containing degrees of freedom for all elements.

    offsets : (N,) array
        Array of offsets into the ``dofs`` array for each element.

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

    vector_fields : tuple of arrays
        Tuple of vector field values used for interior products. These are compuated at
        integration nodes for each element, then flattened, and packed in a tuple. The
        ordering of the fields in the tuple must match that of the system instructions
        were generated from.

    element_field_offsets : array
        Array of offsets that indicates where the vector fields for each element begins.
        It should contain one more entry than the element count.

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

class SparseVector:
    """Vector which stores only the non-zero components."""

    @classmethod
    def from_entries(cls, n: int, indices: npt.ArrayLike, values: npt.ArrayLike) -> Self:
        """Create sparse vector from an array of indices and values.

        Parameters
        ----------
        n : int
            Dimension of the vector.

        indices : array_like
            Indices of the entries. Must be sorted.

        values : array_like
            Values of the entries.

        Returns
        -------
        SparseVector
            New vector with indices and values as given.
        """
        ...

    @property
    def n(self) -> int:
        """Dimension of the vector."""
        ...

    @n.setter
    def n(self, v: int, /) -> None: ...
    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Values of non-zero entries of the vector."""
        ...

    @property
    def indices(self) -> npt.NDArray[np.uint64]:
        """Indices of non-zero entries of the vector."""
        ...

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        """Convert the vector to a full array."""
        ...

    @overload
    def __getitem__(self, idx: int, /) -> float: ...
    @overload
    def __getitem__(self, idx: slice, /) -> SparseVector: ...
    @classmethod
    def concatenate(cls, *vectors: SparseVector) -> Self:
        """Merge sparse vectors together into a single vector.

        Parameters
        ----------
        *vectors : SparseVector
            Sparse vectors that should be concatenated.

        Returns
        -------
        Self
            Newly created sparse vector.
        """
        ...

    @property
    def count(self) -> int:
        """Number of entries in the vector."""
        ...

class GivensRotation:
    """Representation of a Givens rotation matrix."""

    def __new__(cls, n: int, i1: int, i2: int, c: float, s: float) -> Self: ...
    @property
    def n(self) -> int:
        """Dimension of the rotation."""
        ...

    @property
    def i1(self) -> int:
        """First index of rotation."""
        ...

    @property
    def i2(self) -> int:
        """Second index of rotation."""
        ...

    @property
    def c(self) -> float:
        """Cosine rotation value."""
        ...

    @property
    def s(self) -> float:
        """Sine rotation value."""
        ...

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        """Convert the object into a full numpy matrix."""
        ...

    @overload
    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    @overload
    def __matmul__(self, other: SparseVector) -> SparseVector: ...
    @property
    def T(self) -> GivensRotation:
        """Inverse rotation."""
        ...

class GivensSeries:
    """Series of GivensRotations."""

    @overload
    def __new__(cls, n: int, /) -> Self: ...
    @overload
    def __new__(cls, *rotations: GivensRotation) -> Self: ...
    @property
    def n(self) -> int:
        """Size of the rotations."""
        ...

    def __len__(self) -> int:
        """Return number of Givens rotations."""
        ...

    # @overload
    def __getitem__(self, idx: int, /) -> GivensRotation: ...
    # @overload
    # def __getitem__(self, idx: slice, /) -> GivensSeries: ...
    @overload
    def __matmul__(self, other: SparseVector) -> SparseVector: ...
    @overload
    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def apply(self, v: npt.NDArray[np.float64], /) -> None:
        """Apply in-place as fast as possible.

        This function works as fast as possible, without any allocations.

        Parameters
        ----------
        v : array
            One dimensional array to rotate.
        """
        ...

class LiLMatrix:
    """Matrix which has a list of is used to store sparse rows.

    Parameters
    ----------
    rows : int
        Number of rows of the matrix.
    cols : int
        Number of columns of the matrix.
    """

    def __new__(cls, rows: int, cols: int) -> Self: ...
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the matrix."""
        ...

    def __getitem__(self, idx: int, /) -> SparseVector:
        """Get the row of the matrix."""
        ...

    def __setitem__(self, idx: int, val: SparseVector, /) -> None:
        """Set the row of the matrix."""
        ...

    def count_entries(self) -> int:
        """Return the number of entries in the matrix."""
        ...

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        """Convert the matrix into a numpy array."""
        ...

    @classmethod
    def from_full(cls, mat: npt.ArrayLike, /) -> Self:
        """Create A LiLMatrix from a full matrix.

        Parameters
        ----------
        mat : array
            Full matrix to convert.

        Returns
        -------
        LiLMatrix
            Matrix represented in the LiLMatrix format.ž
        """
        ...

    def qr_decompose(self, n: int | None = None, /) -> GivensSeries:
        """Decompose the matrix into a series of Givens rotations and a triangular matrix.

        Parameters
        ----------
        n : int, optional
            Maximum number of steps to perform.

        Returns
        -------
        (GivensRotation, ...)
            Givens rotations in the order they were applied to the matrix.
            This means that for the solution, they should be applied in the
            reversed order.
        """
        ...

    @classmethod
    def block_diag(cls, *blocks: LiLMatrix) -> Self:
        """Construct a new matrix from blocks along the diagonal.

        Parameters
        ----------
        *blocks : LiLMatrix
            Block matrices. These are placed on the diagonal of the resulting matrix.

        Returns
        -------
        LiLMatrix
            Block diagonal matrix resulting from the blocks.
        """
        ...

    def add_columns(self, *cols: SparseVector) -> None:
        """Add columns to the matrix.

        Parameters
        ----------
        *cols : SparseVectors
            Columns to be added to the matrix.
        """
        ...

    def add_rows(self, *rows: SparseVector) -> LiLMatrix:
        """Create a new matrix with added rows.

        Parameters
        ----------
        *rows : SparaseVector
            Rows to be added.

        Returns
        -------
        LiLMatrix
            Matrix with new rows added.
        """
        ...

    def solve_upper_triangular(
        self, rsh: npt.ArrayLike, /, out: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """Use back-substitution to solve find the right side.

        This assumes the matrix is upper triangualr.

        Parameters
        ----------
        rhs : array_like
            Vector or matrix that gives the right side of the equation.

        out : array, optional
            Array to be used as output. If not given a new one will be created and
            returned, otherwise, the given value is returned. It must match the shape
            of the input array exactly and have the correct data type.

        Returns
        -------
        array
            Vector or matrix that yields the rhs when matrix multiplication is used.
            If the ``out`` parameter is given, the value returned will be exactly that
            matrix.
        """
        ...

    @classmethod
    def empty_diagonal(cls, n: int, /) -> Self:
        """Create empty square matrix with zeros on the diagonal.

        This is intended for padding that allows for computing QR decompositions.

        Parameters
        ----------
        n : int
            Size of the square matrix.

        Returns
        -------
        LiLMatrix
            Sparse matrix that is square and has only zeros on its diagonal.
        """
        ...

    @property
    def usage(self) -> int:
        """Number of non-zero entries."""
        ...

    def to_scipy(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]]:
        """Convert itself into an array of values and an array of coorinates.

        Returns
        -------
        (N,) array of floats
            Values of the entries stored in the matrx.

        (N, 2) array of uint64
            Positions of entries as ``(row, col)``.
        """
        ...

def compute_element_matrix(
    form_orders: Sequence[int],
    expressions: _CompiledCodeMatrix,
    corners: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    vector_fields: Sequence[npt.NDArray[np.float64]],
    basis_1_nodal: npt.NDArray[np.float64],
    basis_1_edge: npt.NDArray[np.float64],
    weights_1: npt.NDArray[np.float64],
    nodes_1: npt.NDArray[np.float64],
    basis_2_nodal: npt.NDArray[np.float64],
    basis_2_edge: npt.NDArray[np.float64],
    weights_2: npt.NDArray[np.float64],
    nodes_2: npt.NDArray[np.float64],
    stack_memory: np.ndarray,
) -> npt.NDArray[np.float64]:
    """Compute a single element matrix."""
    ...

def compute_element_matrix_test(
    corners: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    basis_1_nodal: npt.NDArray[np.float64],
    basis_1_edge: npt.NDArray[np.float64],
    weights_1: npt.NDArray[np.float64],
    nodes_1: npt.NDArray[np.float64],
    basis_2_nodal: npt.NDArray[np.float64],
    basis_2_edge: npt.NDArray[np.float64],
    weights_2: npt.NDArray[np.float64],
    nodes_2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Test computing element matrices for a single element."""
    ...
