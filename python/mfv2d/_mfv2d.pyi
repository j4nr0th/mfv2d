"""Stub for the C implemented types and functions.

This file contains functions signatures and *copies* of docstrings for the
C-extension which implements all the required fast code.
"""

from collections.abc import Sequence
from typing import Self, final, overload

import numpy as np
import numpy.typing as npt

from mfv2d.eval import MatOpCode, _CompiledCodeMatrix
from mfv2d.kform import UnknownFormOrder

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

        >>> from mfv2d._mfv2d import lagrange1d
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

        >>> from mfv2d._mfv2d import dlagrange1d
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
    r"""Compute Gauss-Legendre-Lobatto integration nodes and weights.

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

    Examples
    --------
    Gauss-Legendre-Lobatto nodes computed using this function, along with
    the weights.

    .. jupyter-execute::

        >>> import numpy as np
        >>> from mfv2d._mfv2d import compute_gll
        >>> from matplotlib import pyplot as plt
        >>>
        >>> n = 5
        >>> nodes, weights = compute_gll(n)
        >>>
        >>> # Plot these
        >>> plt.figure()
        >>> plt.scatter(nodes, weights)
        >>> plt.xlabel("$\\xi$")
        >>> plt.ylabel("$w$")
        >>> plt.grid()
        >>> plt.show()

    Since these are computed in an iterative way, giving a tolerance
    which is too strict or not allowing for sufficient iterations
    might cause an exception to be raised to do failiure to converge.
    """
    ...

def compute_legendre(
    order: int, positions: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    r"""Compute Legendre polynomials at given nodes.

    Parameters
    ----------
    order : int
        Order of the scheme. The number of node-weight pairs is one more.

    positions : array_like
        Positions where the polynomials should be evaluated at.

    out : array, optional
        Output array to write to. If not specified, then a new array is allocated.
        Must have the exact correct shape (see return value) and data type
        (double/float64).

    Returns
    -------
    array
        Array with the same shape as ``positions`` parameter, except with an
        additional first dimension, which determines which Legendre polynomial
        it is.

    Examples
    --------
    To quickly illustrate how this function can be used to work with Legendre polynomials,
    some simple examples are shown.

    First things first, the function can be called for any order of polynomials, with
    about any shape of input array (though if you put too many dimensions you will get an
    exception). Also, you can supply an optional output parameter, such that an output
    array need not be newly allocated.

    .. jupyter-execute::

        >>> import numpy as np
        >>> from mfv2d._mfv2d import compute_legendre
        >>>
        >>> n = 5
        >>> positions = np.linspace(-1, +1, 101)
        >>> vals = compute_legendre(n, positions)
        >>> assert vals is compute_legendre(n, positions, vals)

    The output array will always have the same shape as the input array, with the only
    difference being that a new axis is added for the first dimension, which can be
    indexed to distinguish between the different Legendre polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> fig, ax = plt.subplots(1, 1)
        >>>
        >>> for i in range(n + 1):
        >>>     ax.plot(positions, vals[i, ...], label=f"$y = \\mathcal{{L}}_{{{i:d}}}$")
        >>>
        >>> ax.set(xlabel="$x$", ylabel="$y$")
        >>> ax.grid()
        >>> ax.legend()
        >>>
        >>> fig.tight_layout()
        >>> plt.show()

    Lastly, these polynomials are all orthogonal under the :math:`L^2` norm. This can
    be shown numerically as well.

    .. jupyter-execute::

        >>> from mfv2d._mfv2d import IntegrationRule1D
        >>>
        >>> rule = IntegrationRule1D(n + 1)
        >>>
        >>> vals = compute_legendre(n, rule.nodes)
        >>>
        >>> for i1 in range(n + 1):
        >>>     p1 = vals[i1, ...]
        >>>     for i2 in range(n + 1):
        >>>         p2 = vals[i2, ...]
        >>>
        >>>         integral = np.sum(p1 * p2 * rule.weights)
        >>>
        >>>         if i1 != i2:
        >>>             assert abs(integral) < 1e-16
    """

def legendre_l2_to_h1_coefficients(c: npt.ArrayLike, /) -> npt.NDArray[np.double]:
    """Convert Legendre polynomial coefficients to H1 coefficients.

    The :math:`H^1` coefficients are based on being expansion coefficients of hierarchical
    basis which are orthogonal in the :math:`H^1` norm instead of in the :math:`L^2` norm,
    which holds for Legendre polynomials instead.

    Parameters
    ----------
    c : array_like
        Coefficients of the Legendre polynomials.

    Returns
    -------
    array
        Coefficients of integrated Legendre polynomial basis.
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

class Manifold2D(Manifold):
    """Two dimensional manifold consisting of surfaces made of lines.

    Has no constructor, but can be created from its class methods
    :meth:`Manifold2D.from_irregular` and :meth:`Manifold2D.from_regular`. This type
    provides convinience functions for computing its dual and obtaining specific lines or
    surfaces.

    Since manifold only contains topology, the only information it contains pertaining to
    nodes is the tota lnumber of them.

    Examples
    --------
    This is an example of how a manifold may be used:

    .. jupyter-execute::

        >>> import numpy as np
        >>> from mfv2d._mfv2d import Manifold2D, Surface, Line, GeoID
        >>>
        >>> triangle = Manifold2D.from_regular(
        ...     3,
        ...     [Line(1, 2), Line(2, 3), Line(1, 3)],
        ...     [Surface(1, 2, -3)],
        ... )
        >>> print(triangle)

    The previous case only had one surface. In that case, or if all surface
    have the same number of lines, the class method :meth:`Manifold2D.from_regular`
    can be used. If the surface do not have the same number of lines, that can not be
    used. Instead, the :meth:`Manifold2D.from_irregular` class method should be used.

    .. jupyter-execute::

        >>> house = Manifold2D.from_irregular(
        ...     5,
        ...     [
        ...         (1, 2), (2, 3), (3, 4), (4, 1), #Square
        ...         (1, 5), (5, 2), # Roof
        ...     ],
        ...     [
        ...         (1, 2, 3, 4), # Square
        ...         (-1, 5, 6),   # Triangle
        ...     ]
        ... )
        >>> print(house)

    From these manifolds, surfaces or edges can be querried back. This is mostly useful
    when the dual is also computed, which allows to obtain information about neighbouring
    objects. For example, if we want to know what points are neighbours of point with
    index 2, we would do the following:

    .. jupyter-execute::

        >>> dual = house.compute_dual() # Get the dual manifold
        >>> # Dual surface corresponding to primal point 1
        >>> dual_surface = dual.get_surface(1)
        >>> print(dual_surface)
        >>> for line_id in dual_surface:
        ...     if not line_id:
        ...         continue
        ...     primal_line = house.get_line(line_id)
        ...     if primal_line.begin == 1:
        ...         pt = primal_line.end
        ...     else:
        ...         assert primal_line.end == 1
        ...         pt = primal_line.begin
        ...     print(f"Point 1 neighbours point {pt}")

    """

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

    @classmethod
    def from_pairs(cls, n: int, *pairs: tuple[int, float]) -> Self:
        """Create sparse vector from an index-coefficient pairs.

        Parameters
        ----------
        n : int
            Dimension of the vector.

        *pairs : tuple[int, float]
            Pairs of values and indices for the vector.

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
@final
class IntegrationRule1D:
    """Type used to contain integration rule information.

    Parameters
    ----------
    order : int
        Order of integration rule used. Can not be negative.

    Attributes
    ----------
    nodes : array
        Position of integration nodes on the reference domain [-1, +1]
        where the integrated function should be evaluated.

    weights : array
        Weight values by which the values of evaluated function should be
        multiplied by.
    """

    def __new__(cls, order: int) -> Self: ...
    @property
    def order(self) -> int: ...
    @property
    def nodes(self) -> npt.NDArray[np.double]: ...
    @property
    def weights(self) -> npt.NDArray[np.double]: ...

@final
class Basis1D:
    """1D basis functions collection used for FEM space creation.

    Parameters
    ----------
    order : int
        Order of basis used.

    rule : IntegrationRule1D
        Integration rule for basis creation.
    """

    def __new__(cls, order: int, rule: IntegrationRule1D) -> Self: ...
    @property
    def order(self) -> int:
        """Order of the basis."""
    @property
    def node(self) -> npt.NDArray[np.double]:
        """Nodal basis values."""
        ...
    @property
    def edge(self) -> npt.NDArray[np.double]:
        """Edge basis values."""
        ...
    @property
    def rule(self) -> IntegrationRule1D: ...
    @property
    def roots(self) -> npt.NDArray[np.double]:
        """Roots of the nodal basis."""
        ...

@final
class Basis2D:
    """2D basis functions collection used for FEM space creation.

    Parameters
    ----------
    basis_xi : Basis1D
        Basis used in the Xi direction.

    basis_eta : Basis1D
        Basis used in the Eta direction.
    """

    def __new__(cls, basis_xi: Basis1D, basis_eta: Basis1D) -> Self: ...
    @property
    def basis_xi(self) -> Basis1D: ...
    @property
    def basis_eta(self) -> Basis1D: ...

@final
class ElementMassMatrixCache:
    """Caches element mass matrices."""

    def __new__(cls, basis: Basis2D, corners: npt.NDArray[np.float64]) -> Self: ...
    @property
    def basis_xi(self) -> Basis1D:
        """Get 1D basis functions for the first dimension."""
        ...

    @property
    def basis_eta(self) -> Basis1D:
        """Get 2D basis functions for the second dimension."""
        ...

    @property
    def basis_2d(self) -> Basis2D:
        """Get 2D basis functions."""
        ...

    @property
    def corners(self) -> npt.NDArray[np.float64]:
        """Get the element corners as a (4, 2) array."""
        ...

    @property
    def mass_node(self) -> npt.NDArray[np.float64]:
        """Return (cached) node mass matrix."""
        ...

    @property
    def mass_edge(self) -> npt.NDArray[np.float64]:
        """Return (cached) edge mass matrix."""
        ...

    @property
    def mass_surf(self) -> npt.NDArray[np.float64]:
        """Return (cached) surface mass matrix."""
        ...
    @property
    def mass_node_inv(self) -> npt.NDArray[np.float64]:
        """Return (cached) inverse node mass matrix."""
        ...

    @property
    def mass_edge_inv(self) -> npt.NDArray[np.float64]:
        """Return (cached) inverse edge mass matrix."""
        ...

    @property
    def mass_surf_inv(self) -> npt.NDArray[np.float64]:
        """Return (cached) inverse surface mass matrix."""
        ...

    def mass_from_order(
        self, order: UnknownFormOrder, inverse: bool = False
    ) -> npt.NDArray[np.float64]:
        """Compute mass matrix for the given order.

        Parameters
        ----------
        order : UnknownFormOrder
            Order of the differential for to get the matrix from.

        inverse : bool, default: False
            Should the matrix be inverted.

        Returns
        -------
        array
            Mass matrix of the specified order (or inverse if specified).
        """
        ...

def compute_element_matrix(
    form_orders: Sequence[UnknownFormOrder],
    expressions: _CompiledCodeMatrix,
    vector_fields: Sequence[npt.NDArray[np.float64]],
    element_cache: ElementMassMatrixCache,
    stack_memory: int = 1 << 24,
) -> npt.NDArray[np.float64]:
    """Compute a single element matrix.

    Parameters
    ----------
    form_orders : Sequence of UnknownFormOrder
        Orders of differential forms for the degrees of freedom. Must be between 0 and 2.

    expressions
        Compiled bytecode to execute.

    vector_fields : Sequence of arrays
        Vector field arrays as required for interior product evaluations.

    element_cache : ElementMassMatrixCache
        Cache of the element basis and mass matrices.

    stack_memory : int, default: 1 << 24
        Amount of memory to use for the evaluation stack.

    Returns
    -------
    array
        Element matrix for the specified system.
    """
    ...

def compute_element_vector(
    form_orders: Sequence[UnknownFormOrder],
    expressions: _CompiledCodeMatrix,
    vector_fields: Sequence[npt.NDArray[np.float64]],
    element_cache: ElementMassMatrixCache,
    solution: npt.NDArray[np.float64],
    stack_memory: int = 1 << 24,
) -> npt.NDArray[np.float64]:
    """Compute a single element forcing.

    Parameters
    ----------
    form_orders : Sequence of UnknownFormOrder
        Orders of differential forms for the degrees of freedom. Must be between 0 and 2.

    expressions
        Compiled bytecode to execute.

    vector_fields : Sequence of arrays
        Vector field arrays as required for interior product evaluations.

    element_cache : ElementMassMatrixCache
        Cache of the element basis and mass matrices.

    solution : array
        Array with degrees of freedom for the element.

    stack_memory : int, default: 1 << 24
        Amount of memory to use for the evaluation stack.

    Returns
    -------
    array
        Element vector for the specified system.
    """
    ...

def compute_element_projector(
    form_orders: Sequence[UnknownFormOrder],
    corners: npt.NDArray[np.float64],
    basis_in: Basis2D,
    basis_out: Basis2D,
) -> tuple[npt.NDArray[np.float64]]:
    """Compute :math:`L^2` projection from one space to another.

    Projection takes DoFs from primal space of the first and takes
    them to the primal space of the other.

    Parameters
    ----------
    form_orders : Sequence of UnknownFormOrder
        Sequence of orders of forms which are to be projected.

    corners : (4, 2) array
        Array of corner points of the element.

    basis_in : Basis2D
        Basis from which the DoFs should be taken.

    basis_out : Basis2D
        Basis to which the DoFs are taken.

    Returns
    -------
    tuple of square arrays
        Tuple where each entry is the respective projection matrix for that form.
    """
    ...
