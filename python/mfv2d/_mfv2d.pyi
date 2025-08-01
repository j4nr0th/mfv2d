"""Stub for the C implemented types and functions.

This file contains functions signatures and *copies* of docstrings for the
C-extension which implements all the required fast code.
"""

from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, Self, SupportsIndex, final

import numpy as np
import numpy.typing as npt

from mfv2d.eval import _TranslatedBlock, _TranslatedSystem2D
from mfv2d.kform import Function2D, UnknownFormOrder

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

class Manifold2D:
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

def check_bytecode(expression: _TranslatedBlock, /) -> _TranslatedBlock:
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
class ElementFemSpace2D:
    """Type that cotains corners and basis for each (leaf) element.

    It is also used to compute mass matrices and cache them, since they're often
    reused.
    """

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

    @property
    def orders(self) -> tuple[int, int]:
        """Orders of the basis."""
        ...

    @property
    def integration_orders(self) -> tuple[int, int]:
        """Orders of integration rules used by the basis."""
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
    expressions: _TranslatedSystem2D,
    field_specifications: tuple[int | Function2D, ...],
    element_fem_space: ElementFemSpace2D,
    degrees_of_freedom: npt.NDArray[np.float64] | None = None,
    stack_memory: int = 1 << 24,
) -> npt.NDArray[np.float64]:
    """Compute a single element matrix.

    Parameters
    ----------
    form_orders : Sequence of UnknownFormOrder
        Orders of differential forms for the degrees of freedom. Must be between 0 and 2.

    expressions
        Compiled bytecode to execute.

    field_specifications : tuple of int or Function2D
        Specification for fields used for interior products.

    element_fem_space : ElementFemSpace2D
        Element's FEM space.

    degrees_of_freedom : array, optional
        Array with degrees of freedom for the element.

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
    expressions: _TranslatedSystem2D,
    field_specifications: tuple[int | Function2D, ...],
    element_cache: ElementFemSpace2D,
    degrees_of_freedom: npt.NDArray[np.float64],
    stack_memory: int = 1 << 24,
) -> npt.NDArray[np.float64]:
    """Compute a single element forcing.

    Parameters
    ----------
    form_orders : Sequence of UnknownFormOrder
        Orders of differential forms for the degrees of freedom. Must be between 0 and 2.

    expressions
        Compiled bytecode to execute.

    field_specifications : tuple of int or Function2D
        Specification for fields used for interior products.

    element_fem_space : ElementFemSpace2D
        Element's FEM space.

    degrees_of_freedom : array
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

_ParameterType = ParamSpec("_ParameterType")

class Mesh:
    """Mesh containing topology, geometry, and discretization information.

    Parameters
    ----------
    primal : Manifold2D
        Primal topology manifold.

    dual : Manifold2D
        Dual topology manifold.

    corners : (N, 4, 2) array
        Array of element corners.

    orders : (N, 2) array
        Array of element orders.

    boundary : (N,) array
        Array of boundary edge indices.
    """

    def __new__(
        cls,
        primal: Manifold2D,
        dual: Manifold2D,
        corners: npt.NDArray[np.double],
        orders: npt.NDArray[np.uintc],
        boundary: npt.NDArray[np.uintc],
    ) -> Self: ...
    @property
    def primal(self) -> Manifold2D:
        """Primal manifold topology."""
        ...

    @property
    def dual(self) -> Manifold2D:
        """Dual manifold topology."""
        ...

    @property
    def element_count(self) -> int:
        """Number of elements in the mesh."""
        ...

    @property
    def leaf_count(self) -> int:
        """Number of leaf elements in the mesh."""
        ...

    @property
    def boundary_indices(self) -> npt.NDArray[np.uintc]:
        """Indices of the boundary elements."""
        ...

    def get_element_parent(self, idx: SupportsIndex, /) -> int | None:
        """Get the index of the element's parent or ``None`` if it is a root element.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the element to get the parent from.

        Returns
        -------
        int or None
            If the element has a parent, its index is returned. If the element is a
            root element and has no parent, ``None`` is returned instead.
        """
        ...

    def split_element(
        self,
        idx: SupportsIndex,
        /,
        orders_bottom_left: tuple[int, int],
        orders_bottom_right: tuple[int, int],
        orders_top_right: tuple[int, int],
        orders_top_left: tuple[int, int],
    ) -> None:
        """Split a leaf element into four child elements.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the element to split. Must be a leaf.

        orders_bottom_left : (int, int)
            Orders of the newly created bottom left elements.

        orders_bottom_right : (int, int)
            Orders of the newly created bottom right elements.

        orders_top_right : (int, int)
            Orders of the newly created top right elements..

        orders_top_left : (int, int)
            Orders of the newly created top left elements.
        """
        ...

    def get_element_children(
        self, idx: SupportsIndex, /
    ) -> tuple[int, int, int, int] | None:
        """Get indices of element's children.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the element to get the children for.

        Returns
        -------
        (int, int, int, int) or None
            If the element has children, their indices are returned in the order bottom
            left, bottom right, top right, and top left. If the element is a leaf element
            and has no parents, ``None`` is returned.
        """
        ...

    def get_leaf_corners(self, idx: SupportsIndex, /) -> npt.NDArray[np.double]:
        """Get corners of the leaf element.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the leaf element to get the orders for.

        Returns
        -------
        (4, 2) array
            Corners of the element in the counter-clockwise order, starting at
            the bottom left corner.
        """
        ...

    def get_leaf_orders(self, idx: SupportsIndex, /) -> tuple[int, int]:
        """Get orders of the leaf element.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the leaf element to get the orders for.

        Returns
        -------
        (int, int)
            Orders of the leaf element in the first and second direction.
        """
        ...

    def get_leaf_indices(self) -> npt.NDArray[np.uintc]:
        """Get indices of leaf elements.

        Returns
        -------
        (N,) array
            Indices of leaf elements.
        """
        ...

    def copy(self) -> Mesh:
        """Create a copy of the mesh.

        Returns
        -------
        Mesh
            Copy of the mesh.
        """
        ...

    def get_element_depth(self, idx: SupportsIndex, /) -> int:
        """Check how deep the element is in the hierarchy.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the element element to check.

        Returns
        -------
        int
            Number of ancestors of the element.
        """
        ...

    def set_leaf_orders(self, idx: SupportsIndex, /, order_1: int, order_2: int) -> None:
        """Set orders of a leaf element.

        Parameters
        ----------
        idx : SupportsIndex
            Index of the leaf element to set.

        order_1 : int
            New order of the leaf in the first dimension.

        order_2 : int
            New order of the leaf in the second dimension.
        """
        ...

    def split_depth_first(
        self,
        maximum_depth: int,
        predicate: Callable[
            Concatenate[Mesh, int, _ParameterType],
            None
            | tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        ],
        *args: _ParameterType.args,
        **kwargs: _ParameterType.kwargs,
    ) -> Mesh:
        """Split leaf elements based on a predicate in a depth-first approach.

        Parameters
        ----------
        maximum_depth : int
            Maximum number of levels of division allowed.

        predicate : Callable
            Predicate to use for determining if the element should be split.
            It will alway be given the mesh as the first arguement and
            the ID of the element as its second argument. If the element should
            not be split, the function should return ``None``, otherwise it should
            return orders for all four newly created elements.

        *args
            Arguments passed to the predicate function.

        **kwargs
            Keyword arguments passed to the predicate function.

        Returns
        -------
        Mesh
            Mesh with refined elements.
        """
        ...

    def split_breath_first(
        self,
        maximum_depth: int,
        predicate: Callable[
            Concatenate[Mesh, int, _ParameterType],
            None
            | tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        ],
        *args: _ParameterType.args,
        **kwargs: _ParameterType.kwargs,
    ) -> Mesh:
        """Split leaf elements based on a predicate in a breath-first approach.

        Parameters
        ----------
        maximum_depth : int
            Maximum number of levels of division allowed.

        predicate : Callable
            Predicate to use for determining if the element should be split.
            It will always be given the mesh as the first argument and
            the ID of the element as its second argument. If the element should
            not be split, the function should return ``None``, otherwise it should
            return orders for all four newly created elements.

        *args
            Arguments passed to the predicate function.

        **kwargs
            Keyword arguments passed to the predicate function.

        Returns
        -------
        Mesh
            Mesh with refined elements.
        """
        ...

    def uniform_p_change(self, dp_1: int, dp_2: int, /) -> None:
        """Change orders of all elements by specified amounts.

        Note that if the change would result in a negative order for any element,
        an exception is raised.

        Parameters
        ----------
        dp_1 : int
            Change in the orders of the first dimension.

        dp_2 : int
            Change in the orders of the second dimension.
        """
        ...

# Element side enum values

ELEMENT_SIDE_BOTTOM: int
ELEMENT_SIDE_RIGHT: int
ELEMENT_SIDE_TOP: int
ELEMENT_SIDE_LEFT: int

# Matrix operation values

MATOP_INVALID: int
MATOP_IDENTITY: int
MATOP_MASS: int
MATOP_INCIDENCE: int
MATOP_PUSH: int
MATOP_MATMUL: int
MATOP_SCALE: int
MATOP_SUM: int
MATOP_INTERPROD: int

def compute_integrating_fields(
    fem_space: ElementFemSpace2D,
    form_orders: tuple[UnknownFormOrder, ...],
    field_information: tuple[int | Function2D, ...],
    field_orders: tuple[UnknownFormOrder, ...],
    degrees_of_freedom: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.double], ...]:
    """Compute fields at integration points.

    Parameters
    ----------
    fem_space : ElementFemSpace2D
        Element FEM space to use for basis and integration rules.

    form_orders : tuple of UnknownFormOrder
        Orders of differential forms in the system.

    field_information : tuple of int or Function2D
        Information of how to compute the field - an integer indicates to use degrees of
        freedom of that form, while a function indicates it should be called and
        evaluated.

    degrees_of_freedom : array
        Array with degrees of freedom from which the fields may be computed.

    Returns
    -------
    tuple of arrays
        Fields reconstructed at the integration points.
    """
    ...
