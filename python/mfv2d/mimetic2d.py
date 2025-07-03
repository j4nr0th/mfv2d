"""Implementation of the 2D mimetic meshes and manifolds.

This file contains many miscellaneous functions that are used in the implementation
of the 2D mimetic meshes and manifolds, most of which can probably be factored out
into a separate file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    IntegrationRule1D,
    Manifold2D,
    Surface,
)


# TODO: remake incidence into working for two different orders
def incidence_10(order: int) -> npt.NDArray[np.float64]:
    r"""Incidence matrix from 0.forms to 1-forms.

    This applies the exterior derivative operation to primal 0-forms and maps them
    into 1-forms. The negative transpose is the equivalent operation for the dual
    1-forms, the derivatives of which are consequently dual 2-forms.

    This is done by mapping degrees of freedom of the original primal 0-form or dual
    1-form into those of the derivative primal 1-forms or dual 2-forms respectively.

    .. math::

        \vec{\mathcal{N}}^{(1)}(f) = \mathbb{E}^{(1,0)} \vec{\mathcal{N}}^{(0)}(f)


    .. math::

        \tilde{\mathcal{N}}^{(2)}(f) = -\left(\mathbb{E}^{(1,0)}\right)^{T}
        \tilde{\mathcal{N}}^{(1)}(f)

    Returns
    -------
    array
        Incidence matrix :math:`\mathbb{E}^{(1,0)}`.
    """
    n_nodes = order + 1
    n_lines = order
    e = np.zeros(((n_nodes * n_lines + n_lines * n_nodes), (n_nodes * n_nodes)))

    for row in range(n_nodes):
        for col in range(n_lines):
            e[row * n_lines + col, n_nodes * row + col] = +1
            e[row * n_lines + col, n_nodes * row + col + 1] = -1

    for row in range(n_lines):
        for col in range(n_nodes):
            e[n_nodes * n_lines + row * n_nodes + col, n_nodes * row + col] = -1
            e[n_nodes * n_lines + row * n_nodes + col, n_nodes * (row + 1) + col] = +1

    return e


def apply_e10(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the E10 matrix to the given input.

    Calling this function is equivalent to left multiplying by E10.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((2 * order * (order + 1), other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        for row in range(n_nodes):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_nodes * row + col
                col_e2 = n_nodes * row + col + 1
                out[row_e, i_col] = other[col_e1, i_col] - other[col_e2, i_col]

        for row in range(n_lines):
            for col in range(n_nodes):
                row_e = n_nodes * n_lines + row * n_nodes + col
                col_e1 = n_nodes * (row + 1) + col
                col_e2 = n_nodes * row + col
                out[row_e, i_col] = other[col_e1, i_col] - other[col_e2, i_col]

    return out


def apply_e10_t(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the E10 transpose matrix to the given input.

    Calling this function is equivalent to left multiplying by E10 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros(((order + 1) ** 2, other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        # Nodes with lines on their right
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col
                col_e1 = n_lines * row + col
                out[row_e, i_col] += other[col_e1, i_col]

        # Nodes with lines on their left
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col + 1
                col_e1 = n_lines * row + col
                out[row_e, i_col] -= other[col_e1, i_col]

        # Nodes with lines on their top
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = row * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out[row_e, i_col] -= other[col_e1, i_col]

        # Nodes with lines on their bottom
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = (row + 1) * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out[row_e, i_col] += other[col_e1, i_col]

    return out


def apply_e10_r(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the right E10 matrix to the given input.

    Calling this function is equivalent to right multiplying by E10.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((other.shape[0], (order + 1) ** 2), np.float64)

    for i_row in range(other.shape[0]):
        # Nodes with lines on their right
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col
                col_e1 = n_lines * row + col
                out[i_row, row_e] += other[i_row, col_e1]

        # Nodes with lines on their left
        for row in range(n_nodes):
            for col in range(n_nodes - 1):
                row_e = row * n_nodes + col + 1
                col_e1 = n_lines * row + col
                out[i_row, row_e] -= other[i_row, col_e1]

        # Nodes with lines on their top
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = row * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out[i_row, row_e] -= other[i_row, col_e1]

        # Nodes with lines on their bottom
        for row in range(n_nodes - 1):
            for col in range(n_nodes):
                row_e = (row + 1) * n_nodes + col
                col_e1 = (n_nodes * (n_nodes - 1)) + row * n_nodes + col
                out[i_row, row_e] += other[i_row, col_e1]

    return out


def apply_e10_rt(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the right transposed E10 matrix to the given input.

    Calling this function is equivalent to right multiplying by E10 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((other.shape[0], 2 * order * (order + 1)), np.float64)

    for i_row in range(other.shape[0]):
        for row in range(n_nodes):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_nodes * row + col
                col_e2 = n_nodes * row + col + 1
                out[i_row, row_e] = other[i_row, col_e1] - other[i_row, col_e2]

        for row in range(n_lines):
            for col in range(n_nodes):
                row_e = n_nodes * n_lines + row * n_nodes + col
                col_e1 = n_nodes * (row + 1) + col
                col_e2 = n_nodes * row + col
                out[i_row, row_e] = other[i_row, col_e1] - other[i_row, col_e2]

    return out


def incidence_21(order: int) -> npt.NDArray[np.float64]:
    r"""Incidence matrix from 1-forms to 2-forms.

    This applies the exterior derivative operation to primal 1-forms and maps them
    into 2-forms. The negative transpose is the equivalent operation for the dual
    0-forms, the derivatives of which are consequently dual 1-forms.

    This is done by mapping degrees of freedom of the original primal 1-form or dual
    0-form into those of the derivative primal 2-forms or dual 1-forms respectively.

    .. math::

        \vec{\mathcal{N}}^{(2)}(f) = \mathbb{E}^{(2,1)} \vec{\mathcal{N}}^{(1)}(f)


    .. math::

        \tilde{\mathcal{N}}^{(1)}(f) = -\left(\mathbb{E}^{(2,1)}\right)^{T}
        \tilde{\mathcal{N}}^{(0)}(f)

    Returns
    -------
    array
        Incidence matrix :math:`\mathbb{E}^{(2,1)}`.
    """
    n_nodes = order + 1
    n_lines = order
    e = np.zeros(((n_lines * n_lines), (n_nodes * n_lines + n_lines * n_nodes)))

    for row in range(n_lines):
        for col in range(n_lines):
            e[row * n_lines + col, n_lines * row + col] = +1
            e[row * n_lines + col, n_lines * (row + 1) + col] = -1
            e[row * n_lines + col, n_nodes * n_lines + n_nodes * row + col] = +1
            e[row * n_lines + col, n_nodes * n_lines + n_nodes * row + col + 1] = -1

    return e


def apply_e21(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the E21 matrix to the given input.

    Calling this function is equivalent to left multiplying by E21.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((order**2, other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_lines * row + col  # +
                col_e2 = n_lines * (row + 1) + col  # -
                col_e3 = n_nodes * n_lines + n_nodes * row + col  # +
                col_e4 = n_nodes * n_lines + n_nodes * row + col + 1  # -
                out[row_e, i_col] = (
                    other[col_e1, i_col]
                    - other[col_e2, i_col]
                    + other[col_e3, i_col]
                    - other[col_e4, i_col]
                )

    return out


def apply_e21_t(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the E21 transposed matrix to the given input.

    Calling this function is equivalent to left multiplying by E21 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros(((2 * order * (order + 1)), other.shape[1]), np.float64)

    for i_col in range(other.shape[1]):
        # Lines with surfaces on the top
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = row * n_lines + col
                out[row_e, i_col] = other[col_e1, i_col]

        # Lines with surfaces on the bottom
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (row + 1) * n_lines + col
                col_e1 = row * n_lines + col
                out[row_e, i_col] -= other[col_e1, i_col]

        # Lines with surfaces on the left
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col
                col_e1 = row * n_lines + col
                out[row_e, i_col] += other[col_e1, i_col]

        # Lines with surfaces on the right
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col + 1
                col_e1 = row * n_lines + col
                out[row_e, i_col] -= other[col_e1, i_col]

    return out


def apply_e21_r(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the right E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((other.shape[0], (order + 1) * order * 2), np.float64)

    for i_row in range(other.shape[0]):
        # Lines with surfaces on the top
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = row * n_lines + col
                out[i_row, row_e] = other[i_row, col_e1]

        # Lines with surfaces on the bottom
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (row + 1) * n_lines + col
                col_e1 = row * n_lines + col
                out[i_row, row_e] -= other[i_row, col_e1]

        # Lines with surfaces on the left
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col
                col_e1 = row * n_lines + col
                out[i_row, row_e] += other[i_row, col_e1]

        # Lines with surfaces on the right
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = (order + 1) * order + row * n_nodes + col + 1
                col_e1 = row * n_lines + col
                out[i_row, row_e] -= other[i_row, col_e1]

    return out


def apply_e21_rt(order: int, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the right transpose E21 matrix to the given input.

    Calling this function is equivalent to right multiplying by E21 transposed.
    """
    assert other.ndim == 2
    n_nodes = order + 1
    n_lines = order
    out = np.zeros((other.shape[0], order**2), np.float64)

    for i_row in range(other.shape[0]):
        for row in range(n_lines):
            for col in range(n_lines):
                row_e = row * n_lines + col
                col_e1 = n_lines * row + col  # +
                col_e2 = n_lines * (row + 1) + col  # -
                col_e3 = n_nodes * n_lines + n_nodes * row + col  # +
                col_e4 = n_nodes * n_lines + n_nodes * row + col + 1  # -
                out[i_row, row_e] = (
                    other[i_row, col_e1]
                    - other[i_row, col_e2]
                    + other[i_row, col_e3]
                    - other[i_row, col_e4]
                )

    return out


def vtk_lagrange_ordering(order: int) -> npt.NDArray[np.uint32]:
    """Ordering for vtkLagrangeQuadrilateral.

    VTK has an option to create cells of type LagrangeQuadrilateral. These
    allow for arbitrary order of interpolation with nodal basis. Due to
    backwards compatibility the ordering of the nodes in these is done in
    an unique way. As such, either the positions or ordering of the nodes
    must be adjusted.

    This function returns the correct order which can be used for either
    given a specific polynomial order.

    Parameters
    ----------
    order : int
        Order of the element.

    Returns
    -------
    array
        Array of indices which correctly order nodes on an element of
        the specified order.
    """
    n = int(order) + 1
    v = np.arange(n)
    return np.astype(
        np.concatenate(
            (
                (0, n - 1, n**2 - 1, n * (n - 1)),  # corners
                v[1:-1],  # bottom edge
                n - 1 + n * v[1:-1],  # right edge
                n * (n - 1) + v[1:-1],  # top edge
                n * v[1:-1],  # left edge
                np.concatenate([v[1:-1] + n * k for k in v[1:-1]]),
            )
        )
        if order > 1
        else np.concatenate(
            (
                (0, n - 1, n**2 - 1, n * (n - 1)),  # corners
            )
        ),
        np.uint32,
        copy=False,
    )


@dataclass(eq=False)
class Element2D:
    """General 2D element."""

    parent: ElementNode2D | None


@dataclass(eq=False)
class ElementNode2D(Element2D):
    """Two dimensional element that contains children."""

    child_bl: Element2D
    child_br: Element2D
    child_tl: Element2D
    child_tr: Element2D

    def children(self) -> tuple[Element2D, Element2D, Element2D, Element2D]:
        """Return children of the element ordered."""
        return (self.child_bl, self.child_br, self.child_tr, self.child_tl)


@dataclass(eq=False)
class ElementLeaf2D(Element2D):
    """Two dimensional square element.

    This type facilitates operations related to calculations which need
    to be carried out on the reference element itself, such as calculation
    of the mass and incidence matrices, as well as the reconstruction of
    the solution.

    Parameters
    ----------
    order_h : int
        Order of the basis functions used for the nodal basis in the first dimension.
    order_v : int
        Order of the basis functions used for the nodal basis in the second dimension.
    bottom_left : (float, float)
        Coordinates of the bottom left corner.
    bottom_right : (float, float)
        Coordinates of the bottom right corner.
    top_right : (float, float)
        Coordinates of the top right corner.
    top_left : (float, float)
        Coordinates of the top left corner.
    """

    order: int

    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]
    top_left: tuple[float, float]

    def divide(
        self,
        order_bl: int,
        order_br: int,
        order_tl: int,
        order_tr: int,
    ) -> tuple[
        ElementNode2D,
        tuple[tuple[ElementLeaf2D, ElementLeaf2D], tuple[ElementLeaf2D, ElementLeaf2D]],
    ]:
        """Divide the element into four child elements of the specified order.

        Parameters
        ----------
        order_bl : int
            Order of the bottom left element.
        order_br : int
            Order of the bottom right element.
        order_tl : int
            Order of the top left element.
        order_tr : int
            Order of the top right element.

        Returns
        -------
        ElementNode2D
            Parent element which contains the nodes.

        (2, 2) tuple of ElementLeaf2D
            Child elements of the same order as the element itself. Indexing the
            tuple will give bottom/top for the first axis and left/right for the
            second.
        """
        bottom_mid = (np.array(self.bottom_left) + np.array(self.bottom_right)) / 2
        left_mid = (np.array(self.bottom_left) + np.array(self.top_left)) / 2
        right_mid = (np.array(self.bottom_right) + np.array(self.top_right)) / 2
        top_mid = (np.array(self.top_left) + np.array(self.top_right)) / 2
        center_mid = (
            np.array(self.bottom_left)
            + np.array(self.bottom_right)
            + np.array(self.top_left)
            + np.array(self.top_right)
        ) / 4
        btm_l = ElementLeaf2D(
            None,
            order_bl,
            self.bottom_left,
            tuple(bottom_mid),
            tuple(center_mid),
            tuple(left_mid),
        )
        btm_r = ElementLeaf2D(
            None,
            order_br,
            tuple(bottom_mid),
            self.bottom_right,
            tuple(right_mid),
            tuple(center_mid),
        )
        top_r = ElementLeaf2D(
            None,
            order_tr,
            tuple(center_mid),
            tuple(right_mid),
            self.top_right,
            tuple(top_mid),
        )
        top_l = ElementLeaf2D(
            None,
            order_tl,
            tuple(left_mid),
            tuple(center_mid),
            tuple(top_mid),
            self.top_left,
        )

        parent = ElementNode2D(self.parent, btm_l, btm_r, top_l, top_r)

        btm_l.parent = parent
        btm_r.parent = parent
        top_l.parent = parent
        top_r.parent = parent

        return parent, ((btm_l, btm_r), (top_l, top_r))


class Mesh2D:
    """Two dimensional manifold with associated geometry.

    Mesh holds the primal manifold, which describes the topology of surfaces
    and lines that make it up. It also contains the dual mesh, which contains
    duals of all primal geometrical objects. The dual is useful when connectivity
    is needed.

    Parameters
    ----------
    order : int or Sequence of int or array-like
        Orders of elements. If a single value is specified, then the same order
        is used for all elements. If a sequence is specified, then each element
        is given that order.

    positions : (N, 2) array-like
        Positions of the nodes.

    lines : (N, 2) array-like
        Lines of the mesh specified as pairs of nodes connected. These
        use 1-based indexing.

    surfaces : (N, 4) array-like
        Surfaces of the mesh specified in their positive orientation
        as 1-based indices of lines that make up the surface, with
        negative sign indicating reverse direction.
    """

    orders: npt.NDArray[np.uint32]
    """Orders of individual elements."""
    positions: npt.NDArray[np.float64]
    """Array of positions for each node in the mesh."""
    primal: Manifold2D
    """Primal topology of the mesh."""
    dual: Manifold2D
    """Dual topology of the mesh."""
    boundary_indices: npt.NDArray[np.int32]
    """Indices of lines that make up the boundary of the mesh. These
        can be useful when prescribing boundary conditions."""

    def __init__(
        self,
        order: int | Sequence[int] | npt.ArrayLike,
        positions: Sequence[tuple[float, float, float]]
        | Sequence[Sequence[float]]
        | Sequence[npt.ArrayLike]
        | npt.ArrayLike,
        lines: Sequence[tuple[int, int]]
        | Sequence[npt.ArrayLike]
        | Sequence[Sequence[int]]
        | npt.ArrayLike,
        surfaces: Sequence[tuple[int, ...]]
        | Sequence[Sequence[int]]
        | Sequence[npt.ArrayLike]
        | npt.ArrayLike,
    ) -> None:
        """Create new mesh from given geometry."""
        pos = np.array(positions, np.float64, copy=True, ndmin=2)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("Positions must be a (N, 2) array.")
        # First try the regular surfaces
        surf = np.array(surfaces, np.int32, copy=None)
        if surf.ndim != 2 or surf.shape[1] != 4:
            raise ValueError("Surfaces should be a (M, 4) array of integers")

        orders_array = np.array(order, dtype=np.uint32)
        if orders_array.ndim == 0:
            orders_array = np.full(surf.shape[0], orders_array)
        else:
            if orders_array.ndim != 1 or orders_array.size != surf.shape[0]:
                raise ValueError(
                    "Orders must be 1D sequence with as many elements as the elements."
                )

        if np.any(orders_array < 1):
            raise ValueError("Order can not be lower than 1.")

        self.orders = orders_array

        lns = np.array(lines, np.int32, copy=None)
        man = Manifold2D.from_regular(pos.shape[0], lns, surf)

        self.positions = pos
        self.primal = man
        self.dual = man.compute_dual()
        bnd: list[int] = []
        for n_line in range(self.dual.n_lines):
            ln = self.dual.get_line(n_line + 1)
            if not ln.begin or not ln.end:
                bnd.append(n_line)
        self.boundary_indices = np.array(bnd, np.int32)

    @property
    def n_elements(self) -> int:
        """Number of (surface) elements in the mesh."""
        return self.primal.n_surfaces

    def surface_to_element(self, idx: int, /) -> ElementLeaf2D:
        """Create a 2D element from a surface with the given index.

        Parameters
        ----------
        idx : int
            Index of the surface.

        Returns
        -------
        ElementLeaf2D
            Leaf element with the geometry of the specified surface.
        """
        s = self.primal.get_surface(idx + 1)
        assert len(s) == 4, "Primal surface must be square."
        indices = np.zeros(4, dtype=int)
        for i in range(4):
            line = self.primal.get_line(s[i])
            indices[i] = line.begin.index
        return ElementLeaf2D(
            None,
            int(self.orders[idx]),
            (float(self.positions[indices[0], 0]), float(self.positions[indices[0], 1])),
            (float(self.positions[indices[1], 0]), float(self.positions[indices[1], 1])),
            (float(self.positions[indices[2], 0]), float(self.positions[indices[2], 1])),
            (float(self.positions[indices[3], 0]), float(self.positions[indices[3], 1])),
        )


class FemCache:
    """Cache for integration rules and basis functions.

    This type allows for caching 1D integration rules and 1D basis.
    The 2D basis are not cached, since the :class:`Basis2D` is just
    a container for two 1D Basis objects, so it is probably just as
    cheap to create a new one as it would be to cache it.

    Parameters
    ----------
    order_difference : int
        Difference of orders between the integration rule and the basis
        in the case that it is not specified.
    """

    order_diff: int
    _int_cache: dict[int, IntegrationRule1D]
    _b1_cache: dict[tuple[int, int], Basis1D]
    _min_cache: dict[int, npt.NDArray[np.float64]]
    _mie_cache: dict[int, npt.NDArray[np.float64]]

    def __init__(self, order_difference: int) -> None:
        self._int_cache = dict()
        self._b1_cache = dict()
        self.order_diff = order_difference
        self._min_cache = dict()
        self._mie_cache = dict()

    def get_integration_rule(self, order: int) -> IntegrationRule1D:
        """Return integration rule.

        Parameters
        ----------
        order : int
            Order of integration rule to use.

        Returns
        -------
        IntegrationRule1D
            Integration rule that was obtained from cache or created and cached if
            it was previously not there.
        """
        res = self._int_cache.get(order, None)
        if res is not None:
            return res
        rule = IntegrationRule1D(order)
        self._int_cache[order] = rule
        return rule

    def get_basis1d(self, order: int, int_order: int | None = None) -> Basis1D:
        """Get requested one-dimensional basis.

        Parameters
        ----------
        order : int
            Order of the basis.

        int_order : int, optional
            Order of the integration rule for the basis. If it is not specified,
            then ``order + self.order_diff`` is used.

        Returns
        -------
        Basis1D
            One-dimensional basis.
        """
        if int_order is None:
            int_order = order + self.order_diff

        res = self._b1_cache.get((order, int_order), None)
        if res is not None:
            return res

        rule = self.get_integration_rule(int_order)
        basis = Basis1D(order, rule)
        self._b1_cache[(order, int_order)] = basis
        return basis

    def get_basis2d(
        self,
        order1: int,
        order2: int,
        int_order1: int | None = None,
        int_order2: int | None = None,
    ) -> Basis2D:
        """Get two-dimensional basis.

        These are not cached, since there is zero calculations involved in
        their computations.

        Parameters
        ----------
        order1 : int
            Order of basis in the first dimension.

        order2 : int
            Order of basis in the second dimension.

        int_order1 : int, optional
            Order of the integration rule in the first direction. If unspecified,
            it defaults to ``order1 + self.order_diff``.

        int_order2 : int, optional
            Order of the integration rule in the second direction. If unspecified,
            it defaults to ``order2 + self.order_diff``.

        Returns
        -------
        Basis2D
            Requested two-dimensional basis.
        """
        b_xi = self.get_basis1d(order1, int_order1)
        if order2 != order1 or int_order1 != int_order2:
            b_eta = self.get_basis1d(order2, int_order2)
        else:
            b_eta = b_xi
        return Basis2D(b_xi, b_eta)

    def clean(self) -> None:
        """Clear all caches."""
        self._int_cache = dict()
        self._b1_cache = dict()

    def get_mass_inverse_1d_node(self, order: int) -> npt.NDArray[np.float64]:
        """Get the 1D nodal mass matrix inverse."""
        if order in self._min_cache:
            return self._min_cache[order]

        basis = self.get_basis1d(order)
        rule = basis.rule
        weights = rule.weights

        mat = np.sum(
            basis.node[:, None, :] * basis.node[None, :, :] * weights[None, None, :],
            axis=2,
        )
        inv = np.linalg.inv(mat)
        self._min_cache[order] = inv

        return inv

    def get_mass_inverse_1d_edge(self, order: int) -> npt.NDArray[np.float64]:
        """Get the 1D edge mass matrix inverse."""
        if order in self._mie_cache:
            return self._mie_cache[order]

        basis = self.get_basis1d(order)
        rule = basis.rule
        weights = rule.weights

        mat = np.sum(
            basis.edge[:, None, :] * basis.edge[None, :, :] * weights[None, None, :],
            axis=2,
        )
        inv = np.linalg.inv(mat)
        self._mie_cache[order] = inv

        return inv


class ElementSide(IntEnum):
    """Enum specifying the side of an element."""

    SIDE_BOTTOM = 1
    SIDE_RIGHT = 2
    SIDE_TOP = 3
    SIDE_LEFT = 4

    @property
    def next(self) -> ElementSide:
        """Next side."""
        return ElementSide((self.value & 3) + 1)

    @property
    def prev(self) -> ElementSide:
        """Previous side."""
        return ElementSide(((self.value - 2) & 3) + 1)


def find_surface_boundary_id_line(s: Surface, i: int) -> ElementSide:
    """Find what boundary the line with a given index is in the surface."""
    if s[0].index == i:
        return ElementSide.SIDE_BOTTOM
    if s[1].index == i:
        return ElementSide.SIDE_RIGHT
    if s[2].index == i:
        return ElementSide.SIDE_TOP
    if s[3].index == i:
        return ElementSide.SIDE_LEFT
    raise ValueError(f"Line with index {i} is not in the surface {s}.")
