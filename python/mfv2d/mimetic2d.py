"""Implementation of the 2D mimetic meshes and manifolds.

This file contains many miscellaneous functions that are used in the implementation
of the 2D mimetic meshes and manifolds, most of which can probably be factored out
into a separate file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import accumulate

import numpy as np
import numpy.typing as npt

from mfv2d._mfv2d import (
    ELEMENT_SIDE_BOTTOM as _ELEMENT_SIDE_BOTTOM,
)
from mfv2d._mfv2d import (
    ELEMENT_SIDE_LEFT as _ELEMENT_SIDE_LEFT,
)
from mfv2d._mfv2d import (
    ELEMENT_SIDE_RIGHT as _ELEMENT_SIDE_RIGHT,
)
from mfv2d._mfv2d import (
    ELEMENT_SIDE_TOP as _ELEMENT_SIDE_TOP,
)
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementMassMatrixCache,
    IntegrationRule1D,
    Manifold2D,
    Mesh,
    Surface,
    compute_gll,
    dlagrange1d,
    lagrange1d,
)
from mfv2d.kform import UnknownFormOrder, UnknownOrderings


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

    SIDE_BOTTOM = _ELEMENT_SIDE_BOTTOM
    SIDE_RIGHT = _ELEMENT_SIDE_RIGHT
    SIDE_TOP = _ELEMENT_SIDE_TOP
    SIDE_LEFT = _ELEMENT_SIDE_LEFT

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


def mesh_create(
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
) -> Mesh:
    """Create new mesh from given geometry."""
    pos = np.array(positions, np.float64, copy=True, ndmin=2)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("Positions must be a (N, 2) array.")
    # First try the regular surfaces
    surf = np.array(surfaces, np.int32, copy=None)
    if surf.ndim != 2 or surf.shape[1] != 4:
        raise ValueError("Surfaces should be a (M, 4) array of integers")

    n_surf = surf.shape[0]

    orders_array = np.array(order, dtype=np.uint32)
    if orders_array.ndim == 0:
        orders_array = np.full((n_surf, 2), orders_array)
    elif orders_array.shape[0] != n_surf:
        raise ValueError(
            "Orders array must contain as many entries as there are surfaces."
        )
    elif orders_array.ndim == 1:
        orders_array = np.stack((orders_array, orders_array), axis=1)
    elif orders_array.ndim != 2 or orders_array.shape[1] != 2:
        raise ValueError(
            "Orders must be given as either a single value, a (N,) sequence, or (N, 2)"
            " sequence where N is the number of elements."
        )

    if np.any(orders_array < 1):
        raise ValueError("Order can not be lower than 1.")

    orders = np.astype(orders_array, np.uintc, copy=False)

    lns = np.array(lines, np.int32, copy=None)
    primal = Manifold2D.from_regular(pos.shape[0], lns, surf)
    dual = primal.compute_dual()

    corners = np.empty((n_surf, 4, 2), np.double)
    indices = np.empty(4, np.uintc)
    for idx_surf in range(n_surf):
        s = primal.get_surface(idx_surf + 1)
        assert len(s) == 4
        for n_line in range(4):
            line = primal.get_line(s[n_line])
            indices[n_line] = line.begin.index
        corners[idx_surf] = pos[indices, :]

    bnd: list[int] = []
    for n_line in range(dual.n_lines):
        ln = dual.get_line(n_line + 1)
        if not ln.begin or not ln.end:
            bnd.append(n_line)
    boundary_indices = np.array(bnd, np.uintc)

    return Mesh(primal, dual, corners, orders, boundary_indices)


def element_node_children_on_side(
    side: ElementSide, children: tuple[int, int, int, int]
) -> tuple[int, int]:
    """Get children from the 4 child array for the correct side."""
    i_begin = side.value - 1
    i_end = side.value & 3
    return int(children[i_begin]), int(children[i_end])


def get_element_side_nodes(
    mesh: Mesh, element: int, side: ElementSide
) -> npt.NDArray[np.float64]:
    """Get nodes on the specified side."""
    children = mesh.get_element_children(element)
    if children is None:
        orders = mesh.get_leaf_orders(element)
        order = orders[(side.value - 1) & 1]
        return compute_gll(order)[0]

    c1, c2 = element_node_children_on_side(side, children)

    return np.concatenate(
        (
            (get_element_side_nodes(mesh, c1, side) - 1) / 2,
            (get_element_side_nodes(mesh, c2, side) + 1) / 2,
        )
    )


def element_boundary_dofs(
    side: ElementSide, order: UnknownFormOrder, order_1: int, order_2: int
) -> npt.NDArray[np.uint32]:
    """Get indices of boundary DoFs for an element on specified side.

    Parameters
    ----------
    side : ElementSide
        Side of the element to get the DoFs from.

    order : UnknownFormOrder
        Order of the form to the the boundary degrees are for. Can only be a
        0-form or an 1-form, since 2-forms have no boundary degrees of freedom
        on the boundary.

    order_1 : int
        Order of the element in the horizontal direction.

    order_2 : int
        Order of the element in the vertical direction.

    Returns
    -------
    array
        Array of indices of the degrees of freedom for the given form on a
        given side.
    """
    indices: npt.NDArray[np.uint32]
    if order == UnknownFormOrder.FORM_ORDER_1:
        if side == ElementSide.SIDE_BOTTOM:
            indices = np.arange(0, order_1, dtype=np.uint32)
        elif side == ElementSide.SIDE_RIGHT:
            indices = np.astype(
                (order_1 * (order_2 + 1))
                + order_2
                + np.arange(0, order_2, dtype=np.uint32) * (order_1 + 1),
                np.uint32,
                copy=False,
            )
        elif side == ElementSide.SIDE_TOP:
            indices = np.astype(
                np.flip(order_1 * order_2 + np.arange(0, order_1, dtype=np.uint32)),
                np.uint32,
                copy=False,
            )
        elif side == ElementSide.SIDE_LEFT:
            indices = np.astype(
                np.flip(
                    (order_1 * (order_2 + 1))
                    + np.arange(0, order_2, dtype=np.uint32) * (order_1 + 1)
                ),
                np.uint32,
                copy=False,
            )
        else:
            raise ValueError(f"Invalid side given by {side=}.")

    elif order == UnknownFormOrder.FORM_ORDER_0:
        if side == ElementSide.SIDE_BOTTOM:
            indices = np.arange(0, order_1 + 1, dtype=np.uint32)
        elif side == ElementSide.SIDE_RIGHT:
            indices = np.astype(
                order_1 + np.arange(0, order_2 + 1, dtype=np.uint32) * (order_1 + 1),
                np.uint32,
                copy=False,
            )
        elif side == ElementSide.SIDE_TOP:
            indices = np.astype(
                np.flip(
                    (order_1 + 1) * order_2 + np.arange(0, order_1 + 1, dtype=np.uint32)
                ),
                np.uint32,
                copy=False,
            )
        elif side == ElementSide.SIDE_LEFT:
            indices = np.astype(
                np.flip(np.arange(0, order_2 + 1, dtype=np.uint32) * (order_1 + 1)),
                np.uint32,
                copy=False,
            )
        else:
            raise ValueError(f"Invalid side given by {side=}")

    elif order == UnknownFormOrder.FORM_ORDER_2:
        raise ValueError("2-forms have no boundary DoFs.")

    else:
        raise ValueError(f"Invalid order given by {order=}")
    return indices


@dataclass(frozen=True)
class ElementConstraint:
    """Type intended to enforce a constraint on an element.

    Parameters
    ----------
    i_e : int
        Index of the element for which this constraint is applied.

    dofs : (n,) array
        Array with indices of the degrees of freedom of the element involved.

    coeffs : (n,) array
        Array with coefficients of degrees of freedom of the element involved.
    """

    i_e: int
    dofs: npt.NDArray[np.uint32]
    coeffs: npt.NDArray[np.float64]


def get_side_dof_nodes(
    mesh: Mesh,
    element: int,
    side: ElementSide,
    order: UnknownFormOrder,
    /,
) -> list[ElementConstraint]:
    """Get equations for obtaining DoFs on the side for the element.

    Parameters
    ----------
    element_collection : ElementCollection
        Collection of elements the ``element`` is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side of the element to get the DoFs for.

    order : UnknownFormOrder
        Order of unknown forms for which to get the form orders from. It
        can only be ``UnknownFormOrder.FORM_ORDER_0`` or
        ``UnknownFormOrder.FORM_ORDER_1``, since 2-forms do not have any
        boundary DoFs.

    Returns
    -------
    constraints : list of ElementConstraint
        Specifiecation of which *leaf* elements are involved. These
        are also ordered. The ``coeff`` member specifies positions
        of these in the element.
    """
    children = mesh.get_element_children(element)
    if children is not None:
        c1: int
        c2: int
        c1, c2 = element_node_children_on_side(side, children)

        dofs1 = get_side_dof_nodes(mesh, c1, side, order)
        dofs2 = get_side_dof_nodes(mesh, c2, side, order)

        if order == UnknownFormOrder.FORM_ORDER_0:
            # Do not include the last row (DoF shared between the two)
            # since it overlaps with M1. Otherwise it overconstrains.
            dofs2[0] = ElementConstraint(
                dofs2[0].i_e, dofs2[0].dofs[1:], dofs2[0].coeffs[1:]
            )
        elif order == UnknownFormOrder.FORM_ORDER_1:
            # Still have to remove coeffs (since that stands for child nodes!)
            dofs2[0] = ElementConstraint(dofs2[0].i_e, dofs2[0].dofs, dofs2[0].coeffs[1:])
        else:
            assert False

        combined_dofs = [
            ElementConstraint(dof.i_e, dof.dofs, (dof.coeffs - 1) / 2) for dof in dofs1
        ] + [ElementConstraint(dof.i_e, dof.dofs, (dof.coeffs + 1) / 2) for dof in dofs2]

        return combined_dofs

    # This is a leaf
    n1, n2 = mesh.get_leaf_orders(element)

    indices = element_boundary_dofs(side, order, n1, n2)

    side_orders = mesh.get_leaf_orders(element)
    side_order = side_orders[(side.value - 1) & 1]

    return [ElementConstraint(element, indices, compute_gll(side_order)[0])]


def get_side_order(mesh: Mesh, element_idx: int, side: ElementSide, /) -> int:
    """Get order for the specified element boundary side.

    element_collection : ElementCollection
        Element collection the element is a part of.

    element : int
        Index of the element in the collection to get the side order from.

    side : ElementSide
        Side of the element to get the order for.

    Returns
    -------
    int
        Order on that side of the element.
    """
    children = mesh.get_element_children(element_idx)
    if children is not None:
        c1, c2 = element_node_children_on_side(side, children)
        return get_side_order(mesh, c1, side) + get_side_order(mesh, c2, side)

    orders = mesh.get_leaf_orders(element_idx)
    return int(orders[(side.value - 1) & 1])


@dataclass(frozen=True)
class Constraint:
    """Type used to specify constraints on degrees of freedom.

    This type combines the individual :class:`ElementConstraint` together
    with a right-hand side of the constraint.

    Parameters
    ----------
    rhs : float
        The right-hand side of the constraint.

    *element_constraints : ElementConstraint
        Constraints to combine together.
    """

    rhs: float
    element_constraints: tuple[ElementConstraint, ...]

    def __init__(self, rhs: float, *element_constraints: ElementConstraint) -> None:
        object.__setattr__(self, "rhs", float(rhs))
        object.__setattr__(self, "element_constraints", element_constraints)


def get_side_dofs(
    mesh: Mesh,
    element: int,
    side: ElementSide,
    form_order: UnknownFormOrder,
    output_order: int | None = None,
    /,
) -> tuple[Constraint, ...]:
    """Get DoFs on the boundary in terms of leaf element DoFs.

    Parameters
    ----------
    element_collection : ElementCollection
        Collection of elements the ``element`` is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side of the element to get the DoFs for.

    order : UnknownFormOrder
        Order of unknown forms for which to get the form orders from. It
        can only be ``UnknownFormOrder.FORM_ORDER_0`` or
        ``UnknownFormOrder.FORM_ORDER_1``, since 2-forms do not have any
        boundary DoFs.

    Returns
    -------
    tuple of Constraint
        Tuple of constraints, each of which specifies how the "normal" DoFs
        on the element's boundary may be constructed from DoFs of the element.
    """
    self_order = get_side_order(mesh, element, side)

    # If output order is not specified, use own order
    if output_order is None:
        output_order = self_order

    if mesh.get_element_children(element) is None and output_order == self_order:
        # fast track for leaf elements with no projection, since it should be identity
        indices = element_boundary_dofs(side, form_order, *mesh.get_leaf_orders(element))
        out_c = tuple(
            Constraint(
                0.0,
                ElementConstraint(
                    element, np.array([idx], np.uint32), np.ones(1, np.float64)
                ),
            )
            for idx in indices
        )
        return out_c

    dofs = get_side_dof_nodes(mesh, element, side, form_order)

    self_nodes = compute_gll(self_order)[0]
    input_nodes = np.concatenate([dof.coeffs for dof in dofs])

    # Values of output basis (axis 1) at input points (axis 0)
    # nodal_basis_vals and edge_basis_vals are maps from parent dofs to child dofs
    nodal_basis_vals = lagrange1d(self_nodes, input_nodes)
    if form_order == UnknownFormOrder.FORM_ORDER_0:
        m = np.linalg.inv(nodal_basis_vals)

    elif form_order == UnknownFormOrder.FORM_ORDER_1:
        diffs = nodal_basis_vals[:-1, :] - nodal_basis_vals[+1:, :]
        edge_basis_vals = np.stack(
            [x for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))],
            axis=-1,
            dtype=np.float64,
        )
        m = np.linalg.inv(edge_basis_vals)

    elif form_order == UnknownFormOrder.FORM_ORDER_2:
        raise ValueError("2-forms have no boundary DoFs.")

    else:
        raise ValueError(f"Invalid for order {form_order=}.")

    if self_order != output_order:
        # Values of output basis (axis 1) at input points (axis 0)
        # nodal_basis_vals and edge_basis_vals are maps from parent dofs to child dofs
        output_nodes = compute_gll(output_order)[0]
        map_nodal_basis_vals = lagrange1d(self_nodes, output_nodes)
        if form_order == UnknownFormOrder.FORM_ORDER_0:
            m = map_nodal_basis_vals @ m

        elif form_order == UnknownFormOrder.FORM_ORDER_1:
            diffs = map_nodal_basis_vals[:-1, :] - map_nodal_basis_vals[+1:, :]
            map_edge_basis_vals = np.stack(
                [
                    x
                    for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))
                ],
                axis=-1,
                dtype=np.float64,
            )
            m = map_edge_basis_vals @ m

        elif form_order == UnknownFormOrder.FORM_ORDER_2:
            raise ValueError("2-forms have no boundary DoFs.")

        else:
            raise ValueError(f"Invalid for order {form_order=}.")

    constraints: list[Constraint] = list()

    vrow: npt.NDArray[np.float64]
    for vrow in m:
        col_offset = 0
        elem_constraints: list[ElementConstraint] = list()

        for elem_dofs in dofs:
            cnt = elem_dofs.dofs.size
            element_constraint = ElementConstraint(
                elem_dofs.i_e, elem_dofs.dofs, vrow[col_offset : col_offset + cnt]
            )
            col_offset += cnt
            elem_constraints.append(element_constraint)

        assert col_offset == vrow.size
        constraint = Constraint(0.0, *elem_constraints)
        constraints.append(constraint)

    return tuple(constraints)


def get_corner_dof(mesh: Mesh, element: int, side: ElementSide, /) -> tuple[int, int]:
    """Get element index and degree of freedom index for the corner of the element.

    Parameters
    ----------
    element_collection : ElementCollection
        Collection of elements the ``element`` is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side which begins with the corner that should be obtained.

    Returns
    -------
    element_id : int
        Index of the (leaf) element to which the corner belongs to.

    dof_index : int
        Index of the (0-form) degree of freedom which is in that corner.
    """
    children = mesh.get_element_children(element)
    if children is None:
        # Actual leaf
        order_1, order_2 = mesh.get_leaf_orders(element)

        if side == ElementSide.SIDE_BOTTOM:
            idx = 0
        elif side == ElementSide.SIDE_RIGHT:
            idx = order_1
        elif side == ElementSide.SIDE_TOP:
            idx = (order_1 + 1) * order_2 + order_1
        elif side == ElementSide.SIDE_LEFT:
            idx = order_2 * (order_1 + 1)
        else:
            raise ValueError(f"Invalid side given by {side=}")

        return (element, idx)

    child = children[side.value - 1]

    return get_corner_dof(mesh, child, side)


def compute_leaf_dof_counts(
    ie: int, ordering: UnknownOrderings, mesh: Mesh
) -> npt.NDArray[np.uint32]:
    """Compute number of DoFs for each element.

    Parameters
    ----------
    ie : int
        Index of the element.

    ordering : UnknownOrderings
        Orders of differential forms in the system.

    elements : ElementCollection
        Element collection to use.

    Returns
    -------
    array of int
        Array with count of degrees of freedom for of each differential form
        for the element.
    """
    sizes = np.zeros(ordering.count, np.uint32)
    # This is not a child-less array
    order_1, order_2 = mesh.get_leaf_orders(ie)
    for i_f, form in enumerate(ordering.form_orders):
        sizes[i_f] = form.full_unknown_count(order_1, order_2)

    return sizes


def jacobian(
    corners: npt.NDArray[np.floating], nodes_1: npt.ArrayLike, nodes_2: npt.ArrayLike
) -> tuple[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    r"""Evaluate the Jacobian matrix entries.

    The Jacobian matrix :math:`\mathbf{J}` is defined such that:

    .. math::

        \mathbf{J} = \begin{bmatrix}
        \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\
        \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta} \\
        \end{bmatrix}

    Which means that a coordinate transformation is performed by:

    .. math::

        \begin{bmatrix} {dx} \\ {dy} \end{bmatrix} = \mathbf{J}
        \begin{bmatrix} {d\xi} \\ {d\eta} \end{bmatrix}

    Parameters
    ----------
    corners : (4, 2) array_like
        Array-like containing the corners of the element.

    nodes_1 : array_like
        The first computational component for the element where the Jacobian should
        be evaluated.
    nodes_2 : array_like
        The second computational component for the element where the Jacobian should
        be evaluated.

    Returns
    -------
    j00 : array
        The :math:`(1, 1)` component of the Jacobian corresponding to the value of
        :math:`\frac{\partial x}{\partial \xi}`.

    j01 : array
        The :math:`(1, 2)` component of the Jacobian corresponding to the value of
        :math:`\frac{\partial y}{\partial \xi}`.

    j10 : array
        The :math:`(2, 1)` component of the Jacobian corresponding to the value of
        :math:`\frac{\partial x}{\partial \eta}`.

    j11 : array
        The :math:`(2, 2)` component of the Jacobian corresponding to the value of
        :math:`\frac{\partial y}{\partial \eta}`.
    """
    t0 = np.asarray(nodes_1)
    t1 = np.asarray(nodes_2)

    x0, y0 = corners[0, :]
    x1, y1 = corners[1, :]
    x2, y2 = corners[2, :]
    x3, y3 = corners[3, :]

    dx_dxi = np.astype(
        ((x1 - x0) * (1 - t1) + (x2 - x3) * (1 + t1)) / 4, np.float64, copy=False
    )
    dx_deta = np.astype(
        ((x3 - x0) * (1 - t0) + (x2 - x1) * (1 + t0)) / 4, np.float64, copy=False
    )
    dy_dxi = np.astype(
        ((y1 - y0) * (1 - t1) + (y2 - y3) * (1 + t1)) / 4, np.float64, copy=False
    )
    dy_deta = np.astype(
        ((y3 - y0) * (1 - t0) + (y2 - y1) * (1 + t0)) / 4, np.float64, copy=False
    )
    return ((dx_dxi, dy_dxi), (dx_deta, dy_deta))


def bilinear_interpolate(
    corner_vals: npt.NDArray[np.floating], xi: npt.ArrayLike, eta: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""Compute bilinear interpolation at (xi, eta) points.

    The relation for the bilinear interpolation is given by equation
    :eq:`bilinear-interpolation`, where :math:`u_0` is the bottom-left corner,
    :math:`u_1` the bottom right, :math:`u_2` the top right, and :math:`u_3` is
    the top left corner.

    .. math::
        :label: bilinear-interpolation

        u(\xi, \eta) = u_0 \cdot \frac{(1 - \xi)(1 - \eta)}{4}
        + u_1 \cdot \frac{(1 + \xi)(1 - \eta)}{4}
        + u_2 \cdot \frac{(1 + \xi)(1 + \eta)}{4}
        + u_3 \cdot \frac{(1 - \xi)(1 + \eta)}{4}

    Parameters
    ----------
    corner_vals : (4,) array_like
        Array with the values in the corners of the element.

    xi : array_like
        Array of the xi-coordinates on the reference domain where the
        x-coordinate should be evaluated.

    eta : array_like
        Array of the eta-coordinates on the reference domain where the
        x-coordinate should be evaluated.

    Returns
    -------
    array
        Array of interpolated values at the given xi and eta points.
    """
    t0 = np.asarray(xi)
    t1 = np.asarray(eta)
    b11 = (1 - t0) / 2
    b12 = (1 + t0) / 2
    b21 = (1 - t1) / 2
    b22 = (1 + t1) / 2
    return np.astype(
        (corner_vals[0] * b11 + corner_vals[1] * b12) * b21
        + (corner_vals[3] * b11 + corner_vals[2] * b12) * b22,
        np.float64,
        copy=False,
    )


def element_dual_dofs(
    order: UnknownFormOrder,
    element_cache: ElementMassMatrixCache,
    function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    r"""Compute the dual degrees of freedom (projection) of the function on the element.

    Dual projection of a function is computed by taking an integral over the
    reference domain of the function with each of the two dimensional basis.
    This also means that these integrals are the values of the :math:`L^2`
    projection of the function on the element.

    For 0-forms are quite straight forward:

    .. math::

        \left(f(x, y), \psi^{(0)}_i \right)_\Omega = \int_{\Omega} f(x, y)
        \psi^{(0)}_i(x, y) {dx} \wedge {dy} =
        \int_{\bar{\Omega}} f(x(\xi, \eta), y(\xi, \eta)) \psi^{(0)}_i(x(\xi, \eta),
        y(\xi, \eta))
        \left| \mathbf{J} \right| d\xi \wedge d\eta


    Similarly, 2-forms are also straight forward:

    .. math::

        \left(f(x, y), \psi^{(2)}_i \right)_\Omega = \int_{\Omega} f(x, y)
        \psi^{(2)}_i(x, y) {dx} \wedge {dy} =
        \int_{\bar{\Omega}} f(x(\xi, \eta), y(\xi, \eta)) \psi^{(2)}_i(x(\xi, \eta),
        y(\xi, \eta))
        \left| \mathbf{J} \right|^{-1} d\xi \wedge d\eta

    Lastly, for 1-forms it is a bit more involved, since it has multiple components:

    .. math::

        \left(f_x dy - f_y dx, {\psi^{(1)}_i}_x dy - {\psi^{(1)}_i}_y dx \right)_\Omega =
        \int_{\Omega} f_x {\psi^{(1)}_i}_x + f_y {\psi^{(1)}_i}_y {dx} \wedge {dy} =
        \int_{\bar{\Omega}} \left( \mathbf{J} \vec{f} \right) \cdot \left(
        \mathbf{J} \vec{\psi^{(1)}} \right) d\xi \wedge d\eta

    Parameters
    ----------
    order : UnknownFormOrder
        Order of the differential form to use for basis.

    corners : array_like
        Array of corners of the element.

    basis : Basis2D
        Basis function to use for the element.

    function : Callable
        Function to project.

    Returns
    -------
    array
        Array with dual degrees of freedom.
    """
    corners = element_cache.corners
    basis = element_cache.basis_2d
    ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) = corners
    nds_xi = basis.basis_xi.rule.nodes[None, :]
    nds_eta = basis.basis_eta.rule.nodes[:, None]

    ((j00, j01), (j10, j11)) = jacobian(corners, nds_xi, nds_eta)
    det = j00 * j11 - j10 * j01

    nds_x = bilinear_interpolate(np.array((x0, x1, x2, x3), np.float64), nds_xi, nds_eta)
    nds_y = bilinear_interpolate(np.array((y0, y1, y2, y3), np.float64), nds_xi, nds_eta)

    fv = np.array(function(nds_x, nds_y))
    weights = basis.basis_xi.rule.weights[None, :] * basis.basis_eta.rule.weights[:, None]

    dofs: list[float] = list()
    if order == UnknownFormOrder.FORM_ORDER_0:
        f_vals = fv * weights
        f_nod = f_vals * det
        for j in range(basis.basis_eta.order + 1):
            b2 = basis.basis_eta.node[j, ...]
            for i in range(basis.basis_xi.order + 1):
                b1 = basis.basis_xi.node[i, ...]
                v = np.sum(b1[None, ...] * b2[..., None] * f_nod)
                dofs.append(v)

    elif order == UnknownFormOrder.FORM_ORDER_1:
        f_vals = fv * weights[..., None]

        f_xi = j00 * f_vals[..., 0] + j01 * f_vals[..., 1]
        f_eta = j10 * f_vals[..., 0] + j11 * f_vals[..., 1]

        for j in range(basis.basis_eta.order + 1):
            b2 = basis.basis_eta.node[j, ...]
            for i in range(basis.basis_xi.order):
                b1 = basis.basis_xi.edge[i, ...]
                v = np.sum(b1[None, ...] * b2[..., None] * f_eta)
                dofs.append(v)

        for j in range(basis.basis_eta.order):
            b2 = basis.basis_eta.edge[j, ...]
            for i in range(basis.basis_xi.order + 1):
                b1 = basis.basis_xi.node[i, ...]
                v = np.sum(b1[None, ...] * b2[..., None] * f_xi)
                dofs.append(v)

    elif order == UnknownFormOrder.FORM_ORDER_2:
        f_vals = fv * weights

        for j in range(basis.basis_eta.order):
            b2 = basis.basis_eta.edge[j, ...]
            for i in range(basis.basis_xi.order):
                b1 = basis.basis_xi.edge[i, ...]
                v = np.sum(b1[None, ...] * b2[..., None] * f_vals)
                dofs.append(v)

    else:
        raise ValueError(f"Invalid form order {order}.")

    return np.array(dofs, np.float64)


def element_primal_dofs(
    order: UnknownFormOrder,
    element_cache: ElementMassMatrixCache,
    function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    r"""Compute the primal degrees of freedom of projection of a function on the element.

    Primal degrees of freedom allow for reconstruction of the function's :math:`L^2`
    projection. This means that given these degrees of freedom, the following relation
    holds:

    .. math::

        \left(\psi^{(k)}, f^{(k)} \right)_\Omega = \left(\psi^{(k)}, \bar{f}^{(k)} \right)
        = \int_{\Omega} \psi^{(k)} \sum\left( f_i \psi^{(k)}_i \right) dx \wedge dy

    for all basis functions :math:`\psi^{(k)}`.

    Parameters
    ----------
    order : UnknownFormOrder
        Order of the differential form to use for basis.

    corners : array_like
        Array of corners of the element.

    basis : Basis2D
        Basis function to use for the element.

    function : Callable
        Function to project.

    Returns
    -------
    array
        Array with primal degrees of freedom.
    """
    dofs = element_dual_dofs(order, element_cache, function)
    mat = element_cache.mass_from_order(order, inverse=True)
    assert np.allclose(
        mat, np.linalg.inv(element_cache.mass_from_order(order, inverse=False))
    )

    return np.astype(mat @ dofs, np.float64, copy=False)


def reconstruct(
    corners: npt.NDArray[np.floating],
    k: UnknownFormOrder,
    dofs: npt.ArrayLike,
    xi: npt.ArrayLike,
    eta: npt.ArrayLike,
    basis: Basis2D,
) -> npt.NDArray[np.float64]:
    """Reconstruct a k-form on the element from its primal degrees of freedom.

    Parameters
    ----------
    corners : array_like
        Array of corners of the element.

    k : UnknownFormOrder
        Order of the differential form to use for basis.

    dofs : array_like
        Degrees of freedom of the k-form.

    xi : array_like
        Coordinates of the xi-coordinate in the reference domain.

    eta : array_like
        Coordinates of the eta-coordinate in the reference domain.

    basis : Basis2D
        Basis function to use for the element.

    Returns
    -------
    array
        Array with the point values of the k-form at the specified coordinates.
    """
    k = UnknownFormOrder(k)
    out: float | npt.NDArray[np.floating] = 0.0
    c = np.asarray(dofs, dtype=np.float64, copy=None)
    if c.ndim != 1:
        raise ValueError("Coefficient array must be one dimensional.")

    if k == UnknownFormOrder.FORM_ORDER_0:
        assert c.size == (basis.basis_xi.order + 1) * (basis.basis_eta.order + 1)
        vals_xi = lagrange1d(basis.basis_xi.roots, xi)
        vals_eta = lagrange1d(basis.basis_eta.roots, eta)
        for i in range(basis.basis_eta.order + 1):
            v = vals_eta[..., i]
            for j in range(basis.basis_xi.order + 1):
                u = vals_xi[..., j]
                out += c[i * (basis.basis_xi.order + 1) + j] * (u * v)

    elif k == UnknownFormOrder.FORM_ORDER_1:
        assert c.size == (
            basis.basis_eta.order + 1
        ) * basis.basis_xi.order + basis.basis_eta.order * (basis.basis_xi.order + 1)
        values_xi = lagrange1d(basis.basis_xi.roots, xi)
        values_eta = lagrange1d(basis.basis_eta.roots, eta)
        in_dvalues_xi = dlagrange1d(basis.basis_xi.roots, xi)
        in_dvalues_eta = dlagrange1d(basis.basis_eta.roots, eta)
        dvalues_xi = tuple(
            accumulate(-in_dvalues_xi[..., i] for i in range(basis.basis_xi.order))
        )
        dvalues_eta = tuple(
            accumulate(-in_dvalues_eta[..., i] for i in range(basis.basis_eta.order))
        )
        (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
        det = j00 * j11 - j10 * j01
        out_xi: float | npt.NDArray[np.floating] = 0.0
        out_eta: float | npt.NDArray[np.floating] = 0.0
        for i1 in range(basis.basis_eta.order + 1):
            v1 = values_eta[..., i1]
            for j1 in range(basis.basis_xi.order):
                u1 = dvalues_xi[j1]
                out_eta += c[i1 * basis.basis_xi.order + j1] * u1 * v1

        for i1 in range(basis.basis_eta.order):
            v1 = dvalues_eta[i1]
            for j1 in range(basis.basis_xi.order + 1):
                u1 = values_xi[..., j1]

                out_xi += (
                    c[
                        (basis.basis_eta.order + 1) * basis.basis_xi.order
                        + i1 * (basis.basis_xi.order + 1)
                        + j1
                    ]
                    * u1
                    * v1
                )
        out = np.stack(
            (out_xi * j00 + out_eta * j10, out_xi * j01 + out_eta * j11), axis=-1
        )
        out /= det[..., None]

    elif k == UnknownFormOrder.FORM_ORDER_2:
        assert c.size == basis.basis_xi.order * basis.basis_eta.order
        in_dvalues_xi = dlagrange1d(basis.basis_xi.roots, xi)
        in_dvalues_eta = dlagrange1d(basis.basis_eta.roots, eta)
        dvalues_xi = tuple(
            accumulate(-in_dvalues_xi[..., i] for i in range(basis.basis_xi.order))
        )
        dvalues_eta = tuple(
            accumulate(-in_dvalues_eta[..., i] for i in range(basis.basis_eta.order))
        )
        (j00, j01), (j10, j11) = jacobian(corners, xi, eta)
        det = j00 * j11 - j10 * j01
        for i1 in range(basis.basis_eta.order):
            v1 = dvalues_eta[i1]
            for j1 in range(basis.basis_xi.order):
                u1 = dvalues_xi[j1]
                out += c[i1 * basis.basis_xi.order + j1] * u1 * v1

        out /= det
    else:
        raise ValueError(f"Order of the differential form {k} is not valid.")

    return np.array(out, np.float64, copy=None)
