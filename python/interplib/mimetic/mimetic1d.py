"""Prototypes of Mimetic operations."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from itertools import accumulate

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D
from interplib._mimetic import Line, Manifold1D


class Mesh1D:
    """Mainfold with geometry and basis attached."""

    primal: Manifold1D
    dual: Manifold1D
    positions: npt.NDArray[np.float64]  # 1D array of positions
    element_orders: npt.NDArray[np.uint8]

    def __init__(
        self,
        positions: npt.ArrayLike,
        element_order: int | Sequence[int] | npt.NDArray[np.integer],
    ) -> None:
        self.positions = np.array(positions, np.float64)
        if self.positions.ndim != 1:
            raise ValueError(
                "The dimension of the positions array was not 1 but "
                f"{self.positions.ndim}."
            )
        self.primal = Manifold1D.line_mesh(self.positions.size - 1)
        self.dual = self.primal.compute_dual()
        n_elem = self.primal.n_lines
        order = np.array(element_order, np.uint8)
        if order.ndim == 0:
            self.element_orders = np.full(n_elem, order)
        elif order.ndim == 1 and order.size == n_elem:
            self.element_orders = order
        else:
            raise ValueError(
                "The element order should either be an integer or an array with the same"
                f" length as the number of elements (expected ({n_elem},) but got "
                f"{order.shape} instead)."
            )

    def get_element(self, index: int) -> Element1D:
        """Return the element at specified index."""
        assert index >= 0
        line = self.primal.get_line(index + 1)
        return Element1D(
            self.element_orders[index],
            self.positions[line.begin.index],
            self.positions[line.end.index],
        )

    def get_dual(self, index: int) -> Line:
        """Return the dual of an element at specified index."""
        assert index >= 0
        return self.dual.get_line(index + 1)


@cache
def _reference_matrices_1d(
    p: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the mass matrices on a reference element."""
    nodes = np.astype(np.linspace(-1.0, +1.0, p + 1), np.float64)
    node_basis = Polynomial1D.lagrange_nodal_basis(nodes)
    edge_basis = tuple(accumulate(basis.derivative for basis in node_basis))

    mass_node = np.empty((p + 1, p + 1), np.float64)
    for i in range(p + 1):
        for j in range(i + 1):
            inner = node_basis[i] * node_basis[j]
            anti = inner.antiderivative
            mass_node[i, j] = mass_node[j, i] = anti(+1) - anti(-1)
    mass_edge = np.empty((p, p), np.float64)
    for i in range(p):
        for j in range(i + 1):
            inner = edge_basis[i] * edge_basis[j]
            anti = inner.antiderivative
            mass_edge[i, j] = mass_edge[j, i] = anti(+1) - anti(-1)
    return mass_node, mass_edge


@cache
def _cached_element_nodes(p: int) -> npt.NDArray[np.float64]:
    """Get cached element nodes."""
    return np.astype(np.linspace(-1.0, +1.0, p + 1), np.float64)


class Element1D:
    """An element of a mesh in 1D."""

    xleft: float
    xright: float
    nodes: npt.NDArray[np.float64]
    order: int
    _node_basis: tuple[Polynomial1D, ...] | None
    _edge_basis: tuple[Polynomial1D, ...] | None
    jacobian: float
    mass_node: npt.NDArray[np.floating]
    mass_edge: npt.NDArray[np.floating]

    def __init__(self, p: int, x0: float, x1: float) -> None:
        self.order = p
        self.jacobian = (x1 - x0) / 2.0
        self.nodes = _cached_element_nodes(p)
        self._node_basis = None
        self._edge_basis = None
        self.xleft = x0
        self.xright = x1
        mat_n, mat_e = _reference_matrices_1d(p)
        self.mass_node = mat_n * self.jacobian
        self.mass_edge = mat_e / self.jacobian

    @property
    def node_basis(self) -> tuple[Polynomial1D, ...]:
        """Compute nodal basis if explicitly asked for."""
        if self._node_basis is None:
            self._node_basis = Polynomial1D.lagrange_nodal_basis(self.nodes)
        return self._node_basis

    @property
    def edge_basis(self) -> tuple[Polynomial1D, ...]:
        """Compute edge (and nodal) basis if explicitly asked for."""
        if self._edge_basis is None:
            self._edge_basis = tuple(
                accumulate(
                    -(1.0 / self.jacobian) * basis.derivative for basis in self.node_basis
                )
            )
        return self._edge_basis

    def incidence_primal_0(
        self, mat: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        r"""Apply exterior derivative operation to the 0-forms on the primal mesh.

        Parameters
        ----------
        mat : array_like, optional
            Matrix on which to apply the operation on. If not given, the output will
            be the actual incidence matrix.

        Returns
        -------
        array
            Returns the result of applying :math:`\mathbb{E}^{0,1}` on the matrix given
            by `mat`, or just :math:`\mathbb{E}^{0,1}` if not specified.
        """
        out: npt.NDArray
        n_in = self.order + 1
        n_out = self.order
        if mat is not None:
            m = np.asarray(mat)
            out = np.zeros((n_out, n_in), m.dtype)
            out[:, :-1] -= m
            out[:, +1:] += m
        else:
            out = np.zeros((n_out, n_in), np.int8)
            write = np.reshape(out, (-1,))
            write[0 : -1 : n_in + 1] = -1
            write[1 :: n_in + 1] = +1
        return out
