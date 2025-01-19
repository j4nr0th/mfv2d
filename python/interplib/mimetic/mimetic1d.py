"""Prototypes of Mimetic operations."""

from __future__ import annotations

from collections.abc import Sequence
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


class Element1D:
    """An element of a mesh in 1D."""

    xleft: float
    xright: float
    nodes: npt.NDArray[np.float64]
    order: int
    node_basis: tuple[Polynomial1D, ...]
    edge_basis: tuple[Polynomial1D, ...]
    jacobian: float

    def __init__(self, p: int, x0: float, x1: float) -> None:
        self.order = p
        self.jacobian = (x1 - x0) / 2.0
        self.nodes = np.astype(np.linspace(-1.0, +1.0, p + 1), np.float64)
        self.node_basis = Polynomial1D.lagrange_nodal_basis(self.nodes)
        self.edge_basis = tuple(
            accumulate(
                -(1.0 / self.jacobian) * basis.derivative for basis in self.node_basis
            )
        )
        self.xleft = x0
        self.xright = x1

    @property
    def mass_node(self) -> npt.NDArray[np.float64]:
        """Compute mass matrix of nodal basis functions."""
        matrix = np.empty((self.order + 1, self.order + 1), np.float64)
        for i in range(self.order + 1):
            for j in range(i + 1):
                inner = self.node_basis[i] * self.node_basis[j]
                anti = inner.antiderivative
                matrix[i, j] = matrix[j, i] = anti(+1) - anti(-1)
        return np.astype(matrix * self.jacobian, np.float64)

    @property
    def mass_edge(self) -> npt.NDArray[np.float64]:
        """Compute mass matrix of edge basis functions."""
        matrix = np.empty((self.order, self.order), np.float64)
        for i in range(self.order):
            for j in range(i + 1):
                inner = self.edge_basis[i] * self.edge_basis[j]
                anti = inner.antiderivative
                matrix[i, j] = matrix[j, i] = anti(+1) - anti(-1)
        return np.astype(matrix * self.jacobian, np.float64)

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
