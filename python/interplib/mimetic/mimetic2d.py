"""Implementation of the 2D mimetic meshes and manifolds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import numpy.typing as npt
import pyvista as pv

from interplib._interp import compute_gll, dlagrange1d, lagrange1d
from interplib._mimetic import Manifold2D
from interplib.kforms.eval import (
    Identity,
    Incidence,
    MassMat,
    MatMul,
    MatOp,
    Push,
    Scale,
    Sum,
    Transpose,
)
from interplib.kforms.kform import (
    KBoundaryProjection,
    KElementProjection,
    KExplicit,
    KFormSystem,
    KWeight,
)


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


def vtk_lagrange_ordering(order: int) -> npt.NDArray[np.int32]:
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
    return (
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
        )
    )


class BasisCache:
    """Cache for basis evaluation to allow for faster evaluation of mass matrices."""

    basis_order: int
    integration_order: int
    nodal_1d: npt.NDArray[np.double]
    edge_1d: npt.NDArray[np.double]
    nodes_1d: npt.NDArray[np.float64]
    int_nodes_1d: npt.NDArray[np.float64]
    int_weights_1d: npt.NDArray[np.float64]
    _precomp_node: npt.NDArray[np.float64] | None = None
    _precomp_edge: npt.NDArray[np.float64] | None = None
    _precomp_surf: npt.NDArray[np.float64] | None = None
    _precomp_mix01: npt.NDArray[np.float64] | None = None
    _precomp_mix12: npt.NDArray[np.float64] | None = None

    def __init__(self, basis_order: int, integration_order: int, /) -> None:
        self.basis_order = int(basis_order)
        self.integration_order = int(integration_order)
        n, _ = compute_gll(self.basis_order)
        self.nodes_1d = n
        ni, wi = compute_gll(self.integration_order)
        self.int_nodes_1d = ni
        self.int_weights_1d = wi
        self.nodal_1d = lagrange1d(self.nodes_1d, self.int_nodes_1d)
        dvals = dlagrange1d(self.nodes_1d, self.int_nodes_1d)
        self.edge_1d = np.cumsum(-dvals[..., :-1], axis=-1)
        self.int_weights_2d = wi[:, None] * wi[None, :]

    @property
    def mass_node_precomp(self) -> npt.NDArray[np.float64]:
        """Precomputed entries for nodal mass matrix which just need to be scaled."""
        if self._precomp_node is not None:
            return self._precomp_node

        n = self.basis_order + 1
        m = self.integration_order + 1
        mat = np.empty((n**2, n**2, m, m), np.float64)
        values = self.nodal_1d
        weights_2d = self.int_weights_2d

        basis_vals: list[npt.NDArray] = list()

        for i1 in range(n):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)

        for i in range(n * n):
            for j in range(i + 1):
                res = basis_vals[i] * basis_vals[j] * weights_2d
                mat[i, j, ...] = mat[j, i, ...] = res

        self._precomp_node = mat
        return mat

    @property
    def mass_edge_precomp(self) -> npt.NDArray[np.float64]:
        """Precomputed entries for edge mass matrix which just need to be scaled."""
        if self._precomp_edge is not None:
            return self._precomp_edge

        n = self.basis_order
        m = self.integration_order + 1
        mat = np.empty((2 * n * (n + 1), 2 * n * (n + 1), m, m), np.float64)
        values = self.nodal_1d
        dvalues = self.edge_1d
        weights_2d = self.int_weights_2d

        basis_eta: list[npt.NDArray] = list()
        basis_xi: list[npt.NDArray] = list()

        for i1 in range(n + 1):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = dvalues[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_eta.append(basis1)

        for i1 in range(n):
            v1 = dvalues[..., i1]
            for j1 in range(n + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_xi.append(basis1)

        nb = n * (n + 1)

        for i in range(nb):
            for j in range(i + 1):
                res = weights_2d * basis_eta[i] * basis_eta[j]
                mat[i, j, ...] = mat[j, i, ...] = res

        for i in range(nb):
            for j in range(i + 1):
                res = weights_2d * basis_xi[i] * basis_xi[j]
                mat[nb + i, nb + j, ...] = mat[nb + j, nb + i, ...] = res

        for i in range(nb):
            for j in range(nb):
                res = weights_2d * basis_eta[j] * basis_xi[i]
                mat[nb + i, j, ...] = mat[j, nb + i, ...] = res

        self._precomp_edge = mat
        return mat

    @property
    def mass_surf_precomp(self) -> npt.NDArray[np.float64]:
        """Precomputed entries for surface mass matrix which just need to be scaled."""
        if self._precomp_surf is not None:
            return self._precomp_surf

        n = self.basis_order
        m = self.integration_order + 1
        mat = np.empty((n**2, n**2, m, m), np.float64)
        values = self.edge_1d
        weights_2d = self.int_weights_2d

        basis_vals: list[npt.NDArray] = list()

        for i1 in range(n):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)

        for i in range(n * n):
            for j in range(i + 1):
                res = basis_vals[i] * basis_vals[j] * weights_2d
                mat[i, j, ...] = mat[j, i, ...] = res

        self._precomp_surf = mat
        return mat

    @property
    def mass_mix01_precomp(self) -> npt.NDArray[np.float64]:
        """Pre-computed products of 0-form and 1-form basis."""
        if self._precomp_mix01 is not None:
            return self._precomp_mix01

        ndl = self.nodal_1d.T
        edg = self.edge_1d.T

        basis_node = np.reshape(
            ndl[None, :, None, :] * ndl[:, None, :, None],
            (
                (self.basis_order + 1) ** 2,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )
        basis_edge_eta = np.reshape(
            edg[None, :, None, :] * ndl[:, None, :, None],
            (
                (self.basis_order + 1) * self.basis_order,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )
        basis_edge_xi = np.reshape(
            ndl[None, :, None, :] * edg[:, None, :, None],
            (
                (self.basis_order + 1) * self.basis_order,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )

        mat = np.concatenate(
            (
                basis_edge_eta[None, :, :, :] * basis_node[:, None, :, :],
                basis_edge_xi[None, :, :, :] * basis_node[:, None, :, :],
            ),
            axis=1,
        )
        mat *= self.int_weights_2d[None, None, :, :]
        self._precomp_mix01 = mat

        return mat

    @property
    def mass_mix12_precomp(self) -> npt.NDArray[np.float64]:
        """Pre-computed products of 1-form and 2-form basis."""
        if self._precomp_mix12 is not None:
            return self._precomp_mix12

        ndl = self.nodal_1d.T
        edg = self.edge_1d.T

        basis_edge_eta = np.reshape(
            edg[None, :, None, :] * ndl[:, None, :, None],
            (
                (self.basis_order + 1) * self.basis_order,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )
        basis_edge_xi = np.reshape(
            ndl[None, :, None, :] * edg[:, None, :, None],
            (
                (self.basis_order + 1) * self.basis_order,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )
        basis_surf = np.reshape(
            edg[None, :, None, :] * edg[:, None, :, None],
            (
                self.basis_order**2,
                self.integration_order + 1,
                self.integration_order + 1,
            ),
        )

        mat = np.concatenate(
            (
                basis_edge_eta[:, None, :, :] * basis_surf[None, :, :, :],
                basis_edge_xi[:, None, :, :] * basis_surf[None, :, :, :],
            ),
            axis=0,
        )
        mat *= self.int_weights_2d[None, None, :, :]
        self._precomp_mix12 = mat

        return mat

    def c_serialization(
        self,
    ) -> tuple[
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
    ]:
        """Serialize into form which C code understands."""
        edge_pre_comp = self.mass_edge_precomp
        m = self.basis_order * (self.basis_order + 1)
        return (
            self.basis_order,
            self.integration_order + 1,
            self.int_nodes_1d,
            self.mass_node_precomp,
            np.ascontiguousarray(edge_pre_comp[0 * m : 1 * m, 0 * m : 1 * m, ...]),
            np.ascontiguousarray(edge_pre_comp[1 * m : 2 * m, 0 * m : 1 * m, ...]),
            np.ascontiguousarray(edge_pre_comp[1 * m : 2 * m, 1 * m : 2 * m, ...]),
            self.mass_surf_precomp,
            self.mass_mix01_precomp,
            self.mass_mix12_precomp,
        )

    def clean(self) -> None:
        """Clear the pre-computes, which are the majority of the memory."""
        self._precomp_node = None
        self._precomp_edge = None
        self._precomp_surf = None


@dataclass(frozen=True, eq=False)
class Element2D:
    """General 2D element."""

    parent: ElementNode2D | None

    def order_on_side(self, side: int) -> int:
        """Effective order of the element on the specified side."""
        raise NotImplementedError

    def dof_sizes(self, form_orders: Sequence[int]) -> tuple[int, ...]:
        """Compute number unknown DoFs for differential forms."""
        raise NotImplementedError

    def dof_offsets(self, form_orders: Sequence[int]) -> tuple[int, ...]:
        """Compute offset of unknown DoFs."""
        sizes = self.dof_sizes(form_orders)
        return (0, *accumulate(sizes))

    def element_edge_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of edge DoFs on the boundary of an element."""
        raise NotImplementedError

    def element_node_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of node DoFs on the boundary of an element."""
        raise NotImplementedError

    def total_dof_count(
        self, form_orders: Sequence[int], lagrange: bool, children: bool
    ) -> int:
        """Return the number of all DoFs needed by the element.

        Lagrange multipliers involved in the continuity between child and parent elements
        can included in this.

        Parameters
        ----------
        form_orders : Sequence of int
            Orders of differential forms defined in the element.
        lagrange : bool
            Include Lagrange multiplier related degrees of freedom.
        children : bool
            Include all child degrees of freedom.

        Returns
        -------
        int
            Number of specified degrees of freedom.
        """
        raise NotImplementedError

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Coordinates of the bottom left corner."""
        raise NotImplementedError

    @property
    def bottom_right(self) -> tuple[float, float]:
        """Coordinates of the bottom right corner."""
        raise NotImplementedError

    @property
    def top_right(self) -> tuple[float, float]:
        """Coordinates of the top right corner."""
        raise NotImplementedError

    @property
    def top_left(self) -> tuple[float, float]:
        """Coordinates of the top left corner."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class ElementNode2D(Element2D):
    """Two dimensional element that contains children."""

    child_bl: Element2D
    child_br: Element2D
    child_tl: Element2D
    child_tr: Element2D

    maximum_order: int | None = None

    def order_on_side(self, side: int) -> int:
        """Effective order of the element on the specified side."""
        if side == 0:
            base_size = self.child_bl.order_on_side(0) + self.child_br.order_on_side(0)
        elif side == 1:
            base_size = self.child_br.order_on_side(1) + self.child_tr.order_on_side(1)
        elif side == 2:
            base_size = self.child_tr.order_on_side(2) + self.child_tl.order_on_side(2)
        elif side == 3:
            base_size = self.child_tl.order_on_side(3) + self.child_bl.order_on_side(3)
        else:
            raise ValueError(f"Invalid value of the side (can not be {side}).")

        if self.maximum_order is not None:
            base_size = min(self.maximum_order, base_size)

        return base_size

    def dof_sizes(self, form_orders: Sequence[int]) -> tuple[int, ...]:
        """Compute number unknown DoFs for differential forms."""
        sizes: list[int] = list()
        for form_order in form_orders:
            if form_order == 2:
                n = 0
            elif form_order == 1 or form_order == 0:
                n = sum(self.order_on_side(i_side) for i_side in range(4))
            else:
                raise ValueError(f"Invalid differential form order {form_order}.")
            sizes.append(n)

        return tuple(sizes)

    def total_dof_count(
        self, form_orders: Sequence[int], lagrange: bool, children: bool
    ) -> int:
        """Return the number of all DoFs needed by the element.

        Lagrange multipliers involved in the continuity between child and parent elements
        can included in this.

        Parameters
        ----------
        form_orders : Sequence of int
            Orders of differential forms defined in the element.
        lagrange : bool
            Include Lagrange multiplier related degrees of freedom.
        children : bool
            Include all child degrees of freedom.

        Returns
        -------
        int
            Number of specified degrees of freedom.
        """
        n_lagrange = 0
        if lagrange:
            for order in form_orders:
                if order == 2:
                    continue

                # There's always the same number of parent-child as the order of the child
                # on that boundary
                n_btm = self.child_bl.order_on_side(0) + self.child_br.order_on_side(0)
                n_rth = self.child_br.order_on_side(1) + self.child_tr.order_on_side(1)
                n_top = self.child_tr.order_on_side(2) + self.child_tl.order_on_side(2)
                n_lft = self.child_tl.order_on_side(3) + self.child_bl.order_on_side(3)

                n_lagrange += n_btm + n_rth + n_top + n_lft

                n_bl_br = max(
                    self.child_bl.order_on_side(1), self.child_br.order_on_side(3)
                )
                n_br_tr = max(
                    self.child_br.order_on_side(2), self.child_tr.order_on_side(0)
                )
                n_tr_tl = max(
                    self.child_tr.order_on_side(3), self.child_tl.order_on_side(1)
                )
                n_tl_bl = max(
                    self.child_tl.order_on_side(0), self.child_bl.order_on_side(2)
                )

                n_lagrange += n_bl_br + n_br_tr + n_tr_tl + n_tl_bl

                if order == 0:
                    # Add the center node connectivity relations (but not cyclical)
                    n_lagrange += 3

        count = sum(self.dof_sizes(form_orders)) + n_lagrange

        if children:
            count += (
                self.child_bl.total_dof_count(form_orders, lagrange, children)
                + self.child_br.total_dof_count(form_orders, lagrange, children)
                + self.child_tl.total_dof_count(form_orders, lagrange, children)
                + self.child_tr.total_dof_count(form_orders, lagrange, children)
            )

        return count

    def element_edge_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of edge DoFs on the boundary of an element."""
        offset = sum(self.order_on_side(i_side) for i_side in range(bnd_idx))
        count = self.order_on_side(bnd_idx)

        return np.astype(
            offset + np.arange(count, dtype=np.uint32), np.uint32, copy=False
        )

    def element_node_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of node DoFs on the boundary of an element."""
        offset = sum(self.order_on_side(i_side) for i_side in range(bnd_idx))
        count = self.order_on_side(bnd_idx) + 1

        res = np.astype(offset + np.arange(count, dtype=np.uint32), np.uint32, copy=False)

        if bnd_idx == 3:
            # This node is shared between left and bottom side
            res[-1] = 0

        return res

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Coordinates of the bottom left corner."""
        return self.child_bl.bottom_left

    @property
    def bottom_right(self) -> tuple[float, float]:
        """Coordinates of the bottom right corner."""
        return self.child_br.bottom_right

    @property
    def top_right(self) -> tuple[float, float]:
        """Coordinates of the top right corner."""
        return self.child_tr.top_right

    @property
    def top_left(self) -> tuple[float, float]:
        """Coordinates of the top left corner."""
        return self.child_tl.top_left


@dataclass(frozen=True, eq=False)
class ElementLeaf2D(Element2D):
    """Two dimensional square element.

    This type facilitates operations related to calculations which need
    to be carried out on the reference element itself, such as calculation
    of the mass and incidence matrices, as well as the reconstruction of
    the solution.

    Parameters
    ----------
    p : int
        Order of the basis functions used for the nodal basis.
    bl : (float, float)
        Coordinates of the bottom left corner.
    br : (float, float)
        Coordinates of the bottom right corner.
    tr : (float, float)
        Coordinates of the top right corner.
    tl : (float, float)
        Coordinates of the top left corner.
    """

    order: int

    _bottom_left: tuple[float, float]
    _bottom_right: tuple[float, float]
    _top_right: tuple[float, float]
    _top_left: tuple[float, float]

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Coordinates of the bottom left corner."""
        return self._bottom_left

    @property
    def bottom_right(self) -> tuple[float, float]:
        """Coordinates of the bottom right corner."""
        return self._bottom_right

    @property
    def top_right(self) -> tuple[float, float]:
        """Coordinates of the top right corner."""
        return self._top_right

    @property
    def top_left(self) -> tuple[float, float]:
        """Coordinates of the top left corner."""
        return self._top_left

    def element_node_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of node DoFs on the boundary of an element."""
        n = self.order

        if bnd_idx == 0:
            return np.arange(0, n + 1, dtype=np.uint32)

        elif bnd_idx == 1:
            return np.astype(
                n + np.arange(0, n + 1, dtype=np.uint32) * (n + 1),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 2:
            return np.astype(
                np.flip((n + 1) * n + np.arange(0, n + 1, dtype=np.uint32)),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 3:
            return np.astype(
                np.flip(np.arange(0, n + 1, dtype=np.uint32) * (n + 1)),
                np.uint32,
                copy=False,
            )

        raise ValueError("Only boundary ID of up to 3 is allowed.")

    def element_edge_dofs(self, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of edge DoFs on the boundary of an element."""
        n = self.order
        if bnd_idx == 0:
            return np.arange(0, n, dtype=np.uint32)

        elif bnd_idx == 1:
            return np.astype(
                (n * (n + 1)) + n + np.arange(0, n, dtype=np.uint32) * (n + 1),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 2:
            return np.astype(
                np.flip(n * n + np.arange(0, n, dtype=np.uint32)),
                np.uint32,
                copy=False,
            )

        elif bnd_idx == 3:
            return np.astype(
                np.flip((n * (n + 1)) + np.arange(0, n, dtype=np.uint32) * (n + 1)),
                np.uint32,
                copy=False,
            )

        raise ValueError("Only boundary ID of up to 3 is allowed.")

    def order_on_side(self, side: int) -> int:
        """Effective order of the element on the specified side."""
        if side < 0 or side >= 4:
            raise ValueError(f"Side index can not be {side}.")
        return self.order

    def dof_sizes(self, form_orders: Sequence[int]) -> tuple[int, ...]:
        """Compute number unknown DoFs for differential forms."""
        sizes: list[int] = list()
        for form_order in form_orders:
            if form_order == 2:
                n = self.order**2
            elif form_order == 1:
                n = self.order * (self.order + 1) * 2
            elif form_order == 0:
                n = (self.order + 1) ** 2
            else:
                raise ValueError(f"Invalid differential form order {form_order}.")
            sizes.append(n)

        return tuple(sizes)

    def total_dof_count(
        self, form_orders: Sequence[int], lagrange: bool, children: bool
    ) -> int:
        """Return the number of all DoFs needed by the element.

        Lagrange multipliers involved in the continuity between child and parent elements
        can included in this.

        Parameters
        ----------
        form_orders : Sequence of int
            Orders of differential forms defined in the element.
        lagrange : bool
            Include Lagrange multiplier related degrees of freedom.
        children : bool
            Include all child degrees of freedom.

        Returns
        -------
        int
            Number of specified degrees of freedom.
        """
        del lagrange, children
        return sum(self.dof_sizes(form_orders))

    def poly_x(self, xi: npt.ArrayLike, eta: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute the x-coordiate of (xi, eta) points."""
        t0 = np.asarray(xi)
        t1 = np.asarray(eta)
        b11 = (1 - t0) / 2
        b12 = (1 + t0) / 2
        b21 = (1 - t1) / 2
        b22 = (1 + t1) / 2
        return np.astype(
            (self._bottom_left[0] * b11 + self._bottom_right[0] * b12) * b21
            + (self._top_left[0] * b11 + self._top_right[0] * b12) * b22,
            np.float64,
            copy=False,
        )

    def poly_y(self, xi: npt.ArrayLike, eta: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute the y-coordiate of (xi, eta) points."""
        t0 = np.asarray(xi)
        t1 = np.asarray(eta)
        b11 = (1 - t0) / 2
        b12 = (1 + t0) / 2
        b21 = (1 - t1) / 2
        b22 = (1 + t1) / 2
        return np.astype(
            (self._bottom_left[1] * b11 + self._bottom_right[1] * b12) * b21
            + (self._top_left[1] * b11 + self._top_right[1] * b12) * b22,
            np.float64,
            copy=False,
        )

    def mass_matrix_node(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for nodal basis."""
        assert cache.basis_order == self.order
        precomp = cache.mass_node_precomp
        (j00, j01), (j10, j11) = self.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01

        # Does not use symmetry (yet)
        mat = np.sum(precomp * det[None, None, ...], axis=(-2, -1))

        return mat

    def mass_matrix_edge(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for mixed node-edge basis."""
        assert cache.basis_order == self.order
        precomp = cache.mass_edge_precomp
        (j00, j01), (j10, j11) = self.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01

        khh = j11**2 + j10**2
        kvv = j01**2 + j00**2
        kvh = j01 * j11 + j00 * j10
        khh = khh / det
        kvv = kvv / det
        kvh = kvh / det

        nb = self.order * (self.order + 1)
        mat = np.empty((2 * nb, 2 * nb), np.float64)

        mat[0 * nb : 1 * nb, 0 * nb : 1 * nb] = np.sum(
            precomp[0 * nb : 1 * nb, 0 * nb : 1 * nb, ...] * khh[None, None, ...],
            axis=(-2, -1),
        )
        mat[1 * nb : 2 * nb, 0 * nb : 1 * nb] = np.sum(
            precomp[1 * nb : 2 * nb, 0 * nb : 1 * nb, ...] * kvh[None, None, ...],
            axis=(-2, -1),
        )
        mat[0 * nb : 1 * nb, 1 * nb : 2 * nb] = np.sum(
            precomp[0 * nb : 1 * nb, 1 * nb : 2 * nb, ...] * kvh[None, None, ...],
            axis=(-2, -1),
        )
        mat[1 * nb : 2 * nb, 1 * nb : 2 * nb] = np.sum(
            precomp[1 * nb : 2 * nb, 1 * nb : 2 * nb, ...] * kvv[None, None, ...],
            axis=(-2, -1),
        )
        return mat

    def mass_matrix_surface(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for surface basis."""
        assert cache.basis_order == self.order
        precomp = cache.mass_surf_precomp
        (j00, j01), (j10, j11) = self.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01

        # Does not use symmetry (yet)
        mat = np.sum(precomp / det[None, None, ...], axis=(-2, -1))
        return mat

    def jacobian(
        self, xi: npt.ArrayLike, eta: npt.ArrayLike, /
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
        xi : array_like
            The first computational component for the element where the Jacobian should
            be evaluated.
        eta : array_like
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
        t0 = np.asarray(xi)
        t1 = np.asarray(eta)

        x0 = self._bottom_left[0]
        x1 = self._bottom_right[0]
        x2 = self._top_right[0]
        x3 = self._top_left[0]

        y0 = self._bottom_left[1]
        y1 = self._bottom_right[1]
        y2 = self._top_right[1]
        y3 = self._top_left[1]

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

    def reconstruct(
        self,
        k: int,
        coeffs: npt.ArrayLike,
        xi: npt.ArrayLike,
        eta: npt.ArrayLike,
        cache: BasisCache,
        /,
    ) -> npt.NDArray[np.float64]:
        """Reconstruct a k-form on the element."""
        assert k >= 0 and k < 3
        assert cache.basis_order == self.order
        out: float | npt.NDArray[np.floating] = 0.0
        c = np.asarray(coeffs, dtype=np.float64, copy=None)
        if c.ndim != 1:
            raise ValueError("Coefficient array must be one dimensional.")

        if k == 0:
            vals_xi = lagrange1d(cache.nodes_1d, xi)
            vals_eta = lagrange1d(cache.nodes_1d, eta)
            for i in range(self.order + 1):
                v = vals_eta[..., i]
                for j in range(self.order + 1):
                    u = vals_xi[..., j]
                    out += c[i * (self.order + 1) + j] * (u * v)

        elif k == 1:
            # TODO: check if reconstruction is done correctly on non-unit domain.
            values_xi = lagrange1d(cache.nodes_1d, xi)
            values_eta = lagrange1d(cache.nodes_1d, eta)
            in_dvalues_xi = dlagrange1d(cache.nodes_1d, xi)
            in_dvalues_eta = dlagrange1d(cache.nodes_1d, eta)
            dvalues_xi = tuple(
                accumulate(-in_dvalues_xi[..., i] for i in range(self.order))
            )
            dvalues_eta = tuple(
                accumulate(-in_dvalues_eta[..., i] for i in range(self.order))
            )
            (j00, j01), (j10, j11) = self.jacobian(xi, eta)
            det = j00 * j11 - j10 * j01
            out_xi: float | npt.NDArray[np.floating] = 0.0
            out_eta: float | npt.NDArray[np.floating] = 0.0
            for i1 in range(self.order + 1):
                v1 = values_eta[..., i1]
                for j1 in range(self.order):
                    u1 = dvalues_xi[j1]
                    out_eta += c[i1 * self.order + j1] * u1 * v1

            for i1 in range(self.order):
                v1 = dvalues_eta[i1]
                for j1 in range(self.order + 1):
                    u1 = values_xi[..., j1]

                    out_xi += (
                        c[(self.order + 1) * self.order + i1 * (self.order + 1) + j1]
                        * u1
                        * v1
                    )
            out = np.stack(
                (out_xi * j00 + out_eta * j10, out_xi * j01 + out_eta * j11), axis=-1
            )
            out /= det[..., None]

        elif k == 2:
            in_dvalues_xi = dlagrange1d(cache.nodes_1d, xi)
            in_dvalues_eta = dlagrange1d(cache.nodes_1d, eta)
            dvalues_xi = tuple(
                accumulate(-in_dvalues_xi[..., i] for i in range(self.order))
            )
            dvalues_eta = tuple(
                accumulate(-in_dvalues_eta[..., i] for i in range(self.order))
            )
            (j00, j01), (j10, j11) = self.jacobian(xi, eta)
            det = j00 * j11 - j10 * j01
            for i1 in range(self.order):
                v1 = dvalues_eta[i1]
                for j1 in range(self.order):
                    u1 = dvalues_xi[j1]
                    out += c[i1 * self.order + j1] * u1 * v1

            out /= det
        else:
            raise ValueError(f"Order of the differential form {k} is not valid.")

        return np.array(out, np.float64, copy=None)

    def divide(
        self,
        order_bl: int,
        order_br: int,
        order_tl: int,
        order_tr: int,
        order_p: int | None = None,
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
        order_p : int, optional
            Order of the parent element. If given, the parent will have a fixed order.

        Returns
        -------
        ElementNode2D
            Parent element which contains the nodes.

        (2, 2) tuple of ElementLeaf2D
            Child elements of the same order as the element itself. Indexing the
            tuple will give bottom/top for the first axis and left/right for the
            second.
        """
        bottom_mid = (np.array(self._bottom_left) + np.array(self._bottom_right)) / 2
        left_mid = (np.array(self._bottom_left) + np.array(self._top_left)) / 2
        right_mid = (np.array(self._bottom_right) + np.array(self._top_right)) / 2
        top_mid = (np.array(self._top_left) + np.array(self._top_right)) / 2
        center_mid = (
            np.array(self._bottom_left)
            + np.array(self._bottom_right)
            + np.array(self._top_left)
            + np.array(self._top_right)
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

        parent = ElementNode2D(self.parent, btm_l, btm_r, top_l, top_r, order_p)

        object.__setattr__(btm_l, "parent", parent)
        object.__setattr__(btm_r, "parent", parent)
        object.__setattr__(top_l, "parent", parent)
        object.__setattr__(top_r, "parent", parent)
        return parent, ((btm_l, btm_r), (top_l, top_r))


def rhs_2d_element_projection(
    right: KElementProjection, element: ElementLeaf2D, cache: BasisCache
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element2D
        The element on which the projection is evaluated on.
    cache : BasisCache
        Cache for the correct element order.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    # TODO: don't recompute basis, just reuse the cached values.
    assert cache.basis_order == element.order
    fn = right.func
    p = element.order
    n_dof: int
    if right.weight.order == 0:
        n_dof = (p + 1) ** 2
    elif right.weight.order == 1:
        n_dof = 2 * p * (p + 1)
    elif right.weight.order == 2:
        n_dof = p**2
    else:
        assert False

    if fn is None:
        return np.zeros(n_dof)

    out_vec = np.empty(n_dof)

    basis_vals: list[npt.NDArray[np.floating]] = list()

    nodes = cache.int_nodes_1d
    weights = cache.int_weights_1d
    (j00, j01), (j10, j11) = element.jacobian(nodes[None, :], nodes[:, None])
    det = j00 * j11 - j10 * j01

    real_x = element.poly_x(nodes[None, :], nodes[:, None])
    real_y = element.poly_y(nodes[None, :], nodes[:, None])
    f_vals = fn(real_x, real_y)
    weights_2d = weights[None, :] * weights[:, None]

    # Deal with vectors first. These need special care.
    if right.weight.order == 1:
        values = lagrange1d(cache.nodes_1d, nodes)
        d_vals = dlagrange1d(cache.nodes_1d, nodes)
        d_values = tuple(accumulate(-d_vals[..., i] for i in range(p)))

        new_f0 = j00 * f_vals[..., 0] + j01 * f_vals[..., 1]
        new_f1 = j10 * f_vals[..., 0] + j11 * f_vals[..., 1]

        for i1 in range(p + 1):
            v1 = values[..., i1]
            for j1 in range(p):
                u1 = d_values[j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[i1 * p + j1] = np.sum(basis1 * weights_2d * new_f1)

        for i1 in range(p):
            v1 = d_values[i1]
            for j1 in range(p + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[p * (p + 1) + i1 * (p + 1) + j1] = np.sum(
                    basis1 * weights_2d * new_f0
                )
        return out_vec

    if right.weight.order == 2:
        d_vals = dlagrange1d(cache.nodes_1d, nodes)
        d_values = tuple(accumulate(-d_vals[..., i] for i in range(p)))
        for i1 in range(p):
            v1 = d_values[i1]
            for j1 in range(p):
                u1 = d_values[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof
        # weights_2d /= det

    elif right.weight.order == 0:
        values = lagrange1d(cache.nodes_1d, nodes)
        for i1 in range(p + 1):
            v1 = values[..., i1]
            for j1 in range(p + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        assert len(basis_vals) == n_dof
        weights_2d *= det

    else:
        assert False

    # Compute rhs integrals
    for i, bv in enumerate(basis_vals):
        out_vec[i] = np.sum(bv * f_vals * weights_2d)

    return out_vec


def rhs_2d_boundary_projection(
    right: KBoundaryProjection, element: ElementLeaf2D, cache: BasisCache, bid: int
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 2D element boundary.

    Parameters
    ----------
    right : KBoundaryProjection
        The projection onto a k-form.
    element : Element2D
        The element on which the projection is evaluated on.
    bid : int
        Id of the boundary. The following values have meaning:

        - ``0`` is bottom
        - ``1`` is right
        - ``2`` is top
        - ``3`` is left

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    if bid == 0:
        begin = element.bottom_left
        end = element.bottom_right
    elif bid == 1:
        begin = element.bottom_right
        end = element.top_right
    elif bid == 2:
        begin = element.top_right
        end = element.top_left
    elif bid == 3:
        begin = element.top_left
        end = element.bottom_left
    else:
        assert False

    nds, w = cache.int_nodes_1d, cache.int_weights_1d
    dx = (end[0] - begin[0]) / 2
    dy = (end[1] - begin[1]) / 2
    x = (end[0] + begin[0]) / 2 + dx * nds
    y = (end[1] + begin[1]) / 2 + dy * nds

    if bid == 0 or bid == 3:
        normal = np.array((-dy, dx), np.float64)
    elif bid == 1 or bid == 2:
        normal = np.array((dy, -dx), np.float64)
    else:
        assert False

    w1 = np.astype(w * normal[0], np.float64, copy=False)
    w2 = np.astype(w * normal[1], np.float64, copy=False)

    fn = right.func
    assert fn is not None
    vals = np.asarray(fn(x, y), np.float64, copy=None)

    if right.weight.order == 0:
        # the function is 1 form, weight is 0 form
        w_vals = cache.nodal_1d
        f_vals = vals[..., 0] * w1 + vals[..., 1] * w2
        p = f_vals[..., None] * w_vals
        n = p.shape[-1]
        r = np.empty(n)
        for i in range(n):
            r[i] = np.sum(p[..., i])
        return r

    # if right.weight.order == 1:
    #     w_vals = cache.edge_1d
    #     f_vals = vals
    raise NotImplementedError("Not finished yet.")


def _extract_rhs_2d(
    proj: Sequence[tuple[float, KExplicit]],
    weight: KWeight,
    element: ElementLeaf2D,
    cache: BasisCache,
) -> npt.NDArray[np.float64]:
    """Extract the rhs resulting from element projections."""
    if weight.order == 0:
        n_out = (element.order + 1) ** 2
    elif weight.order == 1:
        n_out = 2 * (element.order + 1) * element.order
    elif weight.order == 2:
        n_out = element.order**2
    else:
        assert False

    vec = np.zeros(n_out)

    for k, f in filter(lambda v: isinstance(v[1], KElementProjection), proj):
        assert isinstance(f, KElementProjection)
        rhs = rhs_2d_element_projection(f, element, cache)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def element_rhs(
    system: KFormSystem,
    element: ElementLeaf2D,
    cache: BasisCache,
) -> npt.NDArray[np.float64]:
    """Compute element matrix and vector.

    Parameters
    ----------
    system : KFormSystem
        System to discretize.
    element : Element2D
        The element on which the discretization should be performed.

    Returns
    -------
    array
        Element vector representing the right side of the system
    """
    assert element.order == cache.basis_order
    vecs: list[npt.NDArray[np.float64]] = list()

    for equation in system.equations:
        vecs.append(
            _extract_rhs_2d(
                equation.right.explicit_terms, equation.weight, element, cache
            )
        )

    return np.concatenate(vecs)


class Mesh2D:
    """Two dimensional manifold with associated geometry."""

    orders: npt.NDArray[np.uint32]
    positions: npt.NDArray[np.float64]
    primal: Manifold2D
    dual: Manifold2D
    boundary_indices: npt.NDArray[np.int32]

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

    def get_element(self, idx: int, /) -> ElementLeaf2D:
        """Obtain the 2D element corresponding to the index."""
        s = self.primal.get_surface(idx + 1)
        assert len(s) == 4, "Primal surface must be square."
        indices = np.zeros(4, dtype=int)
        for i in range(4):
            line = self.primal.get_line(s[i])
            indices[i] = line.begin.index
        return ElementLeaf2D(
            None,
            int(self.orders[idx]),
            tuple(self.positions[indices[0], :]),  # type: ignore
            tuple(self.positions[indices[1], :]),  # type: ignore
            tuple(self.positions[indices[2], :]),  # type: ignore
            tuple(self.positions[indices[3], :]),  # type: ignore
        )

    def as_polydata(self) -> pv.PolyData:
        """Convert the mesh into PyVista's polydata.

        Returns
        -------
        PolyData
            PolyData representation of the mesh.
        """
        man = self.primal
        pos = self.positions

        indices: list[list[int]] = [
            [man.get_line(i_line).begin.index for i_line in man.get_surface(i_surf + 1)]  # type: ignore
            for i_surf in range(man.n_surfaces)
        ]

        return pv.PolyData.from_irregular_faces(np.pad(pos, ((0, 0), (0, 1))), indices)
        # for i_surf in range(man.n_surfaces):
        #     s = man.get_surface(i_surf + 1)
        #     idx: list[int] = [man.get_line(i_line).begin.index for i_line in s]


def eval_expression(
    expr: Iterable[MatOp], element: ElementLeaf2D, cache: BasisCache
) -> npt.NDArray[np.float64] | np.float64:
    """Evaluate the matrix expression."""
    stack: list[tuple[float, npt.NDArray[np.floating] | None]] = []
    val: tuple[float, npt.NDArray[np.floating] | None] | None = None
    mat: npt.NDArray[np.floating]
    for op in expr:
        if type(op) is MassMat:
            if op.order == 0:
                mat = element.mass_matrix_node(cache)
            elif op.order == 1:
                mat = element.mass_matrix_edge(cache)
            elif op.order == 2:
                mat = element.mass_matrix_surface(cache)
            else:
                assert False

            if op.inv:
                mat = np.linalg.inv(mat)

            if val is not None:
                c, s = val
                if s is None:
                    val = (c, mat)
                else:
                    val = (c, mat @ s)
            else:
                val = (1.0, mat)

        elif type(op) is Incidence:
            if op.dual:
                if op.begin == 0:
                    mat = -incidence_21(element.order).T
                elif op.begin == 1:
                    mat = -incidence_10(element.order).T
                else:
                    assert False
            else:
                if op.begin == 0:
                    mat = incidence_10(element.order)
                elif op.begin == 1:
                    mat = incidence_21(element.order)
                else:
                    assert False

            if val is not None:
                c, s = val
                if s is None:
                    val = (c, mat)
                else:
                    val = (c, mat @ s)
            else:
                val = (1.0, mat)

        elif type(op) is Push:
            if val is None:
                raise ValueError("Invalid Push operation.")
            stack.append(val)
            val = None

        elif type(op) is Scale:
            if val is None:
                val = (op.k, None)
            else:
                c, s = val
                val = (c * op.k, s)

        elif type(op) is MatMul:
            k, m = stack.pop()
            if val is None:
                raise ValueError("Invalid MatMul operation.")

            c, s = val
            c *= k
            if m is None:
                val = (c, s)
            elif s is None:
                val = (c, m)
            else:
                val = (c, s @ m)

        elif type(op) is Transpose:
            if val is None:
                raise ValueError("Invalid Transpose operation.")
            c, s = val
            if s is not None:
                val = (c, s.T)

        elif type(op) is Sum:
            n = op.count
            if n <= 0:
                raise ValueError("Sum must be of a non-zero number of matrices.")
            if val is None:
                raise ValueError("Invalid Sum operation.")
            if len(stack) < n:
                raise ValueError(
                    f"Not enough matrices on the stack to Sum ({len(stack)} on stack,"
                    f" but {n} should be summed)."
                )
            c, s = val
            if s is not None:
                s *= c

            for _ in range(n):
                k, m = stack.pop()

                if s is not None and m is not None:
                    s = s + k * m
                elif s is None:
                    if m is None:
                        # Both are None
                        c *= k
                        s = None
                    else:
                        s = c * np.eye(m.shape[0]) + k * m
                else:
                    s = k * np.eye(s.shape[0]) + s

            val = (1.0, s)

        elif type(op) is Identity:
            if val is None:
                val = (1.0, None)

        else:
            raise TypeError("Unknown operation.")

    if len(stack):
        raise ValueError(f"{len(stack)} matrices still on the stack.")
    if val is None:
        return np.float64(1.0)
    c, s = val
    if s is None:
        return np.float64(c)
    return np.astype(c * s, np.float64, copy=False)
