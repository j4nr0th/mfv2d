"""Implementation of the 2D mimetic meshes and manifolds."""

from collections.abc import Sequence
from itertools import accumulate

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D, compute_gll, dlagrange1d, lagrange1d
from interplib._mimetic import Manifold2D
from interplib.interp2d import Polynomial2D
from interplib.kforms.kform import (
    KBoundaryProjection,
    KElementProjection,
    KFormDerivative,
    KFormSystem,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KProjectionCombination,
    KSum,
    KWeight,
    Term,
)
from interplib.product_basis import BasisProduct2D


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


class Element2D:
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

    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]
    top_left: tuple[float, float]

    nodes_1d: npt.NDArray[np.float64]

    poly_x: Polynomial2D
    poly_y: Polynomial2D

    _mass_node: npt.NDArray[np.float64] | None
    _mass_edge: npt.NDArray[np.float64] | None
    _mass_surf: npt.NDArray[np.float64] | None

    def __init__(
        self,
        p: int,
        bl: tuple[float, float],
        br: tuple[float, float],
        tr: tuple[float, float],
        tl: tuple[float, float],
    ) -> None:
        basis_geo = BasisProduct2D.outer_product_basis(
            Polynomial1D.lagrange_nodal_basis([-1, +1])
        )
        geo_basis = tuple(tuple(b.as_polynomial() for b in bg) for bg in basis_geo)
        self.poly_x = (
            bl[0] * geo_basis[0][0]
            + br[0] * geo_basis[1][0]
            + tl[0] * geo_basis[0][1]
            + tr[0] * geo_basis[1][1]
        )
        self.poly_y = (
            bl[1] * geo_basis[0][0]
            + br[1] * geo_basis[1][0]
            + tl[1] * geo_basis[0][1]
            + tr[1] * geo_basis[1][1]
        )

        self.order = int(p)
        self.bottom_left = bl
        self.bottom_right = br
        self.top_right = tr
        self.top_left = tl

        nodes1d, _ = compute_gll(p)
        self.nodes_1d = nodes1d

        self._mass_node = None
        self._mass_edge = None
        self._mass_surf = None

    def mass_matrix_node(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for nodal basis."""
        if self._mass_node is not None:
            return self._mass_node
        assert cache.basis_order == self.order
        precomp = cache.mass_node_precomp
        (j00, j01), (j10, j11) = self.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01

        # Does not use symmetry (yet)
        mat = np.sum(precomp * det[None, None, ...], axis=(-2, -1))

        self._mass_node = mat
        return mat

    def mass_matrix_edge(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for mixed node-edge basis."""
        if self._mass_edge is not None:
            return self._mass_edge
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
        self._mass_edge = mat
        return mat

    def mass_matrix_surface(self, cache: BasisCache) -> npt.NDArray[np.float64]:
        """Element's mass matrix for surface basis."""
        if self._mass_surf is not None:
            return self._mass_surf
        assert cache.basis_order == self.order
        precomp = cache.mass_surf_precomp
        (j00, j01), (j10, j11) = self.jacobian(
            cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None]
        )
        det = j00 * j11 - j10 * j01

        # Does not use symmetry (yet)
        mat = np.sum(precomp / det[None, None, ...], axis=(-2, -1))
        self._mass_surf = mat
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

        x0 = self.bottom_left[0]
        x1 = self.bottom_right[0]
        x2 = self.top_right[0]
        x3 = self.top_left[0]

        y0 = self.bottom_left[1]
        y1 = self.bottom_right[1]
        y2 = self.top_right[1]
        y3 = self.top_left[1]

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

    def incidence_10(self) -> npt.NDArray[np.float64]:
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
        n_nodes = self.order + 1
        n_lines = self.order
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

    def incidence_21(self) -> npt.NDArray[np.float64]:
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
        n_nodes = self.order + 1
        n_lines = self.order
        e = np.zeros(((n_lines * n_lines), (n_nodes * n_lines + n_lines * n_nodes)))

        for row in range(n_lines):
            for col in range(n_lines):
                e[row * n_lines + col, n_lines * row + col] = +1
                e[row * n_lines + col, n_lines * (row + 1) + col] = -1
                e[row * n_lines + col, n_nodes * n_lines + n_nodes * row + col] = +1
                e[row * n_lines + col, n_nodes * n_lines + n_nodes * row + col + 1] = -1

        return e

    @property
    def boundary_edge_bottom(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with edges on the bottom side."""
        return np.arange(0, self.order, dtype=np.uint32)

    @property
    def boundary_edge_left(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with edges on the left side."""
        return np.astype(
            np.flip(
                (self.order * (self.order + 1))
                + np.arange(0, self.order, dtype=np.uint32) * (self.order + 1)
            ),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_edge_top(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with edges on the top side."""
        return np.astype(
            np.flip(self.order * self.order + np.arange(0, self.order, dtype=np.uint32)),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_edge_right(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with edges on the right side."""
        return np.astype(
            (self.order * (self.order + 1))
            + self.order
            + np.arange(0, self.order, dtype=np.uint32) * (self.order + 1),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_edge_dof_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with boundary lines."""
        return np.concatenate(
            (
                self.boundary_edge_bottom,
                self.boundary_edge_right,
                self.boundary_edge_top,
                self.boundary_edge_left,
            )
        )

    @property
    def boundary_nodes_bottom(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with nodes on the bottom side."""
        return np.arange(0, self.order + 1, dtype=np.uint32)

    @property
    def boundary_nodes_left(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with nodes on the left side."""
        return np.astype(
            np.flip(np.arange(0, self.order + 1, dtype=np.uint32) * (self.order + 1)),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_nodes_top(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with nodes on the top side."""
        return np.astype(
            np.flip(
                (self.order + 1) * self.order
                + np.arange(0, self.order + 1, dtype=np.uint32)
            ),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_nodes_right(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with nodes on the right side."""
        return np.astype(
            self.order + np.arange(0, self.order + 1, dtype=np.uint32) * (self.order + 1),
            np.uint32,
            copy=False,
        )

    @property
    def boundary_node_dof_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with boundary nodes."""
        return np.concatenate(
            (
                self.boundary_nodes_bottom,
                self.boundary_nodes_right,
                self.boundary_nodes_top,
                self.boundary_nodes_left,
            )
        )

    def reconstruct(
        self,
        k: int,
        coeffs: npt.ArrayLike,
        xi: npt.ArrayLike,
        eta: npt.ArrayLike,
        /,
    ) -> npt.NDArray[np.float64]:
        """Reconstruct a k-form on the element."""
        assert k >= 0 and k < 3
        out: float | npt.NDArray[np.floating] = 0.0
        c = np.asarray(coeffs, dtype=np.float64, copy=None)
        if c.ndim != 1:
            raise ValueError("Coefficient array must be one dimensional.")

        if k == 0:
            vals_xi = lagrange1d(self.nodes_1d, xi)
            vals_eta = lagrange1d(self.nodes_1d, eta)
            for i in range(self.order + 1):
                v = vals_eta[..., i]
                for j in range(self.order + 1):
                    u = vals_xi[..., j]
                    out += c[i * (self.order + 1) + j] * (u * v)

        elif k == 1:
            values_xi = lagrange1d(self.nodes_1d, xi)
            values_eta = lagrange1d(self.nodes_1d, eta)
            in_dvalues_xi = dlagrange1d(self.nodes_1d, xi)
            in_dvalues_eta = dlagrange1d(self.nodes_1d, eta)
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
            in_dvalues_xi = dlagrange1d(self.nodes_1d, xi)
            in_dvalues_eta = dlagrange1d(self.nodes_1d, eta)
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


def rhs_2d_element_projection(
    right: KElementProjection, element: Element2D
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element2D
        The element on which the projection is evaluated on.

    Returns
    -------
    array of :class:`numpy.float64`
        weights_2d = weights[:, None] * weights[None, :]
        The resulting projection vector.
    """
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

    nodes, weights = compute_gll(3 * (p + 1))
    (j00, j01), (j10, j11) = element.jacobian(nodes[None, :], nodes[:, None])
    det = j00 * j11 - j10 * j01

    real_x = element.poly_x(nodes[None, :], nodes[:, None])
    real_y = element.poly_y(nodes[None, :], nodes[:, None])
    f_vals = fn(real_x, real_y)
    weights_2d = weights[None, :] * weights[:, None]

    # Deal with vectors first. These need special care.
    if right.weight.order == 1:
        values = lagrange1d(element.nodes_1d, nodes)
        d_vals = dlagrange1d(element.nodes_1d, nodes)
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
        d_vals = dlagrange1d(element.nodes_1d, nodes)
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
        values = lagrange1d(element.nodes_1d, nodes)
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
    right: KBoundaryProjection, element: Element2D, cache: BasisCache, bid: int
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
    proj: KProjectionCombination, element: Element2D
) -> npt.NDArray[np.float64]:
    """Extract the rhs resulting from element projections."""
    if proj.weight.order == 0:
        n_out = (element.order + 1) ** 2
    elif proj.weight.order == 1:
        n_out = 2 * (element.order + 1) * element.order
    elif proj.weight.order == 2:
        n_out = element.order**2
    else:
        assert False

    vec = np.zeros(n_out)

    for k, f in filter(lambda v: isinstance(v[1], KElementProjection), proj.pairs):
        assert isinstance(f, KElementProjection)
        rhs = rhs_2d_element_projection(f, element)
        if k != 1.0:
            rhs *= k
        vec += rhs

    return vec


def _equation_2d(
    form: Term,
    element: Element2D,
    cache: BasisCache,
) -> dict[Term, npt.NDArray[np.float64] | np.float64]:
    """Compute the matrix operations on individual forms.

    Parameter
    ---------
    form : Term
        Form to evaluate.

    Returns
    -------
    dict of KForm -> array or float
        Dictionary mapping forms to either a matrix that represents the operation to
        perform on them, or ``float``, if it should be multiplication with a constant.
    """
    if type(form) is KSum:
        left: dict[Term, npt.NDArray[np.float64] | np.float64] = {}

        for c, ip in form.pairs:
            right = _equation_2d(ip, element, cache)
            if c != 1.0:
                for f in right:
                    right[f] *= c  # type: ignore

            for k in right:
                vr = right[k]
                if k in left:
                    vl = left[k]
                    if vl.ndim == vr.ndim:
                        left[k] = np.asarray(
                            vl + vr, np.float64
                        )  # vl and vr are non-none
                    elif vl.ndim == 0:
                        assert isinstance(vr, np.ndarray)
                        mat = np.eye(vr.shape[0], vr.shape[1]) * vr
                        left[k] = np.astype(mat + vr, np.float64)
                else:
                    left[k] = right[k]  # k is not in left
        return left

    if type(form) is KInnerProduct:
        unknown: dict[Term, npt.NDArray[np.float64] | np.float64]
        if isinstance(form.function, KHodge):
            unknown = _equation_2d(form.function.base_form, element, cache)
        else:
            unknown = _equation_2d(form.function, element, cache)
        weight = _equation_2d(form.weight, element, cache)
        dv = tuple(v for v in weight.keys())[0]
        for k in unknown:
            vd = weight[dv]
            vp = unknown[k]
            order_p = form.function.primal_order
            order_d = form.weight.primal_order
            assert order_p == order_d
            mass: npt.NDArray[np.float64]
            if order_p == 0 and order_d == 0:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_node(cache)
                    else:
                        mass = np.eye((element.order + 1) ** 2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye((element.order + 1) ** 2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_node(cache))  # type: ignore
            elif order_p == 1 and order_d == 1:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_edge(cache)
                    else:
                        mass = np.eye((element.order + 1) * element.order * 2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye((element.order + 1) * element.order * 2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_edge(cache))  # type: ignore
            elif order_p == 2 and order_d == 2:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_surface(cache)
                    else:
                        mass = np.eye(element.order**2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye(element.order**2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_surface(cache))  # type: ignore
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 2D mesh."
                )
            if vd.ndim != 0:
                assert isinstance(vd, np.ndarray)
                mass = np.astype(vd.T @ mass, np.float64)
            else:
                assert isinstance(vd, np.float64)
                mass *= vd
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            unknown[k] = mass
        return unknown
    if type(form) is KFormDerivative:
        res = _equation_2d(form.form, element, cache)
        e: npt.NDArray[np.float64]
        if form.is_primal:
            if form.form.order == 0:
                e = element.incidence_10()
            elif form.form.order == 1:
                e = element.incidence_21()
            else:
                assert False
        else:
            if form.form.order == 0:
                e = -element.incidence_21().T
            elif form.form.order == 1:
                e = -element.incidence_10().T
            else:
                assert False

        for k in res:
            rk = res[k]
            if rk.ndim != 0:
                res[k] = np.astype(e @ rk, np.float64)
            else:
                assert isinstance(rk, np.float64)
                res[k] = np.astype(e * rk, np.float64)
        return res

    if type(form) is KHodge:
        unknown = _equation_2d(form.base_form, element, cache)
        prime_order = form.primal_order
        for k in unknown:
            if prime_order == 0:
                mass = element.mass_matrix_node(cache)
            elif prime_order == 1:
                mass = element.mass_matrix_edge(cache)
            elif prime_order == 2:
                mass = element.mass_matrix_surface(cache)
            else:
                assert False
            if form.is_primal:
                mass = np.linalg.inv(mass)  # type: ignore
            vp = unknown[k]
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            unknown[k] = mass
        return unknown
    if type(form) is KFormUnknown:
        return {form: np.float64(1.0)}
    if type(form) is KWeight:
        return {form: np.float64(1.0)}
    raise TypeError("Unknown type")


def element_system(
    system: KFormSystem, element: Element2D, cache: BasisCache
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        Element matrix representing the left side of the system.
    array
        Element vector representing the right side of the system
    """
    assert element.order == cache.basis_order
    system_size = system.shape_2d(element.order)
    assert system_size[0] == system_size[1], "System must be square."
    system_matrix = np.zeros(system_size, np.float64)
    system_vector = np.zeros(system_size[0], np.float64)
    offset_equations, offset_forms = system.offsets_2d(element.order)

    for ie, equation in enumerate(system.equations):
        form_matrices = _equation_2d(equation.left, element, cache)
        for form in form_matrices:
            val = form_matrices[form]
            idx = system.unknown_forms.index(form)
            assert val is not None
            system_matrix[
                offset_equations[ie] : offset_equations[ie + 1],
                offset_forms[idx] : offset_forms[idx + 1],
            ] = val
        system_vector[offset_equations[ie] : offset_equations[ie + 1]] = _extract_rhs_2d(
            equation.right, element
        )

    return system_matrix, system_vector


class Mesh2D:
    """Two dimensional manifold with associated geometry."""

    order: int
    positions: npt.NDArray[np.float64]
    primal: Manifold2D
    dual: Manifold2D

    def __init__(
        self,
        order: int,
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
        self.order = int(order)
        if order < 1:
            raise ValueError("Order can not be lower than 1.")

        pos = np.array(positions, np.float64, copy=True, ndmin=2)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("Positions must be a (N, 2) array.")
        # First try the regular surfaces
        surf = np.array(surfaces, np.int32, copy=None)
        if surf.ndim != 2 or surf.shape[1] != 4:
            raise ValueError("Surfaces should be a (M, 4) array of integers")
        lns = np.array(lines, np.int32, copy=None)
        man = Manifold2D.from_regular(pos.shape[0], lns, surf)

        self.positions = pos
        self.primal = man
        self.dual = man.compute_dual()

    @property
    def n_elements(self) -> int:
        """Number of (surface) elements in the mesh."""
        return self.primal.n_surfaces

    def get_element(self, idx: int, /) -> Element2D:
        """Obtain the 2D element corresponding to the index."""
        s = self.primal.get_surface(idx + 1)
        assert len(s) == 4, "Primal surface must be square."
        indices = np.zeros(4, dtype=int)
        for i in range(4):
            line = self.primal.get_line(s[i])
            indices[i] = line.begin.index
        return Element2D(
            self.order,
            tuple(self.positions[indices[0], :]),  # type: ignore
            tuple(self.positions[indices[1], :]),  # type: ignore
            tuple(self.positions[indices[2], :]),  # type: ignore
            tuple(self.positions[indices[3], :]),  # type: ignore
        )
