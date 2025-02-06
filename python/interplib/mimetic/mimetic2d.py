"""Implementation of the 2D mimetic meshes and manifolds."""

from itertools import accumulate

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D, compute_gll, dlagrange1d, lagrange1d
from interplib.interp2d import Polynomial2D
from interplib.kforms.kform import (
    KFormDerivative,
    KFormProjection,
    KFormSystem,
    KFormUnknown,
    KHodge,
    KInnerProduct,
    KSum,
    KWeight,
    Term,
)
from interplib.product_basis import BasisProduct2D


class Element2D:
    """Class which represents a 2D square element."""

    order: int

    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    top_right: tuple[float, float]
    top_left: tuple[float, float]

    nodes_1d: npt.NDArray[np.float64]
    nodes_2d: npt.NDArray[np.float64]
    basis_node: npt.NDArray
    basis_edge_h: npt.NDArray
    basis_edge_v: npt.NDArray

    poly_x: Polynomial2D
    poly_y: Polynomial2D

    geo_basis: tuple[tuple[Polynomial2D, ...], ...]

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
        self.geo_basis = tuple(tuple(b.as_polynomial() for b in bg) for bg in basis_geo)
        self.poly_x = (
            bl[0] * self.geo_basis[0][0]
            + br[0] * self.geo_basis[1][0]
            + tl[0] * self.geo_basis[0][1]
            + tr[0] * self.geo_basis[1][1]
        )
        self.poly_y = (
            bl[1] * self.geo_basis[0][0]
            + br[1] * self.geo_basis[1][0]
            + tl[1] * self.geo_basis[0][1]
            + tr[1] * self.geo_basis[1][1]
        )

        self.order = int(p)
        self.bottom_left = bl
        self.bottom_right = br
        self.top_right = tr
        self.top_left = tl

        nodes1d, _ = compute_gll(p)
        self.nodes_1d = nodes1d
        self.nodes_2d = np.empty((p + 1, p + 1, 2))
        for i in range(p + 1):
            self.nodes_2d[i, :, 0] = nodes1d
            self.nodes_2d[:, i, 1] = nodes1d
        node1d = Polynomial1D.lagrange_nodal_basis(nodes1d)
        edge1d = tuple(accumulate(-basis.derivative for basis in node1d[:-1]))

        self.basis_node = np.array(
            [
                [b.as_polynomial() for b in ba]
                for ba in BasisProduct2D.outer_product_basis(node1d, node1d)
            ]
        ).T
        self.basis_edge_h = np.array(
            [
                [b.as_polynomial() for b in ba]
                for ba in BasisProduct2D.outer_product_basis(edge1d, node1d)
            ]
        ).T
        self.basis_edge_v = np.array(
            [
                [b.as_polynomial() for b in ba]
                for ba in BasisProduct2D.outer_product_basis(node1d, edge1d)
            ]
        ).T
        self.basis_surf = np.array(
            [
                [b.as_polynomial() for b in ba]
                for ba in BasisProduct2D.outer_product_basis(edge1d, edge1d)
            ]
        ).T

    @property
    def mass_matrix_node(self) -> npt.NDArray[np.float64]:
        """Element's mass matrix for nodal basis."""
        n = self.order + 1
        mat = np.empty((n**2, n**2), np.float64)
        nodes, weights = compute_gll(5 * self.order + 2)  # `self.order + 2` is exact
        values = lagrange1d(self.nodes_1d, nodes)
        weights_2d = weights[:, None] * weights[None, :]
        jacob = self.jacobian  # (nodes[None, :], nodes[:, None])
        j00 = jacob[0][0](nodes[None, :], nodes[:, None])
        j01 = jacob[0][1](nodes[None, :], nodes[:, None])
        j10 = jacob[1][0](nodes[None, :], nodes[:, None])
        j11 = jacob[1][1](nodes[None, :], nodes[:, None])
        det = j00 * j11 - j10 * j01

        weights_2d *= det
        basis_vals: list[npt.NDArray] = list()
        # analytical_basis: list[Polynomial2D] = tuple(self.basis_node.flat)
        for i1 in range(n):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        for i in range(n * n):
            for j in range(i + 1):
                # prod = analytical_basis[j] * analytical_basis[i]
                # ad0 = prod.antiderivative(0)
                # ad1 = (ad0(+1, None) + (-1) * ad0(-1, None)).antiderivative
                # res_anl = ad1(+1) - ad1(-1)
                res = np.sum(basis_vals[i] * basis_vals[j] * weights_2d)
                # assert np.isclose(res, res_anl)
                mat[i, j] = mat[j, i] = res
        assert np.allclose(mat, mat.T)
        return mat

    @property
    def mass_matrix_edge(self) -> npt.NDArray[np.float64]:
        """Element's mass matrix for mixed node-edge basis."""
        n = self.order
        mat = np.empty((2 * n * (n + 1), 2 * n * (n + 1)), np.float64)
        nodes, weights = compute_gll(5 * self.order)  # I think this gives exact
        values = lagrange1d(self.nodes_1d, nodes)
        in_dvalues = dlagrange1d(self.nodes_1d, nodes)
        dvalues = tuple(accumulate(-in_dvalues[..., i] for i in range(self.order)))
        weights_2d = weights[None, :] * weights[:, None]
        jacob = self.jacobian  # (nodes[None, :], nodes[:, None])
        j00 = jacob[0][0](nodes[None, :], nodes[:, None])
        j01 = jacob[0][1](nodes[None, :], nodes[:, None])
        j10 = jacob[1][0](nodes[None, :], nodes[:, None])
        j11 = jacob[1][1](nodes[None, :], nodes[:, None])
        det = j00 * j11 - j10 * j01

        khh = j00**2 + j10**2
        kvv = j01**2 + j11**2
        kvh = j01 * j00 + j11 * j10

        weights_2d /= det
        basis_h: list[npt.NDArray] = list()
        basis_v: list[npt.NDArray] = list()

        for i1 in range(n + 1):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = dvalues[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_h.append(basis1)

        for i1 in range(n):
            v1 = dvalues[i1]
            for j1 in range(n + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_v.append(basis1)
        nh = len(basis_h)
        nv = len(basis_v)
        for i in range(nh):
            for j in range(i + 1):
                res = np.sum(weights_2d * basis_h[i] * basis_h[j] * khh)
                mat[i, j] = mat[j, i] = res

        for i in range(nv):
            for j in range(i + 1):
                res = np.sum(weights_2d * basis_v[i] * basis_v[j] * kvv)
                mat[nh + i, nh + j] = mat[nh + j, nh + i] = res

        for i in range(nv):
            for j in range(nh):
                res = np.sum(weights_2d * basis_h[j] * basis_v[i] * kvh)
                mat[nh + i, j] = mat[j, nh + i] = res

        assert np.allclose(mat, mat.T)
        return mat

    @property
    def mass_matrix_surface(self) -> npt.NDArray[np.float64]:
        """Element's mass matrix for surface basis."""
        n = self.order
        mat = np.empty((n**2, n**2), np.float64)
        nodes, weights = compute_gll(2 * self.order)  # I think `self.order` gives exact
        in_dvalues = dlagrange1d(self.nodes_1d, nodes)
        values = tuple(accumulate(-in_dvalues[..., i] for i in range(self.order)))
        weights_2d = weights[:, None] * weights[None, :]
        jacob = self.jacobian  # (nodes[None, :], nodes[:, None])
        j00 = jacob[0][0](nodes[None, :], nodes[:, None])
        j01 = jacob[0][1](nodes[None, :], nodes[:, None])
        j10 = jacob[1][0](nodes[None, :], nodes[:, None])
        j11 = jacob[1][1](nodes[None, :], nodes[:, None])
        det = j00 * j11 - j10 * j01

        weights_2d /= det
        basis_vals: list[npt.NDArray] = list()
        # analitical_basis: tuple[Polynomial2D] = tuple(self.basis_surf.flat)
        for i1 in range(n):
            v1 = values[i1]
            for j1 in range(n):
                u1 = values[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        for i in range(n * n):
            for j in range(i + 1):
                # prod = analitical_basis[j] * analitical_basis[i]
                # ad0 = prod.antiderivative(0)
                # ad1 = (ad0(+1, None) + (-1) * ad0(-1, None)).antiderivative
                # res_anl = ad1(+1) - ad1(-1)
                res = np.sum(weights_2d * basis_vals[i] * basis_vals[j])
                # assert np.isclose(res, res_anl)
                mat[i, j] = mat[j, i] = res
        return mat

    @property
    def jacobian(
        self,
    ) -> tuple[tuple[Polynomial2D, Polynomial2D], tuple[Polynomial2D, Polynomial2D]]:
        """Jacobian functions."""
        px = self.poly_x
        py = self.poly_y
        dx_dxi = px.partial(0)
        dx_deta = px.partial(1)
        dy_dxi = py.partial(0)
        dy_deta = py.partial(1)
        return ((dx_dxi, dx_deta), (dy_dxi, dy_deta))

    def incidence_01(self) -> npt.NDArray[np.float64]:
        """Incidence matrix from points to lines."""
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

    def incidence_12(self) -> npt.NDArray[np.float64]:
        """Incidence matrix from lines to surfaces."""
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
    def boundary_edge_dof_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with boundary lines."""
        bottom = np.arange(0, self.order, dtype=np.uint32)
        left = np.flip(
            (self.order * (self.order + 1))
            + np.arange(0, self.order, dtype=np.uint32) * (self.order + 1)
        )
        top = np.flip(self.order * self.order + np.arange(0, self.order, dtype=np.uint32))
        right = (
            (self.order * (self.order + 1))
            + self.order
            + np.arange(0, self.order, dtype=np.uint32) * (self.order + 1)
        )
        return np.concatenate((bottom, right, top, left))

    @property
    def boundary_node_dof_indices(self) -> npt.NDArray[np.uint32]:
        """Indices of degrees of freedom associated with boundary nodes."""
        bottom = np.arange(0, self.order + 1, dtype=np.uint32)
        left = np.flip(np.arange(0, self.order + 1, dtype=np.uint32) * (self.order + 1))
        top = np.flip(
            (self.order + 1) * self.order + np.arange(0, self.order + 1, dtype=np.uint32)
        )
        right = self.order + np.arange(0, self.order + 1, dtype=np.uint32) * (
            self.order + 1
        )
        return np.concatenate((bottom, right, top, left))


def _extract_rhs_2d(
    right: KFormProjection, element: Element2D
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
    jacob = element.jacobian  # (nodes[None, :], nodes[:, None])
    j00 = jacob[0][0](nodes[None, :], nodes[:, None])
    j01 = jacob[0][1](nodes[None, :], nodes[:, None])
    j10 = jacob[1][0](nodes[None, :], nodes[:, None])
    j11 = jacob[1][1](nodes[None, :], nodes[:, None])
    real_x = element.poly_x(nodes[None, :], nodes[:, None])
    real_y = element.poly_y(nodes[None, :], nodes[:, None])
    det = j00 * j11 - j10 * j01
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

                out_vec[i1 * p + j1] = np.sum(basis1 * new_f1 * weights_2d)

        for i1 in range(p):
            v1 = d_values[i1]
            for j1 in range(p + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]

                out_vec[p * (p + 1) + i1 * (p + 1) + j1] = np.sum(
                    basis1 * new_f0 * weights_2d
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
        # from matplotlib import pyplot as plt

        # plt.figure()
        # plt.title(f"Basis {i + 1:d} out of {len(basis_vals):d}")
        # plt.imshow(bv)
        # plt.colorbar()
        # plt.show()
        out_vec[i] = np.sum(bv * f_vals * weights_2d)

    return out_vec


#
def _equation_2d(
    form: Term, element: Element2D
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
            right = _equation_2d(ip, element)
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
        primal: dict[Term, npt.NDArray[np.float64] | np.float64]
        if isinstance(form.function, KHodge):
            primal = _equation_2d(form.function.base_form, element)
        else:
            primal = _equation_2d(form.function, element)
        dual = _equation_2d(form.weight, element)
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            order_p = form.function.primal_order
            order_d = form.weight.primal_order
            assert order_p == order_d
            mass: npt.NDArray[np.float64]
            if order_p == 0 and order_d == 0:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_node
                    else:
                        mass = np.eye((element.order + 1) ** 2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye((element.order + 1) ** 2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_node)  # type: ignore
            elif order_p == 1 and order_d == 1:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_edge
                    else:
                        mass = np.eye((element.order + 1) * element.order * 2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye((element.order + 1) * element.order * 2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_edge)  # type: ignore
            elif order_p == 2 and order_d == 2:
                if form.function.is_primal:
                    if form.weight.is_primal:
                        mass = element.mass_matrix_surface
                    else:
                        mass = np.eye(element.order**2)
                else:
                    if form.weight.is_primal:
                        mass = np.eye(element.order**2)
                    else:
                        mass = np.linalg.inv(element.mass_matrix_surface)  # type: ignore
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
            primal[k] = mass
        return primal
    if type(form) is KFormDerivative:
        res = _equation_2d(form.form, element)
        e: npt.NDArray[np.float64]
        if form.is_primal:
            if form.form.order == 0:
                e = element.incidence_01()
            elif form.form.order == 1:
                e = element.incidence_12()
            else:
                assert False
        else:
            if form.form.order == 0:
                e = -element.incidence_12().T
            elif form.form.order == 1:
                e = -element.incidence_01().T
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
        primal = _equation_2d(form.base_form, element)
        prime_order = form.primal_order
        for k in primal:
            if prime_order == 0:
                mass = element.mass_matrix_node
            elif prime_order == 1:
                mass = element.mass_matrix_edge
            elif prime_order == 2:
                mass = element.mass_matrix_surface
            else:
                assert False
            if form.is_primal:
                mass = np.linalg.inv(mass)  # type: ignore
            vp = primal[k]
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            primal[k] = mass
        return primal
    if type(form) is KFormUnknown:
        return {form: np.float64(1.0)}
    if type(form) is KWeight:
        return {form: np.float64(1.0)}
    raise TypeError("Unknown type")


def element_system(
    system: KFormSystem, element: Element2D
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
    system_size = system.shape_2d(element.order)
    assert system_size[0] == system_size[1], "System must be square."
    system_matrix = np.zeros(system_size, np.float64)
    system_vector = np.zeros(system_size[0], np.float64)
    offset_equations, offset_forms = system.offsets_2d(element.order)

    for ie, equation in enumerate(system.equations):
        form_matrices = _equation_2d(equation.left, element)
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
