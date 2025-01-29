"""Implementation of the 2D mimetic meshes and manifolds."""

from functools import cache
from itertools import accumulate

import numpy as np
import numpy.typing as npt

from interplib._interp import Polynomial1D, compute_gll, dlagrange1d, lagrange1d
from interplib.interp2d import Polynomial2D
from interplib.kforms.kform import (
    KForm,
    KFormDerivative,
    KFormProjection,
    KFormSystem,
    KHodge,
    KInnerProduct,
    KSum,
    KWeight,
    KWeightDerivative,
    Term,
)
from interplib.product_basis import BasisProduct2D

_cached_roots_legendre = cache(compute_gll)


@cache
def _cached_roots_legendre2(
    p: int,
) -> tuple[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]:
    x, w = _cached_roots_legendre(p)
    return (np.meshgrid(x, x), np.prod(np.meshgrid(w, w), axis=-1))  # type: ignore


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
            + br[0] * self.geo_basis[0][1]
            + tl[0] * self.geo_basis[1][0]
            + tr[0] * self.geo_basis[1][1]
        )
        self.poly_y = (
            bl[1] * self.geo_basis[0][0]
            + br[1] * self.geo_basis[0][1]
            + tl[1] * self.geo_basis[1][0]
            + tr[1] * self.geo_basis[1][1]
        )

        self.order = int(p)
        self.bottom_left = bl
        self.bottom_right = br
        self.top_right = tr
        self.top_left = tl

        nodes1d = np.linspace(-1, 1, p + 1, dtype=np.float64)
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
        )
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
        )

    @property
    def mass_matrix_node(self) -> npt.NDArray[np.float64]:
        """Element's mass matrix for nodal basis."""
        n = self.order + 1
        mat = np.empty((n**2, n**2), np.float64)
        nodes, weights = compute_gll(self.order + 2)  # I think this gives exact
        values = lagrange1d(self.nodes_1d, nodes)
        weights_2d = weights[:, None] * weights[None, :]
        jacob = self.jacobian(nodes[:, None], nodes[None, :])
        weights_2d *= jacob
        basis_vals: list[npt.NDArray] = list()
        for i1 in range(n):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        for i in range(n * n):
            for j in range(i + 1):
                mat[i, j] = mat[j, i] = np.sum(basis_vals[i] * basis_vals[j] * weights_2d)
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
        weights_2d = weights[:, None] * weights[None, :]
        jacob = self.jacobian(nodes[:, None], nodes[None, :])
        weights_2d *= jacob
        basis_vals: list[npt.NDArray] = list()

        for i1 in range(n + 1):
            v1 = values[..., i1]
            for j1 in range(n):
                u1 = dvalues[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)

        for i1 in range(n):
            v1 = dvalues[i1]
            for j1 in range(n + 1):
                u1 = values[..., j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)

        for i in range(2 * n * (n + 1)):
            for j in range(i + 1):
                mat[i, j] = mat[j, i] = np.sum(weights_2d * basis_vals[i] * basis_vals[j])

        return mat

    @property
    def mass_matrix_surface(self) -> npt.NDArray[np.float64]:
        """Element's mass matrix for surface basis."""
        n = self.order
        mat = np.empty((n**2, n**2), np.float64)
        nodes, weights = compute_gll(self.order)  # I think this gives exact
        in_dvalues = dlagrange1d(self.nodes_1d, nodes)
        values = tuple(accumulate(-in_dvalues[..., i] for i in range(self.order)))
        weights_2d = weights[:, None] * weights[None, :]
        jacob = self.jacobian(nodes[:, None], nodes[None, :])
        weights_2d *= jacob
        basis_vals: list[npt.NDArray] = list()
        for i1 in range(n):
            v1 = values[i1]
            for j1 in range(n):
                u1 = values[j1]
                basis1 = v1[:, None] * u1[None, :]
                basis_vals.append(basis1)
        for i in range(n * n):
            for j in range(i + 1):
                mat[i, j] = mat[j, i] = np.sum(weights_2d * basis_vals[i] * basis_vals[j])
        return mat

    @property
    def jacobian(self) -> Polynomial2D:
        """Jacobian function."""
        px = self.poly_x
        py = self.poly_y
        dx_dxi = px.partial(0)(0, None)
        dx_deta = px.partial(1)(None, 0)
        dy_dxi = py.partial(0)(0, None)
        dy_deta = py.partial(1)(None, 0)
        return (
            BasisProduct2D(dx_dxi, dy_deta).as_polynomial()
            - BasisProduct2D(dx_deta, dy_dxi).as_polynomial()
        )

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
    def boundary_dof_indices(self) -> npt.NDArray[np.uint32]:
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
    else:
        out_vec: npt.NDArray[np.float64]
        mass: npt.NDArray[np.floating]
        comp_1d = np.linspace(-1, 1, p + 1)
        comp_xi, comp_eta = np.meshgrid(comp_1d, comp_1d)
        real_x = element.poly_x(comp_xi, comp_eta)
        real_y = element.poly_y(comp_xi, comp_eta)
        # TODO: Throw up due to how ugly this is.
        # TODO: fix this >:(
        if right.weight.order == 2:
            out_vec = np.empty(n_dof)
            jac = element.jacobian
            (xi, eta), w = _cached_roots_legendre2(2 * p)
            (b1, b2), (b3, b4) = element.geo_basis
            for i in range(p):
                for j in range(p):
                    # poly_x and poly_y map the computational nodes on element
                    # sub-division to the "physical" space
                    poly_x = (
                        real_x[i, j] * b1
                        + real_x[i, j + 1] * b2
                        + real_x[i + 1, j] * b3
                        + real_x[i + 1, j + 1] * b4
                    )
                    poly_y = (
                        real_y[i, j] * b1
                        + real_y[i, j + 1] * b2
                        + real_y[i + 1, j] * b3
                        + real_y[i + 1, j + 1] * b4
                    )

                    xcomp = poly_x(xi, eta)
                    ycomp = poly_y(xi, eta)

                    out_vec[i * p + j] = np.sum(w * fn(xcomp, ycomp) * jac(xi, eta))
            mass = element.mass_matrix_surface
        elif right.weight.order == 1:
            xi, w = _cached_roots_legendre(2 * p)
            out_vec = np.empty(n_dof)
            for i in range(p + 1):
                for j in range(p):
                    xv = real_x[i, j] * (1 - xi) + real_x[i, j + 1] * xi
                    yv = real_y[i, j] * (1 - xi) + real_y[i, j + 1] * xi
                    res = np.sum(fn(xv, yv) * w) * np.hypot(
                        real_x[i, j] - real_x[i, j + 1], real_y[i, j] - real_y[i, j + 1]
                    )
                    out_vec[i * p + j] = res

            for i in range(p):
                for j in range(p + 1):
                    xv = real_x[i, j] * (1 - xi) + real_x[i + 1, j] * xi
                    yv = real_y[i, j] * (1 - xi) + real_y[i + 1, j] * xi
                    res = np.sum(fn(xv, yv) * w) * np.hypot(
                        real_x[i, j] - real_x[i + 1, j], real_y[i + 1, j] - real_y[i, j]
                    )
                    out_vec[p * (p + 1) + i * (p + 1) + j] = res

            mass = element.mass_matrix_edge
        elif right.weight.order == 0:
            out_vec = np.empty(n_dof)
            out_vec[:] = np.asarray(fn(real_x, real_y)).flatten()
            mass = element.mass_matrix_node
        return np.astype(mass @ np.astype(out_vec, np.float64), np.float64)


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
            order_p = form.function.order
            order_d = form.weight.order
            assert order_p == order_d
            mass: npt.NDArray[np.float64]
            if order_p == 0 and order_d == 0:
                if not form.function.is_primal:
                    mass = np.eye((element.order + 1) ** 2)
                else:
                    mass = element.mass_matrix_node
            elif order_p == 1 and order_d == 1:
                if not form.function.is_primal:
                    mass = np.eye((element.order + 1) * element.order * 2)
                else:
                    mass = element.mass_matrix_edge
            elif order_p == 2 and order_d == 2:
                if not form.function.is_primal:
                    mass = np.eye((element.order) ** 2)
                else:
                    mass = element.mass_matrix_surface
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
        if form.form.order == 0:
            e = element.incidence_01()
        elif form.form.order == 1:
            e = element.incidence_12()
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

    if type(form) is KWeightDerivative:
        res = _equation_2d(form.form, element)
        if form.form.order == 0:
            e = element.incidence_01()
        elif form.form.order == 1:
            e = element.incidence_12()
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
            vp = primal[k]
            if vp.ndim != 0:
                assert isinstance(vp, np.ndarray)
                mass = np.astype(mass @ vp, np.float64)
            else:
                assert isinstance(vp, np.float64)
                mass *= vp
            primal[k] = mass
        return primal
    if type(form) is KForm:
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
            idx = system.primal_forms.index(form)
            assert val is not None
            system_matrix[
                offset_equations[ie] : offset_equations[ie + 1],
                offset_forms[idx] : offset_forms[idx + 1],
            ] = val
        system_vector[offset_equations[ie] : offset_equations[ie + 1]] = _extract_rhs_2d(
            equation.right, element
        )

    return system_matrix, system_vector
