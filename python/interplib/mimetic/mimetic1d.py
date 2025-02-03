"""Prototypes of Mimetic operations."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from itertools import accumulate

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre

from interplib._interp import Polynomial1D
from interplib._mimetic import Line, Manifold, Manifold1D
from interplib.kforms.kform import (
    KForm,
    KFormDerivative,
    KFormProjection,
    KFormSystem,
    KHodge,
    KInnerProduct,
    KSum,
    KWeight,
    Term,
)

_cached_roots_legendre = cache(roots_legendre)


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

    @property
    def manifold(self) -> Manifold:
        """Return the manifold of the mesh."""
        return self.primal


@cache
def _reference_matrices_1d(
    p: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    mass_ne = np.empty((p + 1, p), np.float64)
    for i in range(p + 1):
        for j in range(p):
            inner = node_basis[i] * edge_basis[j]
            anti = inner.antiderivative
            mass_ne[i, j] = anti(+1) - anti(-1)

    return mass_node, mass_edge, mass_ne


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
    mass_node_edge: npt.NDArray[np.floating]

    def __init__(self, p: int, x0: float, x1: float) -> None:
        self.order = p
        self.jacobian = (x1 - x0) / 2.0
        self.nodes = _cached_element_nodes(p)
        self._node_basis = None
        self._edge_basis = None
        self.xleft = x0
        self.xright = x1
        mat_n, mat_e, mat_ne = _reference_matrices_1d(p)
        self.mass_node = mat_n * self.jacobian
        self.mass_edge = mat_e / self.jacobian
        self.mass_node_edge = mat_ne

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


def _extract_rhs_1d(
    right: KFormProjection, element: Element1D
) -> npt.NDArray[np.float64]:
    """Evaluate the differential form projections on the 1D element.

    Parameters
    ----------
    right : KFormProjection
        The projection onto a k-form.
    element : Element1D
        The element on which the projection is evaluated on.

    Returns
    -------
    array of :class:`numpy.float64`
        The resulting projection vector.
    """
    fn = right.func
    p = element.order
    if fn is None:
        if right.weight.order == 1:
            return np.zeros(p)
        elif right.weight.order == 0:
            return np.zeros(p + 1)
        else:
            assert False
    else:
        out_vec: npt.NDArray[np.float64]
        mass: npt.NDArray[np.floating]
        real_nodes = np.linspace(element.xleft, element.xright, p + 1)
        # TODO: find a nice way to do this
        if right.weight.order == 1:
            xi, w = _cached_roots_legendre(2 * p)
            out_vec = np.empty(p)
            func = fn
            for i in range(p):
                # out_vec[i] =  quad(fn[1], real_nodes[i], real_nodes[i + 1])[0]
                dx = real_nodes[i + 1] - real_nodes[i]
                out_vec[i] = np.dot(w, func(dx * (xi + 1) / 2 + real_nodes[i])) * (dx) / 2
            mass = element.mass_edge
        elif right.weight.order == 0:
            out_vec = np.empty(p + 1)
            out_vec[:] = fn(real_nodes)
            mass = element.mass_node
        return np.astype(mass @ np.astype(out_vec, np.float64), np.float64)


def _equation_1d(
    form: Term, element: Element1D
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
            right = _equation_1d(ip, element)
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
            primal = _equation_1d(form.function.base_form, element)
        else:
            primal = _equation_1d(form.function, element)
        dual = _equation_1d(form.weight, element)
        dv = tuple(v for v in dual.keys())[0]
        for k in primal:
            vd = dual[dv]
            vp = primal[k]
            order_p = form.function.order
            order_d = form.weight.order
            assert order_p == order_d
            mass: npt.NDArray[np.float64]
            if not form.function.is_primal:
                mass = np.eye(element.order + 1 - order_p)
            elif order_p == 0 and order_d == 0:
                mass = element.mass_node  # type: ignore
            elif order_p == 1 and order_d == 1:
                mass = element.mass_edge  # type: ignore
            else:
                raise ValueError(
                    f"Order {form.function.order} can't be used on a 1D mesh."
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
        res = _equation_1d(form.form, element)
        if form.is_primal:
            e = element.incidence_primal_0()
        else:
            e = -element.incidence_primal_0().T
        for k in res:
            rk = res[k]
            if rk.ndim != 0:
                res[k] = np.astype(e @ rk, np.float64)
            else:
                assert isinstance(rk, np.float64)
                res[k] = np.astype(e * rk, np.float64)
        return res

    if type(form) is KHodge:
        primal = _equation_1d(form.base_form, element)
        prime_order = form.primal_order
        for k in primal:
            if prime_order == 0:
                mass = element.mass_node  # type: ignore
            elif prime_order == 1:
                mass = element.mass_edge  # type: ignore
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
    system: KFormSystem, element: Element1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute element matrix and vector.

    Parameters
    ----------
    system : KFormSystem
        System to discretize.
    element : Element1D
        The element on which the discretization should be performed.

    Returns
    -------
    array
        Element matrix representing the left side of the system.
    array
        Element vector representing the right side of the system
    """
    system_size = system.shape_1d(element.order)
    assert system_size[0] == system_size[1], "System must be square."
    system_matrix = np.zeros(system_size, np.float64)
    system_vector = np.zeros(system_size[0], np.float64)
    offset_equations, offset_forms = system.offsets_1d(element.order)

    for ie, equation in enumerate(system.equations):
        form_matrices = _equation_1d(equation.left, element)
        for form in form_matrices:
            val = form_matrices[form]
            idx = system.unknown_forms.index(form)
            assert val is not None
            system_matrix[
                offset_equations[ie] : offset_equations[ie + 1],
                offset_forms[idx] : offset_forms[idx + 1],
            ] = val
        system_vector[offset_equations[ie] : offset_equations[ie + 1]] = _extract_rhs_1d(
            equation.right, element
        )

    return system_matrix, system_vector
