"""Tests concerning explicit evaluation of terms versus using a matrix."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    IntegrationRule1D,
    compute_element_matrix,
    compute_element_vector,
)
from mfv2d.eval import translate_system
from mfv2d.kform import KFormSystem, KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import bilinear_interpolate, element_primal_dofs


def test_explicit_evaluation():
    """Check that C function for explicit evaluation works fine."""
    RE = 1.5

    def vel_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Exact velocity field."""
        u = np.asarray(x)
        v = np.asarray(y)
        return np.stack(
            (np.cos(u) * np.sin(v), -np.sin(u) * np.cos(v)), dtype=np.float64, axis=-1
        )

    def vor_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Exact vorticity field."""
        u = np.asarray(x)
        v = np.asarray(y)
        return np.astype((-2 * np.cos(u) * np.cos(v)), np.float64, copy=False)

    def pre_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Exact pressure field."""
        u = np.asarray(x)
        v = np.asarray(y)
        return np.astype((0 * (u + v)), np.float64, copy=False)

    vor = KFormUnknown("vor", UnknownFormOrder.FORM_ORDER_0)
    vel = KFormUnknown("vel", UnknownFormOrder.FORM_ORDER_1)
    pre = KFormUnknown("pre", UnknownFormOrder.FORM_ORDER_2)

    w_vor = vor.weight
    w_vel = vel.weight
    w_pre = pre.weight

    system = KFormSystem(
        w_vor * vor + w_vor.derivative * vel == 0,  # Vorticity
        w_vel.derivative * pre
        + ((1 / RE) * (w_vel * vor.derivative))
        + w_vel * (vel ^ (~vor))
        == 0,  # Momentum
        w_pre * vel.derivative == w_pre @ 0,  # Continuity
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    codes = translate_system(system, vector_fields, newton=False)

    N = 11
    N2 = 12

    rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array(((-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0)), np.float64)

    vector_fields = system.vector_fields

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )

    nodes_xi = basis_2d.basis_xi.rule.nodes[None, :]
    nodes_eta = basis_2d.basis_eta.rule.nodes[:, None]
    x = bilinear_interpolate(corners[:, 0], nodes_xi, nodes_eta)
    y = bilinear_interpolate(corners[:, 1], nodes_xi, nodes_eta)
    func_dict = {vor: vor_exact, vel: vel_exact}
    for i, vec_fld in enumerate(vector_fields):
        assert type(vec_fld) is KFormUnknown
        assert not callable(vec_fld)
        fn = func_dict[vec_fld]
        vf = fn(x, y)
        vec_field_lists[i].append(vf)

    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    elem_cache = ElementFemSpace2D(basis_2d, corners)
    sys_mat = compute_element_matrix(
        [UnknownFormOrder(form.order) for form in system.unknown_forms],
        codes,
        vec_fields,
        elem_cache,
    )

    proj_vor = element_primal_dofs(UnknownFormOrder.FORM_ORDER_0, elem_cache, vor_exact)
    proj_vel = element_primal_dofs(UnknownFormOrder.FORM_ORDER_1, elem_cache, vel_exact)
    proj_pre = element_primal_dofs(UnknownFormOrder.FORM_ORDER_2, elem_cache, pre_exact)

    exact_lhs = np.concatenate((proj_vor, proj_vel, proj_pre), dtype=np.float64)

    explicit_rhs = compute_element_vector(
        [
            UnknownFormOrder.FORM_ORDER_0,
            UnknownFormOrder.FORM_ORDER_1,
            UnknownFormOrder.FORM_ORDER_2,
        ],
        codes,
        vec_fields,
        elem_cache,
        exact_lhs,
    )

    rhs = sys_mat @ exact_lhs
    assert pytest.approx(rhs) == explicit_rhs

    def exact_momentum(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute the exact momentum forcing term."""
        u = np.asarray(x)
        v = np.asarray(y)
        return np.stack(
            (
                2 * np.cos(u) * (-(np.cos(v) ** 2) * np.sin(u) + np.sin(v) / RE),
                -2 * np.cos(v) * (np.cos(u) ** 2 * np.sin(v) + np.sin(u) / RE),
            ),
            dtype=np.float64,
            axis=-1,
        )

    # momentum_rhs = rhs[(N + 1) ** 2 : (N + 1) ** 2 + 2 * N * (N + 1)]
    mass_edge = elem_cache.mass_from_order(UnknownFormOrder.FORM_ORDER_1)
    momentum_rhs = np.linalg.solve(
        mass_edge, rhs[(N + 1) ** 2 : (N + 1) ** 2 + 2 * N * (N + 1)]
    )
    proj_momentum_rhs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, elem_cache, exact_momentum
    )
    assert np.abs(momentum_rhs - proj_momentum_rhs).max() < 1e-6


if __name__ == "__main__":
    test_explicit_evaluation()
