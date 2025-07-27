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
from mfv2d.kform import KFormSystem, KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import element_primal_dofs
from mfv2d.solve_system import CompiledSystem

_ORDERS = ((2, 3), (5, 7), (11, 12))


@pytest.mark.parametrize(("n1", "n2"), _ORDERS)
def test_explicit_evaluation(n1: int, n2: int):
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
    compiled = CompiledSystem(system)

    linearized_system = KFormSystem(
        w_vor * vor + w_vor.derivative * vel == 0,  # Vorticity
        w_vel.derivative * pre
        + ((1 / RE) * (w_vel * vor.derivative))
        + w_vel * (vel_exact * (~vor))
        == 0,  # Momentum
        w_pre * vel.derivative == w_pre @ 0,  # Continuity
        sorting=lambda f: f.order,
    )
    linearized_compiled = CompiledSystem(linearized_system)

    rule = IntegrationRule1D(n2)
    basis_1d = Basis1D(n1, rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array(((-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0)), np.float64)

    elem_fem_space = ElementFemSpace2D(basis_2d, corners)
    sys_mat = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        linearized_compiled.lhs_full,
        linearized_compiled.vector_field_specs,
        elem_fem_space,
    )

    proj_vor = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, elem_fem_space, vor_exact
    )
    proj_vel = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, elem_fem_space, vel_exact
    )
    proj_pre = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, elem_fem_space, pre_exact
    )

    exact_lhs = np.concatenate((proj_vor, proj_vel, proj_pre), dtype=np.float64)

    explicit_rhs = compute_element_vector(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        compiled.vector_field_specs,
        elem_fem_space,
        exact_lhs,
    )

    rhs = sys_mat @ exact_lhs
    print(f"RHS error: {np.abs(rhs - explicit_rhs).max():e}")
    # assert np.abs(rhs - explicit_rhs).max() < 1e-7

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
    mass_edge = elem_fem_space.mass_from_order(UnknownFormOrder.FORM_ORDER_1)
    momentum_rhs = np.linalg.solve(
        mass_edge, rhs[(n1 + 1) ** 2 : (n1 + 1) ** 2 + 2 * n1 * (n1 + 1)]
    )
    proj_momentum_rhs = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, elem_fem_space, exact_momentum
    )
    print(f"Momentum error: {np.abs(momentum_rhs - proj_momentum_rhs).max():e}")
    # assert np.abs(momentum_rhs - proj_momentum_rhs).max() < 1e-6


if __name__ == "__main__":
    for n1, n2 in _ORDERS:
        #
        test_explicit_evaluation(n1, n2)
