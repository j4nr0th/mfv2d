"""Tests concerning explicit evaluation of terms versus using a matrix."""

import numpy as np
import numpy.typing as npt
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    IntegrationRule1D,
    compute_element_matrix,
    compute_element_vector,
)
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import element_primal_dofs
from mfv2d.solve_system import CompiledSystem
from mfv2d.system import KFormSystem


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

    def compute_errors(n1: int, n2: int) -> tuple[float, float]:
        """Compute projection and momentum errors."""
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
        proj_err_linf = np.abs(rhs - explicit_rhs).max()
        # assert np.abs(rhs - explicit_rhs).max() < 1e-7

        # momentum_rhs = rhs[(N + 1) ** 2 : (N + 1) ** 2 + 2 * N * (N + 1)]
        mass_edge = elem_fem_space.mass_from_order(UnknownFormOrder.FORM_ORDER_1)
        momentum_rhs = np.linalg.solve(
            mass_edge, rhs[(n1 + 1) ** 2 : (n1 + 1) ** 2 + 2 * n1 * (n1 + 1)]
        )
        proj_momentum_rhs = element_primal_dofs(
            UnknownFormOrder.FORM_ORDER_1, elem_fem_space, exact_momentum
        )
        momentum_error_linf = np.abs(momentum_rhs - proj_momentum_rhs).max()
        # assert np.abs(momentum_rhs - proj_momentum_rhs).max() < 1e-6
        return momentum_error_linf, proj_err_linf

    n_vals = np.arange(3, 12)
    err_proj = np.zeros(n_vals.size, np.float64)
    err_mome = np.zeros(n_vals.size, np.float64)
    for i, n in enumerate(n_vals):
        ep, em = compute_errors(n, n + 2)
        err_proj[i] = ep
        err_mome[i] = em

    # Projection error must be decreasing
    assert np.all(err_proj[1:] < err_proj[:-1])

    # Momentum error must be decreasing
    assert np.all(err_mome[1:] < err_mome[:-1])

    kp1, kp0 = np.polyfit(n_vals, np.log(err_proj), 1)
    kp1, kp0 = np.exp(kp1), np.exp(kp0)
    # Decreasing error
    assert kp1 < 1
    km1, km0 = np.polyfit(n_vals, np.log(err_mome), 1)
    km1, km0 = np.exp(km1), np.exp(km0)
    # Decreasing error
    assert km1 < 1

    # from matplotlib import pyplot as plt

    # fig, ax = plt.subplots()

    # ax.scatter(n_vals, err_proj, label="proj")
    # ax.plot(
    #     n_vals, kp0 * kp1**n_vals, label=f"${kp0:g} \\cdot \\left( {kp1:+g} \\right)^n$"
    # )

    # ax.scatter(n_vals, err_mome, label="mome")
    # ax.plot(
    #     n_vals, km0 * km1**n_vals, label=f"${km0:g} \\cdot \\left( {km1:+g} \\right)^n$"
    # )

    # ax.set(xlabel="$n$", ylabel="$\\varepsilon$", yscale="log")
    # ax.grid()
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
