"""Check that element system is correctly computed."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    IntegrationRule1D,
    compute_element_matrix,
)
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import element_dual_dofs, element_primal_dofs
from mfv2d.system import KFormSystem

Function2D = Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]


def exact_interior_prod_2_dual(vec: Function2D, form0: Function2D) -> Function2D:
    """Create an interior product dual function."""

    def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute interior product."""
        vec_field = vec(x, y)
        scl_field = form0(x, y)
        out = np.empty_like(vec_field)
        out[..., 0] = -vec_field[..., 1] * scl_field
        out[..., 1] = vec_field[..., 0] * scl_field
        return out

    return wrapped


def exact_interior_prod_2(vec: Function2D, form2: Function2D) -> Function2D:
    """Create an interior product function."""

    def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute interior product."""
        vec_field = vec(x, y)
        scl_field = form2(x, y)
        out = np.empty_like(vec_field)
        out[..., 0] = -vec_field[..., 0] * scl_field
        out[..., 1] = -vec_field[..., 1] * scl_field
        return out

    return wrapped


def exact_interior_prod_1_dual(vec: Function2D, form2: Function2D) -> Function2D:
    """Create a dual interior product function."""

    def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute interior product."""
        vec_field = vec(x, y)
        form_field = form2(x, y)
        return (
            form_field[..., 0] * vec_field[..., 0]
            + form_field[..., 1] * vec_field[..., 1]
        )

    return wrapped


def exact_interior_prod_1(vec: Function2D, form2: Function2D) -> Function2D:
    """Create an interior product function."""

    def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute interior product."""
        vec_field = vec(x, y)
        form_field = form2(x, y)
        return (
            form_field[..., 1] * vec_field[..., 0]
            - form_field[..., 0] * vec_field[..., 1]
        )

    return wrapped


def compute_system_matrix_nonlin(
    system: KFormSystem, fem_space: ElementFemSpace2D, dofs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute system matrix."""
    compiled = CompiledSystem(system)
    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        system.unknown_forms, compiled.nonlin_codes, fem_space, dofs
    )

    return emat


def compute_system_matrix_lin(
    system: KFormSystem, fem_space: ElementFemSpace2D
) -> npt.NDArray[np.float64]:
    """Compute system matrix."""
    compiled = CompiledSystem(system)
    emat = compute_element_matrix(system.unknown_forms, compiled.lhs_full, fem_space)

    return emat


def compute_system_matrix_adj(
    system: KFormSystem, fem_space: ElementFemSpace2D, dofs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute system matrix."""
    compiled = CompiledSystem(system)

    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        system.unknown_forms, compiled.nonlin_codes, fem_space, dofs
    )

    return emat


_CORNER_TEST_VALUES = (
    ((-1, -1), (+1, -1), (+1, +1), (-1, +1)),
    ((-2, -2), (+2, -2), (+2, +2), (-2, +2)),
    ((-0.1, -2), (+0.1, -2), (+0.1, +2), (-0.1, +2)),
    ((-2, -0.1), (+2, -0.1), (+2, +0.1), (-2, +0.1)),
    ((-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0)),
)


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_10(corner_vals: npt.ArrayLike) -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0 * v1**3, -(v0**2) * v1),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_1)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_0)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )

    N = 6
    N2 = 10

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    corners = np.array(corner_vals, np.float64)
    fem_space = ElementFemSpace2D(basis_2d, corners)

    emat = compute_system_matrix_lin(system, fem_space)
    emat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]

    exact_eprod = exact_interior_prod_1(u_exact, omega_exact)

    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, fem_space, omega_exact
    )
    lhs = emat @ omega_proj
    rhs = element_dual_dofs(UnknownFormOrder.FORM_ORDER_0, fem_space, exact_eprod)
    assert pytest.approx(lhs) == rhs


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_dual_advect_10(corner_vals: npt.ArrayLike) -> None:
    """Check dual interior product of a 1-form with a 0-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0**2 * v1, -v0 * v1**3),
            # (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0 * v1**3, -(v0**2) * v1),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_1)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_2)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * ~omega)) == w @ 0,
        (v.derivative * g) == v @ 0,
        sorting=lambda f: f.order,
    )
    # print(system)

    N = 6
    N2 = 10

    N = 5
    N2 = 10
    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    corners = np.array(corner_vals, np.float64)
    fem_space = ElementFemSpace2D(basis_2d, corners)
    emat = compute_system_matrix_lin(system, fem_space)

    emat = emat[2 * (N + 1) * N :, : 2 * (N + 1) * N]

    exact_eprod = exact_interior_prod_1_dual(u_exact, omega_exact)

    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, fem_space, omega_exact
    )
    lhs = emat @ omega_proj
    rhs = element_dual_dofs(UnknownFormOrder.FORM_ORDER_2, fem_space, exact_eprod)

    assert pytest.approx(lhs) == rhs


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_non_linear_10_irregular_deformed(corner_vals: npt.ArrayLike) -> None:
    """Check that non-linear inter-product of 1-form with 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0 * v1**3, -(v0**2) * v1),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_1)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_0)
    w = g.weight
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_1)
    h = u.weight

    system = KFormSystem(
        (w * (u ^ omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        (h * g.derivative) == h @ 0,
        sorting=lambda f: f.order + ord(f.label[0]),
    )

    N = 6
    N2 = 10
    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array(corner_vals, np.float64)
    fem_space = ElementFemSpace2D(basis_2d, corners)
    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, fem_space, omega_exact
    )

    u_proj = element_primal_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, u_exact)

    emat = compute_system_matrix_nonlin(
        system,
        fem_space,
        np.concatenate(
            (
                np.zeros(UnknownFormOrder.FORM_ORDER_0.full_unknown_count(N, N)),
                omega_proj,
                u_proj,
            ),
            dtype=np.float64,
        ),
    )

    fmat = emat[: (N + 1) * (N + 1), -2 * (N + 1) * N :]
    gmat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) : -2 * (N + 1) * N]

    real_dofs = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_0,
        fem_space,
        exact_interior_prod_1(u_exact, omega_exact),
    )

    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    assert pytest.approx(real_dofs) == v2
    assert pytest.approx(real_dofs) == v1


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_dual_non_linear_10_irregular_deformed(corner_vals: npt.ArrayLike) -> None:
    """Check that non-linear inter-product of 1-form with 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0 * v1**3, -(v0**2) * v1),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_1)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_2)
    w = g.weight
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_1)
    h = u.weight

    system = KFormSystem(
        (w * (u ^ ~omega)) == w @ 0,
        (v.derivative * g) == v @ 0,
        (h.derivative * g) == h @ 0,
        sorting=lambda f: f.order + ord(f.label[0]),
    )
    # print(system)

    N = 6
    N2 = 10

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    corners = np.array(corner_vals, np.float64)

    fem_space = ElementFemSpace2D(basis_2d, corners)
    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_1, fem_space, omega_exact
    )

    u_proj = element_primal_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, u_exact)

    emat = compute_system_matrix_nonlin(
        system,
        fem_space,
        np.concatenate(
            (
                np.zeros(UnknownFormOrder.FORM_ORDER_2.full_unknown_count(N, N)),
                omega_proj,
                u_proj,
            ),
            dtype=np.float64,
        ),
    )

    real_dofs = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_2,
        fem_space,
        exact_interior_prod_1_dual(u_exact, omega_exact),
    )
    fmat = emat[: N * N, -2 * (N + 1) * N :]
    gmat = emat[: N * N, N * N : -2 * (N + 1) * N]
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    assert real_dofs == pytest.approx(v1)
    assert real_dofs == pytest.approx(v2)


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_21(corner_vals: npt.ArrayLike) -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return v0 - v1**3

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_2)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_1)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == (v @ 0),
        sorting=lambda f: f.order,
    )

    N = 5
    N2 = 10
    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    corners = np.array(corner_vals, np.float64)
    fem_space = ElementFemSpace2D(basis_2d, corners)
    emat = compute_system_matrix_lin(system, fem_space)
    emat = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]

    exact_eprod = exact_interior_prod_2(u_exact, omega_exact)

    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, fem_space, omega_exact
    )
    lhs = emat @ omega_proj
    rhs = element_dual_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, exact_eprod)

    assert pytest.approx(lhs) == rhs


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_dual_advect_21_undeformed(corner_vals: npt.ArrayLike) -> None:
    """Check dual interior product of a 2-form with a 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return v0 - v1**3
        # return 1 + 0 * (v0 - v1**3)

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_0)
    v = omega.weight
    g = KFormUnknown("g", UnknownFormOrder.FORM_ORDER_1)
    w = g.weight
    system = KFormSystem(
        (w * (u_exact * (~omega))) == w @ 0,
        (v.derivative * g) == (v @ 0),
        sorting=lambda f: 5 - f.order,
    )

    N = 6
    N2 = 10

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)
    corners = np.array(corner_vals, np.float64)
    fem_space = ElementFemSpace2D(basis_2d, corners)
    emat = compute_system_matrix_lin(system, fem_space)

    emat = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]

    exact_eprod = exact_interior_prod_2_dual(u_exact, omega_exact)

    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, fem_space, omega_exact
    )
    lhs = emat @ omega_proj
    rhs = element_dual_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, exact_eprod)

    assert pytest.approx(lhs) == rhs


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_non_linear_21_irregular_deformed(corner_vals: npt.ArrayLike) -> None:
    """Check that non-linear inter-product of 2-form with 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.astype(v0 * v1**3, np.float64, copy=False)

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_2)
    v = omega.weight
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_1)
    h = u.weight
    corners = np.array(corner_vals, np.float64)

    system = KFormSystem(
        (h * (u ^ omega)) == h @ 0,
        (v * u.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )
    N = 6
    N2 = 10
    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    fem_space = ElementFemSpace2D(basis_2d, corners)
    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_2, fem_space, omega_exact
    )

    u_proj = element_primal_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, u_exact)

    emat = compute_system_matrix_nonlin(
        system, fem_space, np.concatenate((u_proj, omega_proj), dtype=np.float64)
    )

    real_dofs = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_1,
        fem_space,
        exact_interior_prod_2(u_exact, omega_exact),
    )
    fmat = emat[: 2 * (N + 1) * N, : 2 * (N + 1) * N]
    gmat = emat[: 2 * (N + 1) * N, 2 * (N + 1) * N :]
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj

    assert pytest.approx(real_dofs) == v1
    assert pytest.approx(real_dofs) == v2


@pytest.mark.parametrize("corner_vals", _CORNER_TEST_VALUES)
def test_advect_dual_non_linear_21_irregular_deformed(corner_vals: npt.ArrayLike) -> None:
    """Check that non-linear inter-product of 1-form with 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.astype(v0 * v1**3, np.float64, copy=False)

    omega = KFormUnknown("omega", UnknownFormOrder.FORM_ORDER_0)
    v = omega.weight
    u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_1)
    h = u.weight

    system = KFormSystem(
        (h * (u ^ ~omega)) == h @ 0,
        (v.derivative * u) == v @ 0,
        sorting=lambda f: f.order,
    )

    N = 6
    N2 = 10

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    corners = np.array(corner_vals, np.float64)

    fem_space = ElementFemSpace2D(basis_2d, corners)
    omega_proj = element_primal_dofs(
        UnknownFormOrder.FORM_ORDER_0, fem_space, omega_exact
    )

    u_proj = element_primal_dofs(UnknownFormOrder.FORM_ORDER_1, fem_space, u_exact)

    emat = compute_system_matrix_nonlin(
        system, fem_space, np.concatenate((omega_proj, u_proj), dtype=np.float64)
    )

    fmat = emat[(N + 1) ** 2 :, (N + 1) ** 2 :]
    gmat = emat[(N + 1) ** 2 :, : (N + 1) ** 2]
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    real_dofs = element_dual_dofs(
        UnknownFormOrder.FORM_ORDER_1,
        fem_space,
        exact_interior_prod_2_dual(u_exact, omega_exact),
    )

    assert pytest.approx(real_dofs) == v1
    assert pytest.approx(real_dofs) == v2
