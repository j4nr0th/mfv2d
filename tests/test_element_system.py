"""Check that element system is correctly computed."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    IntegrationRule1D,
    compute_element_matrices,
    compute_element_matrix,
)
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KFormSystem, KFormUnknown
from mfv2d.mimetic2d import (
    BasisCache,
    ElementLeaf2D,
    rhs_2d_element_projection,
)
from mfv2d.solve_system import compute_vector_fields_nonlin


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def test_poisson_direct(n: int) -> None:
    """Check that matrix for the direct formulation of the Poisson equation is correct."""
    u = KFormUnknown(2, "u", 0)
    w = u.weight

    system = KFormSystem(w * u == 0)
    compiled = CompiledSystem(system)

    basis = BasisCache(n, n + 2)

    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )

    (mat_correct,) = compute_element_matrices(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners[0, :][None, :],
        corners[1, :][None, :],
        corners[2, :][None, :],
        corners[3, :][None, :],
        np.array((n,), np.uint32),
        tuple(),
        np.zeros(2, np.uint64),
        (basis.c_serialization(),),
    )

    int_rule = IntegrationRule1D(n + 2)
    b1 = Basis1D(n, int_rule)
    basis_2d = Basis2D(b1, b1)

    mat_new = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners,
        tuple(),
        basis_2d,
    )
    assert np.allclose(mat_correct, mat_new)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def test_poisson_mixed(n: int) -> None:
    """Check that matrix for the mixed formulation of the Poisson equation is correct."""
    u = KFormUnknown(2, "u", 1)
    v = u.weight
    phi = KFormUnknown(2, "phi", 2)
    omega = phi.weight

    system = KFormSystem(
        v.derivative * phi + v * u == 0,
        omega * u.derivative == 0,
        sorting=lambda f: f.order,
    )
    compiled = CompiledSystem(system)

    basis = BasisCache(n, n + 2)

    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )

    (mat_correct,) = compute_element_matrices(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners[0, :][None, :],
        corners[1, :][None, :],
        corners[2, :][None, :],
        corners[3, :][None, :],
        np.array((n,), np.uint32),
        tuple(),
        np.zeros(2, np.uint64),
        (basis.c_serialization(),),
    )

    int_rule = IntegrationRule1D(n + 2)
    b1 = Basis1D(n, int_rule)
    basis_2d = Basis2D(b1, b1)

    mat_new = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners,
        tuple(),
        basis_2d,
    )
    assert np.allclose(mat_correct, mat_new)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def test_stokes(n: int) -> None:
    """Check that matrix for the Stokes equation is correct."""
    f = KFormUnknown(2, "f", 0)
    g = f.weight
    u = KFormUnknown(2, "u", 1)
    v = u.weight
    phi = KFormUnknown(2, "phi", 2)
    omega = phi.weight

    system = KFormSystem(
        g.derivative * u + g * f == 0,
        v.derivative * phi + v * f.derivative == 0,
        omega * u.derivative == 0,
        sorting=lambda f: f.order,
    )
    compiled = CompiledSystem(system)

    basis = BasisCache(n, n + 2)

    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )

    (mat_correct,) = compute_element_matrices(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners[0, :][None, :],
        corners[1, :][None, :],
        corners[2, :][None, :],
        corners[3, :][None, :],
        np.array((n,), np.uint32),
        tuple(),
        np.zeros(2, np.uint64),
        (basis.c_serialization(),),
    )

    int_rule = IntegrationRule1D(n + 2)
    b1 = Basis1D(n, int_rule)
    basis_2d = Basis2D(b1, b1)

    mat_new = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners,
        tuple(),
        basis_2d,
    )
    assert np.allclose(mat_correct, mat_new)


@pytest.mark.parametrize("n", (1, 2, 4, 5, 7))
def test_lin_navier_stokes(n: int) -> None:
    """Check that matrix for the linearized Navier-Stokes equation is correct."""
    f = KFormUnknown(2, "f", 0)
    g = f.weight
    u = KFormUnknown(2, "u", 1)
    v = u.weight
    phi = KFormUnknown(2, "phi", 2)
    omega = phi.weight
    re = 1e10

    def my_nice_velocity(x, y):
        return np.stack((np.sin(x) * np.sin(y), np.cos(x) * np.cos(y)), axis=-1)

    system = KFormSystem(
        g.derivative * u + g * f == 0,
        v.derivative * phi + 1 / re * (v * f.derivative) + (v * (my_nice_velocity * (~f)))
        == 0,
        omega * u.derivative == 0,
        sorting=lambda f: f.order,
    )
    compiled = CompiledSystem(system)

    basis = BasisCache(n, n + 2)

    corners = np.array(
        (
            (-1, -1),
            (+1, -1),
            (+1, +1),
            (-1, +1),
        ),
        np.float64,
    )

    offsets, fields = compute_vector_fields_nonlin(
        system,
        [0],
        {n: basis},
        system.vector_fields,
        corners[None, :],  # type: ignore
        np.array([(n, n)], np.uint32),  # type: ignore
        np.array([(n, n)], np.uint32),  # type: ignore
        np.array([0, (n + 1) ** 2, 2 * n * (n + 1), n**2]).cumsum(),  # type: ignore
        np.zeros((100, 100)),  # type: ignore
    )

    (mat_correct,) = compute_element_matrices(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners[0, :][None, :],
        corners[1, :][None, :],
        corners[2, :][None, :],
        corners[3, :][None, :],
        np.array((n,), np.uint32),
        fields,
        offsets,
        (basis.c_serialization(),),
    )

    int_rule = IntegrationRule1D(n + 2)
    b1 = Basis1D(n, int_rule)
    basis_2d = Basis2D(b1, b1)

    mat_new = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.lhs_full,
        corners,
        fields,
        basis_2d,
    )
    # with np.printoptions(precision=2):
    #     print(mat_correct)
    #     print(mat_new)

    assert np.allclose(mat_correct, mat_new)


type Function2D = Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]


def exact_interior_prod_2_dual(vec: Function2D, form2: Function2D) -> Function2D:
    """Create an interior product dual function."""

    def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute interior product."""
        vec_field = vec(x, y)
        scl_field = form2(x, y)
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


def test_advect_non_linear_10_irregular_deformed() -> None:
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

    omega = KFormUnknown(2, "omega", 1)
    v = omega.weight
    g = KFormUnknown(2, "g", 0)
    w = g.weight
    u = KFormUnknown(2, "u", 1)
    h = u.weight

    system = KFormSystem(
        (w * (u ^ omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        (h * g.derivative) == h @ 0,
        sorting=lambda f: f.order + ord(f.label[0]),
    )
    # print(system)

    vector_fields = system.vector_fields
    compiled = CompiledSystem(system)

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    func_dict = {omega: omega_exact, u: u_exact}
    for i, vec_fld in enumerate(vector_fields):
        assert type(vec_fld) is KFormUnknown
        assert not callable(vec_fld)
        fn = func_dict[vec_fld]
        vf = fn(x, y)
        if vec_fld.order != 1:
            vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
        vf = np.reshape(vf, (-1, 2))
        vec_field_lists[i].append(vf)
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.nonlin_codes,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    fmat = emat[: (N + 1) * (N + 1), -2 * (N + 1) * N :]
    gmat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) : -2 * (N + 1) * N]

    omega_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )

    u_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ u_exact, e, cache)
    )
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    assert np.max(np.abs(v1 - v2)) < 1e-15


def test_advect_dual_non_linear_10_irregular_deformed() -> None:
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

    omega = KFormUnknown(2, "omega", 1)
    v = omega.weight
    g = KFormUnknown(2, "g", 2)
    w = g.weight
    u = KFormUnknown(2, "u", 1)
    h = u.weight

    system = KFormSystem(
        (w * (u ^ ~omega)) == w @ 0,
        (v.derivative * g) == v @ 0,
        (h.derivative * g) == h @ 0,
        sorting=lambda f: f.order + ord(f.label[0]),
    )
    # print(system)

    vector_fields = system.vector_fields
    compiled = CompiledSystem(system)

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    func_dict = {omega: omega_exact, u: u_exact}
    for i, vec_fld in enumerate(vector_fields):
        assert type(vec_fld) is KFormUnknown
        assert not callable(vec_fld)
        fn = func_dict[vec_fld]
        vf = fn(x, y)
        if vec_fld.order != 1:
            vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
        vf = np.reshape(vf, (-1, 2))
        vec_field_lists[i].append(vf)
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.nonlin_codes,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    fmat = emat[: N * N, -2 * (N + 1) * N :]
    gmat = emat[: N * N, N * N : -2 * (N + 1) * N]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    omega_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )

    u_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ u_exact, e, cache)
    )
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    assert np.max(np.abs(v1 - v2)) < 1e-13


def test_advect_non_linear_21_irregular_deformed() -> None:
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
        return np.astype(v0 * v1**3, np.float64, copy=False)

    omega = KFormUnknown(2, "omega", 2)
    v = omega.weight
    u = KFormUnknown(2, "u", 1)
    h = u.weight

    system = KFormSystem(
        (h * (u ^ omega)) == h @ 0,
        (v * u.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    compiled = CompiledSystem(system)

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-2, -2), (+2, -2), (+2, +2), (-2, +2))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    func_dict = {omega: omega_exact, u: u_exact}
    for i, vec_fld in enumerate(vector_fields):
        assert type(vec_fld) is KFormUnknown
        assert not callable(vec_fld)
        fn = func_dict[vec_fld]
        vf = fn(x, y)
        if vec_fld.order != 1:
            vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
        vf = np.reshape(vf, (-1, 2))
        vec_field_lists[i].append(vf)
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.nonlin_codes,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    fmat = emat[: N**2, : 2 * (N + 1) * N]
    gmat = emat[: N**2, 2 * (N + 1) * N :]

    omega_proj = np.linalg.solve(
        e.mass_matrix_surface(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )

    u_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(h @ u_exact, e, cache)
    )
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj
    assert np.max(np.abs(v1 - v2)) < 1e-13


def test_advect_dual_non_linear_21_irregular_deformed() -> None:
    """Check that non-linear inter-product of 1-form with 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (np.sin(v0) * np.cos(v1), np.cos(v0) * np.sin(v1)),
            (0 + 0 * v0**2 * v1, 1 + 0 * -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    def omega_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.astype(1 + 0 * v0 * v1**3, np.float64, copy=False)

    omega = KFormUnknown(2, "omega", 0)
    v = omega.weight
    u = KFormUnknown(2, "u", 1)
    h = u.weight

    system = KFormSystem(
        (h * (u ^ ~omega)) == h @ 0,
        (v.derivative * u) == v @ 0,
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    compiled = CompiledSystem(system)

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    int_rule = IntegrationRule1D(N2)
    basis_1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis_1d, basis_1d)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    func_dict = {omega: omega_exact, u: u_exact}
    for i, vec_fld in enumerate(vector_fields):
        assert type(vec_fld) is KFormUnknown
        assert not callable(vec_fld)
        fn = func_dict[vec_fld]
        vf = fn(x, y)
        if vec_fld.order != 1:
            vf = np.stack((vf, np.zeros_like(vf)), axis=-1, dtype=np.float64)
        vf = np.reshape(vf, (-1, 2))
        vec_field_lists[i].append(vf)
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists
    assert compiled.nonlin_codes is not None
    emat = compute_element_matrix(
        [form.order for form in system.unknown_forms],
        compiled.nonlin_codes,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    fmat = emat[(N + 1) ** 2 :, (N + 1) ** 2 :]
    gmat = emat[(N + 1) ** 2 :, : (N + 1) ** 2]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(gmat)
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(fmat)
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(e.mass_matrix_edge(cache))
    # plt.colorbar()
    # plt.show()

    omega_proj = np.linalg.solve(
        e.mass_matrix_node(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )

    u_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(h @ u_exact, e, cache)
    )
    v1 = gmat @ omega_proj
    v2 = fmat @ u_proj

    # p1 = np.linalg.solve(e.mass_matrix_edge(cache), v1)
    # p2 = np.linalg.solve(e.mass_matrix_edge(cache), v2)

    assert np.max(np.abs(v1 - v2)) < 1e-13


# if __name__ == "__main__":
#     sizes = (1, 2, 4, 5, 7)
#     for n in sizes:
#         test_poisson_direct(n)
#     for n in sizes:
#         test_poisson_mixed(n)
#     for n in sizes:
#         test_stokes(n)
#     for n in sizes:
#         test_lin_navier_stokes(n)

#     test_advect_non_linear_10_irregular_deformed()
#     test_advect_dual_non_linear_10_irregular_deformed()
#     test_advect_non_linear_21_irregular_deformed()
#     test_advect_dual_non_linear_21_irregular_deformed()
