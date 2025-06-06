"""Check that interior product implementation actually works."""

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
from mfv2d.mimetic2d import BasisCache, ElementLeaf2D


def test_advect_21_deformed() -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown(2, "omega", 2)
    v = omega.weight
    g = KFormUnknown(2, "g", 1)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == (v @ 0),
        sorting=lambda f: f.order,
    )

    compiled = CompiledSystem(system)
    vector_fields = system.vector_fields

    N = 5
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        assert callable(vec_fld)
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    (emat,) = compute_element_matrices(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )

    int_rule = IntegrationRule1D(N2)
    basis1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis1d, basis1d)

    new_mat = compute_element_matrix(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    mat_1 = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    mat_2 = new_mat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    # with np.printoptions(precision=2):
    #     print(mat_1)
    #     print(mat_2)
    assert pytest.approx(mat_1) == mat_2


def test_advect_10_deformed() -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly."""

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown(2, "omega", 1)
    v = omega.weight
    g = KFormUnknown(2, "g", 0)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )

    compiled = CompiledSystem(system)
    vector_fields = system.vector_fields

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-2, -2), (+1, -2), (+1, +2), (-1, +2))
    # e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        assert callable(vec_fld)
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    (emat,) = compute_element_matrices(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )

    int_rule = IntegrationRule1D(N2)
    basis1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis1d, basis1d)

    new_mat = compute_element_matrix(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    mat_1 = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]
    mat_2 = new_mat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]
    # with np.printoptions(precision=2):
    #     print(mat_1)
    #     print(mat_2)
    assert pytest.approx(mat_1) == mat_2


def test_dual_advect_21_irregular() -> None:
    """Check that dual interior product of a 2-form with a 1-form is computed correctly.

    Here the element is deformed in an irregular way.
    """

    def u_exact(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute exact field."""
        v0 = np.asarray(x)
        v1 = np.asarray(y)
        return np.stack(
            # (0 + 0 * (v0 + v1), 1 + 0 * (-v0 * v1**3)),
            (v0**2 * v1, -v0 * v1**3),
            axis=-1,
            dtype=np.float64,
        )

    omega = KFormUnknown(2, "omega", 0)
    v = omega.weight
    g = KFormUnknown(2, "g", 1)
    w = g.weight
    system = KFormSystem(
        (w * (u_exact * (~omega))) == w @ 0,
        (v.derivative * g) == (v @ 0),
        sorting=lambda f: 5 - f.order,
    )

    compiled = CompiledSystem(system)
    vector_fields = system.vector_fields

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-2, -2), (+1, -2), (+1, +2), (-1, +2))
    # e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        assert callable(vec_fld)
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    (emat,) = compute_element_matrices(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )

    int_rule = IntegrationRule1D(N2)
    basis1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis1d, basis1d)

    new_mat = compute_element_matrix(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    mat_1 = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    mat_2 = new_mat[: 2 * N * (N + 1), 2 * N * (N + 1) :]

    # with np.printoptions(precision=2):
    #     print(mat_1)
    #     print(mat_2)
    assert pytest.approx(mat_1) == mat_2


def test_dual_advect_10_deformed() -> None:
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

    omega = KFormUnknown(2, "omega", 1)
    v = omega.weight
    g = KFormUnknown(2, "g", 2)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * ~omega)) == w @ 0,
        (v.derivative * g) == v @ 0,
        sorting=lambda f: f.order,
    )

    compiled = CompiledSystem(system)
    vector_fields = system.vector_fields

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -2), (+2, +0), (+1.75, +0.75), (+1.0, +1.0))
    # e = ElementLeaf2D(None, N, (-2, -2), (+1, -2), (+1, +2), (-1, +2))
    # e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        assert callable(vec_fld)
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    (emat,) = compute_element_matrices(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )

    int_rule = IntegrationRule1D(N2)
    basis1d = Basis1D(N, int_rule)
    basis_2d = Basis2D(basis1d, basis1d)

    new_mat = compute_element_matrix(
        tuple(form.order for form in system.unknown_forms),
        compiled.lhs_full,
        np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left], np.float64),
        vec_fields,
        basis_2d,
    )

    mat_1 = emat[2 * (N + 1) * N :, : 2 * (N + 1) * N]
    mat_2 = new_mat[2 * (N + 1) * N :, : 2 * (N + 1) * N]

    # with np.printoptions(precision=2):
    #     print(mat_1)
    #     print(mat_2)
    assert pytest.approx(mat_1) == mat_2


# if __name__ == "__main__":
#     test_advect_21_deformed()
#     test_advect_10_deformed()
#     test_dual_advect_21_irregular()
#     test_dual_advect_10_deformed()
