"""Tests concerning explicit evaluation of terms versus using a matrix."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import compute_element_explicit, compute_element_matrices
from mfv2d.eval import translate_system
from mfv2d.kform import KFormSystem, KFormUnknown
from mfv2d.mimetic2d import (
    BasisCache,
    ElementLeaf2D,
    rhs_2d_element_projection,
)


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

    vor = KFormUnknown(2, "vor", 0)
    vel = KFormUnknown(2, "vel", 1)
    pre = KFormUnknown(2, "pre", 2)

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
    func_dict = {vor: vor_exact, vel: vel_exact}
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

    sys_mat = compute_element_matrices(
        tuple(form.order for form in system.unknown_forms),
        codes,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )[0]

    proj_vor = np.linalg.solve(
        e.mass_matrix_node(cache), rhs_2d_element_projection(w_vor @ vor_exact, e, cache)
    )
    proj_vel = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(w_vel @ vel_exact, e, cache)
    )
    proj_pre = np.linalg.solve(
        e.mass_matrix_surface(cache),
        rhs_2d_element_projection(w_pre @ pre_exact, e, cache),
    )

    exact_lhs = np.concatenate((proj_vor, proj_vel, proj_pre), dtype=np.float64)

    explicit_rhs = compute_element_explicit(
        exact_lhs,
        np.array((0,), np.uint32),
        tuple(form.order for form in system.unknown_forms),
        codes,
        np.array([e.bottom_left], np.float64),
        np.array([e.bottom_right], np.float64),
        np.array([e.top_right], np.float64),
        np.array([e.top_left], np.float64),
        np.array((e.order,), np.uint32),
        vec_fields,
        vec_field_offsets,
        (cache.c_serialization(),),
    )[0]

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
    momentum_rhs = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs[(N + 1) ** 2 : (N + 1) ** 2 + 2 * N * (N + 1)]
    )
    proj_momentum_rhs = np.linalg.solve(
        e.mass_matrix_edge(cache),
        rhs_2d_element_projection(w_vel @ exact_momentum, e, cache),
    )
    assert np.abs(momentum_rhs - proj_momentum_rhs).max() < 1e-6
