"""Check that interior product implementation actually works."""

from typing import Callable

import numpy as np
import numpy.typing as npt
from interplib._mimetic import compute_element_matrices_2
from interplib.kforms import KFormSystem, KFormUnknown
from interplib.kforms.eval import MatOp, MatOpCode, _ctranslate, translate_equation
from interplib.mimetic import ElementLeaf2D, Manifold2D
from interplib.mimetic.mimetic2d import BasisCache, rhs_2d_element_projection

type Function2D = Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.float64]]


def test_advect_21_undeformed() -> None:
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

    def exact_interior_prod_2(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            scl_field = form2(x, y)
            out = np.empty_like(vec_field)
            out[..., 0] = vec_field[..., 1] * scl_field
            out[..., 1] = vec_field[..., 0] * scl_field
            return out

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 2)
    v = omega.weight
    g = KFormUnknown(man, "g", 1)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == (v @ 0),
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 5
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_2(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_surface(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_edge(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )
    assert np.max(np.abs(lhs - rhs)) < 1e-15

    # print(lhs)
    # print(rhs)

    # eta_1 = lhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # eta_2 = rhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # xi_1 = lhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))
    # xi_2 = rhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))

    # plt.figure()
    # plt.imshow(eta_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(eta_2)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(xi_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(xi_2)
    # plt.colorbar()
    # plt.show()


def test_advect_10_undeformed() -> None:
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

    def exact_interior_prod_1(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            form_field = form2(x, y)
            return (
                form_field[..., 0] * vec_field[..., 0]
                + form_field[..., 1] * vec_field[..., 1]
            )

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 1)
    v = omega.weight
    g = KFormUnknown(man, "g", 0)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )
    print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 6
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_1(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_node(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_node(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )

    # print(lhs)
    # print(rhs)

    # v_1 = lhs.reshape((N + 1, N + 1))
    # v_2 = rhs.reshape((N + 1, N + 1))

    # plt.figure()
    # plt.imshow(v_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(v_2)
    # plt.colorbar()
    # plt.show()
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_advect_21_regular_deformed_1() -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly.

    Here deformation is applied by just scaling, without shear.
    """

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

    def exact_interior_prod_2(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            scl_field = form2(x, y)
            out = np.empty_like(vec_field)
            out[..., 0] = vec_field[..., 1] * scl_field
            out[..., 1] = vec_field[..., 0] * scl_field
            return out

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 2)
    v = omega.weight
    g = KFormUnknown(man, "g", 1)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == (v @ 0),
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 5
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-0.1, -2), (+0.1, -2), (+0.1, +2), (-0.1, +2))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_2(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_surface(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_edge(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )

    # print(lhs / rhs)

    # eta_1 = lhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # eta_2 = rhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # xi_1 = lhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))
    # xi_2 = rhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))

    # plt.figure()
    # plt.imshow(eta_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(eta_2)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(xi_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(xi_2)
    # plt.colorbar()
    # plt.show()
    # print(np.max(np.abs(lhs - rhs)))
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_advect_21_regular_deformed_2() -> None:
    """Check that interior product of a 2-form with a 1-form is computed correctly.

    Here deformation is applied by just scaling, without shear.
    """

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

    def exact_interior_prod_2(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            scl_field = form2(x, y)
            out = np.empty_like(vec_field)
            out[..., 0] = vec_field[..., 1] * scl_field
            out[..., 1] = vec_field[..., 0] * scl_field
            return out

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 2)
    v = omega.weight
    g = KFormUnknown(man, "g", 1)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == (v @ 0),
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 5
    N2 = 10

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-2, -0.1), (+2, -0.1), (+2, +0.1), (-2, +0.1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: 2 * N * (N + 1), 2 * N * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_2(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_surface(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_edge(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )

    # print(lhs / rhs)

    # eta_1 = lhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # eta_2 = rhs[0 * N * (N + 1) : 1 * N * (N + 1)].reshape((N + 1, N))
    # xi_1 = lhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))
    # xi_2 = rhs[1 * N * (N + 1) : 2 * N * (N + 1)].reshape((N, N + 1))

    # plt.figure()
    # plt.imshow(eta_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(eta_2)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.imshow(xi_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(xi_2)
    # plt.colorbar()
    # plt.show()
    # print(np.max(np.abs(lhs - rhs)))
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_advect_10_refgular_deformed_1() -> None:
    """Check that interior product of a 1-form with a 1-form is computed correctly.

    Here deformation is applied by just scaling, without shear.
    """

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

    def exact_interior_prod_1(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            form_field = form2(x, y)
            return (
                form_field[..., 0] * vec_field[..., 0]
                + form_field[..., 1] * vec_field[..., 1]
            )

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 1)
    v = omega.weight
    g = KFormUnknown(man, "g", 0)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 8
    N2 = 20

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-0.1, -2), (+0.1, -2), (+0.1, +2), (-0.1, +2))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_1(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_node(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_node(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )

    # print(lhs / rhs)

    # v_1 = lhs.reshape((N + 1, N + 1))
    # v_2 = rhs.reshape((N + 1, N + 1))

    # plt.figure()
    # plt.imshow(v_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(v_2)
    # plt.colorbar()
    # plt.show()
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_advect_10_refgular_deformed_2() -> None:
    """Check that interior product of a 1-form with a 1-form is computed correctly.

    Here deformation is applied by just scaling, without shear.
    """

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

    def exact_interior_prod_1(vec: Function2D, form2: Function2D) -> Function2D:
        """Create an interior product function."""

        def wrapped(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
            """Compute interior product."""
            vec_field = vec(x, y)
            form_field = form2(x, y)
            return (
                form_field[..., 0] * vec_field[..., 0]
                + form_field[..., 1] * vec_field[..., 1]
            )

        return wrapped

    man = Manifold2D.from_regular(4, ((1, 2), (2, 3), (3, 4), (4, 1)), ((1, 2, 3, 4),))

    omega = KFormUnknown(man, "omega", 1)
    v = omega.weight
    g = KFormUnknown(man, "g", 0)
    w = g.weight

    system = KFormSystem(
        (w * (u_exact * omega)) == w @ 0,
        (v * g.derivative) == v @ 0,
        sorting=lambda f: f.order,
    )
    # print(system)

    vector_fields = system.vector_fields
    bytecodes = [
        translate_equation(eq.left, vector_fields, simplify=True)
        for eq in system.equations
    ]

    codes: list[list[None | list[MatOpCode | float | int]]] = list()
    for bite in bytecodes:
        row: list[list[MatOpCode | float | int] | None] = list()
        expr_row: list[tuple[MatOp, ...] | None] = list()
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
                expr_row.append(tuple(bite[f]))
            else:
                row.append(None)
                expr_row.append(None)

        codes.append(row)

    N = 8
    N2 = 20

    cache = BasisCache(N, N2)

    # Compute vector fields at integration points for leaf elements
    vec_field_lists: tuple[list[npt.NDArray[np.float64]], ...] = tuple(
        list() for _ in vector_fields
    )
    vec_field_offsets = np.zeros(2, np.uint64)
    e = ElementLeaf2D(None, N, (-2, -0.1), (+2, -0.1), (+2, +0.1), (-2, +0.1))

    x = e.poly_x(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    y = e.poly_y(cache.int_nodes_1d[None, :], cache.int_nodes_1d[:, None])
    for i, vec_fld in enumerate(vector_fields):
        vec_field_lists[i].append(np.reshape(vec_fld(x, y), (-1, 2)))
    vec_field_offsets[1] = vec_field_offsets[0] + (cache.integration_order + 1) ** 2
    vec_fields = tuple(
        np.concatenate(vfl, axis=0, dtype=np.float64) for vfl in vec_field_lists
    )
    del vec_field_lists

    mats = compute_element_matrices_2(
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
    )

    emat = mats[0]
    emat = emat[: (N + 1) * (N + 1), (N + 1) * (N + 1) :]
    # print(emat)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(emat)
    # plt.colorbar()
    # plt.show()

    exact_eprod = exact_interior_prod_1(u_exact, omega_exact)

    omega_proj = np.linalg.solve(
        e.mass_matrix_edge(cache), rhs_2d_element_projection(v @ omega_exact, e, cache)
    )
    lhs = np.linalg.solve(e.mass_matrix_node(cache), emat @ omega_proj)
    rhs = np.linalg.solve(
        e.mass_matrix_node(cache), rhs_2d_element_projection(w @ exact_eprod, e, cache)
    )

    # print(lhs / rhs)

    # v_1 = lhs.reshape((N + 1, N + 1))
    # v_2 = rhs.reshape((N + 1, N + 1))

    # plt.figure()
    # plt.imshow(v_1)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(v_2)
    # plt.colorbar()
    # plt.show()
    assert np.max(np.abs(lhs - rhs)) < 1e-14
