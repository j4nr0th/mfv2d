"""Check that element system is correctly computed."""

import numpy as np
import pytest
from mfv2d._mfv2d import compute_element_matrices, compute_element_matrix
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KFormSystem, KFormUnknown
from mfv2d.mimetic2d import Basis1D, Basis2D, BasisCache, IntegrationRule1D
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
        n,
        n,
        tuple(),
        basis_2d.basis_xi.node,
        basis_2d.basis_xi.edge,
        basis_2d.basis_xi.rule.weights,
        basis_2d.basis_xi.rule.nodes,
        basis_2d.basis_eta.node,
        basis_2d.basis_eta.edge,
        basis_2d.basis_eta.rule.weights,
        basis_2d.basis_eta.rule.nodes,
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
        n,
        n,
        tuple(),
        basis_2d.basis_xi.node,
        basis_2d.basis_xi.edge,
        basis_2d.basis_xi.rule.weights,
        basis_2d.basis_xi.rule.nodes,
        basis_2d.basis_eta.node,
        basis_2d.basis_eta.edge,
        basis_2d.basis_eta.rule.weights,
        basis_2d.basis_eta.rule.nodes,
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
        n,
        n,
        tuple(),
        basis_2d.basis_xi.node,
        basis_2d.basis_xi.edge,
        basis_2d.basis_xi.rule.weights,
        basis_2d.basis_xi.rule.nodes,
        basis_2d.basis_eta.node,
        basis_2d.basis_eta.edge,
        basis_2d.basis_eta.rule.weights,
        basis_2d.basis_eta.rule.nodes,
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
        corners[None, :],
        np.array([(n, n)], np.uint32),
        np.array([(n, n)], np.uint32),
        np.array([0, (n + 1) ** 2, 2 * n * (n + 1), n**2]).cumsum(),
        np.zeros((100, 100)),
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
        n,
        n,
        fields,
        basis_2d.basis_xi.node,
        basis_2d.basis_xi.edge,
        basis_2d.basis_xi.rule.weights,
        basis_2d.basis_xi.rule.nodes,
        basis_2d.basis_eta.node,
        basis_2d.basis_eta.edge,
        basis_2d.basis_eta.rule.weights,
        basis_2d.basis_eta.rule.nodes,
    )
    # with np.printoptions(precision=2):
    #     print(mat_correct)
    #     print(mat_new)

    assert np.allclose(mat_correct, mat_new)


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
