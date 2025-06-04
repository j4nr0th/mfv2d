"""Check that element system is correctly computed."""

import timeit

import numpy as np
import pytest
from mfv2d._mfv2d import compute_element_matrices, compute_element_matrix
from mfv2d.eval import CompiledSystem
from mfv2d.kform import KFormSystem, KFormUnknown
from mfv2d.mimetic2d import Basis1D, Basis2D, BasisCache, IntegrationRule1D


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
