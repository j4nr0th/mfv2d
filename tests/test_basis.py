"""Check basis and integration rules work."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import (
    Basis1D,
    Basis2D,
    ElementFemSpace2D,
    IntegrationRule1D,
    compute_gll,
    dlagrange1d,
    lagrange1d,
)
from mfv2d.kform import UnknownFormOrder


@dataclass(frozen=True)
class OldIntegrationRule1D:
    """Type used to cache integration nodes and weights."""

    order: int
    nodes: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]

    def __init__(self, order: int, /) -> None:
        nodes, weights = compute_gll(order)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class OldBasis1D:
    """Type used to store 1D basis information."""

    order: int
    node: npt.NDArray[np.float64]
    edge: npt.NDArray[np.float64]
    rule: OldIntegrationRule1D

    def __init__(self, order: int, rule: OldIntegrationRule1D) -> None:
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "rule", rule)
        gll_nodes, _ = compute_gll(order)
        value_nodal = lagrange1d(gll_nodes, rule.nodes)
        object.__setattr__(self, "node", np.ascontiguousarray(value_nodal.T, np.float64))
        dvalue_nodal = dlagrange1d(gll_nodes, rule.nodes)
        value_edge = np.cumsum(-dvalue_nodal[:, :-1], axis=-1)
        object.__setattr__(self, "edge", np.ascontiguousarray(value_edge.T, np.float64))


@pytest.mark.parametrize("n", (1, 2, 4, 6, 7))
def test_old_int_rule(n: int) -> None:
    """Check integration rule works as expected."""
    old_rule = OldIntegrationRule1D(n)
    new_rule = IntegrationRule1D(n)

    assert pytest.approx(old_rule.nodes) == new_rule.nodes
    assert pytest.approx(old_rule.weights) == new_rule.weights
    assert old_rule.order == new_rule.order


@pytest.mark.parametrize(("nb", "nr"), ((1, 1), (1, 5), (3, 4), (4, 10)))
def test_old_basis(nb: int, nr: int) -> None:
    """Test old basis work."""
    old_rule = OldIntegrationRule1D(nr)
    new_rule = IntegrationRule1D(nr)

    old_basis = OldBasis1D(nb, old_rule)
    new_basis = Basis1D(nb, new_rule)

    assert pytest.approx(new_basis.node) == old_basis.node
    assert pytest.approx(new_basis.edge) == old_basis.edge
    assert new_basis.order == old_basis.order


@pytest.mark.parametrize(("nh", "nv"), ((1, 1), (2, 4), (4, 2), (3, 3), (8, 3)))
def test_mass_matrix_inverses(nh: int, nv: int) -> None:
    """Verify that mass matrix inverses are computed correctly."""
    rng = np.random.default_rng(seed=0)

    corners = np.array(
        ((-1, -1), (+1, -1), (+1, +1), (-1, +1)), np.float64
    ) + rng.uniform(-0.1, +0.1, (4, 2))

    space = ElementFemSpace2D(
        Basis2D(
            Basis1D(nh, IntegrationRule1D(nh + 1)), Basis1D(nv, IntegrationRule1D(nv + 1))
        ),
        corners,
    )

    for order in UnknownFormOrder:
        fwd = space.mass_from_order(order, inverse=False)
        bwd = space.mass_from_order(order, inverse=True)
        n1, n2 = fwd.shape
        assert pytest.approx(fwd @ bwd) == np.eye(n1, n2)
