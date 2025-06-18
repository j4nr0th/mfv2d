"""Check C code correctly computes the mass matrices."""

import numpy as np
import pytest
from mfv2d._mfv2d import check_bytecode, element_matrices
from mfv2d.eval import translate_equation, translate_to_c_instructions
from mfv2d.kform import KFormUnknown
from mfv2d.mimetic2d import BasisCache, ElementLeaf2D


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_straight(order: int):
    """Check that it works for a reference element with identity Jacobian."""
    cache = BasisCache(order, 3 * order)
    elm = ElementLeaf2D(None, order, (-1, -1), (+1, -1), (+1, +1), (-1, +1))

    mass_node, mass_edge, mass_surf, mass_node_i, mass_edge_i, mass_surf_i = (
        element_matrices(-1, +1, +1, -1, -1, -1, +1, +1, cache.c_serialization())
    )

    assert pytest.approx(mass_node) == elm.mass_matrix_node(cache)
    assert pytest.approx(mass_edge) == elm.mass_matrix_edge(cache)
    assert pytest.approx(mass_surf) == elm.mass_matrix_surface(cache)
    assert pytest.approx(mass_node_i) == np.linalg.inv(elm.mass_matrix_node(cache))
    assert pytest.approx(mass_edge_i) == np.linalg.inv(elm.mass_matrix_edge(cache))
    assert pytest.approx(mass_surf_i) == np.linalg.inv(elm.mass_matrix_surface(cache))


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_weird(order: int):
    """Check that it works for a weird element with invertable Jacobian."""
    cache = BasisCache(order, 3 * order)
    X0 = -0.1
    X1 = 0.5
    X2 = 0.4
    X3 = 0.0

    Y0 = -0.5
    Y1 = 0.2
    Y2 = 0.5
    Y3 = 0.3
    elm = ElementLeaf2D(None, order, (X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3))

    mass_node, mass_edge, mass_surf, mass_node_i, mass_edge_i, mass_surf_i = (
        element_matrices(X0, X1, X2, X3, Y0, Y1, Y2, Y3, cache.c_serialization())
    )

    assert pytest.approx(mass_node) == elm.mass_matrix_node(cache)
    assert pytest.approx(mass_edge) == elm.mass_matrix_edge(cache)
    assert pytest.approx(mass_surf) == elm.mass_matrix_surface(cache)
    assert pytest.approx(mass_node_i) == np.linalg.inv(elm.mass_matrix_node(cache))
    assert pytest.approx(mass_edge_i) == np.linalg.inv(elm.mass_matrix_edge(cache))
    assert pytest.approx(mass_surf_i) == np.linalg.inv(elm.mass_matrix_surface(cache))


def test_bytecode():
    """Check that bytecode conversion works."""
    a = KFormUnknown(2, "a", 0)
    u = a.weight
    b = KFormUnknown(2, "b", 1)

    operations = translate_equation(
        u.derivative * a.derivative - 2 * (u.derivative * ~b), (), False, True
    )
    for form in operations:
        ops = operations[form]
        bytecode_in = translate_to_c_instructions(*ops)
        bytecode_out = check_bytecode(bytecode_in)

        for b1, b2 in zip(bytecode_in, bytecode_out, strict=True):
            assert b1 == b2
    v = b.weight
    operations = translate_equation(
        -1 * (v.derivative * b.derivative) + 2.0 * ((~v).derivative * (~b).derivative),
        (),
        False,
        True,
    )
    for form in operations:
        ops = operations[form]
        bytecode_in = translate_to_c_instructions(*ops)
        bytecode_out = check_bytecode(bytecode_in)

        for b1, b2 in zip(bytecode_in, bytecode_out, strict=True):
            assert b1 == b2
