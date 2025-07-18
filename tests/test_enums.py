"""Check that Python and C enums agree."""

from mfv2d._mfv2d import (
    ELEMENT_SIDE_BOTTOM,
    ELEMENT_SIDE_LEFT,
    ELEMENT_SIDE_RIGHT,
    ELEMENT_SIDE_TOP,
    MATOP_IDENTITY,
    MATOP_INCIDENCE,
    MATOP_INTERPROD,
    MATOP_INVALID,
    MATOP_MASS,
    MATOP_MATMUL,
    MATOP_PUSH,
    MATOP_SCALE,
    MATOP_SUM,
)
from mfv2d.eval import MatOpCode
from mfv2d.mimetic2d import ElementSide


def test_mat_ops():
    """Check that MatOpCode values match C values."""
    assert MatOpCode.IDENTITY == MATOP_IDENTITY
    assert MatOpCode.INCIDENCE == MATOP_INCIDENCE
    assert MatOpCode.INTERPROD == MATOP_INTERPROD
    assert MatOpCode.INVALID == MATOP_INVALID
    assert MatOpCode.MASS == MATOP_MASS
    assert MatOpCode.MATMUL == MATOP_MATMUL
    assert MatOpCode.PUSH == MATOP_PUSH
    assert MatOpCode.SCALE == MATOP_SCALE
    assert MatOpCode.SUM == MATOP_SUM


def test_element_side():
    """Check that element sides match C values."""
    assert ElementSide.SIDE_BOTTOM == ELEMENT_SIDE_BOTTOM
    assert ElementSide.SIDE_RIGHT == ELEMENT_SIDE_RIGHT
    assert ElementSide.SIDE_TOP == ELEMENT_SIDE_TOP
    assert ElementSide.SIDE_LEFT == ELEMENT_SIDE_LEFT
