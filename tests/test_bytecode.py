"""Check C code correctly computes the mass matrices."""

from mfv2d._mfv2d import check_bytecode
from mfv2d.eval import translate_equation, translate_to_c_instructions
from mfv2d.kform import KFormUnknown


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
