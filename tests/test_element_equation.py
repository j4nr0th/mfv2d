"""Basic unit tests to check element equation type works as expected."""

import numpy as np
from mfv2d._mfv2d import ElementDofEquation


def test_element_dof_equation() -> None:
    """Check getters and construction work."""
    elem_index = 42
    pairs = [(0, 1.5), (2, -3.0), (5, 7.7)]
    obj = ElementDofEquation(elem_index, *pairs)

    assert obj.element == elem_index

    dofs = obj.dofs
    assert isinstance(dofs, np.ndarray)
    assert np.all(dofs == [p[0] for p in pairs])
    assert dofs.dtype == np.uint32

    coeffs = obj.coeffs
    assert isinstance(coeffs, np.ndarray)
    assert np.all(coeffs == [p[1] for p in pairs])
    assert coeffs.dtype == np.float64

    pairs_list = list(obj.pairs())
    assert pairs_list == pairs

    assert len(obj) == len(pairs)

    # Check that dofs and coeffs can be used in numpy functions
    dofs = obj.dofs
    coeffs = obj.coeffs
    assert np.sum(dofs) == sum(p[0] for p in pairs)
    assert np.isclose(np.dot(dofs, coeffs), sum(p[0] * p[1] for p in pairs))

    caught = None

    try:
        ElementDofEquation()  # type: ignore # Intentionally wrong
    except Exception as e:
        caught = e

    assert type(caught) is TypeError

    try:
        ElementDofEquation(1)  # Missing args
    except Exception as e:
        caught = e

    assert type(caught) is TypeError
