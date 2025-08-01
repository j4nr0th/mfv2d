"""Test element form specifications."""

import pytest
from mfv2d._mfv2d import _ElementFormsSpecification
from mfv2d.kform import UnknownFormOrder


def test_basic_construction_and_properties():
    """Check construction and properties work."""
    efs = _ElementFormsSpecification(
        ("foo", UnknownFormOrder.FORM_ORDER_0),
        ("bar", UnknownFormOrder.FORM_ORDER_2),
        ("bar2", UnknownFormOrder.FORM_ORDER_1),
    )
    assert efs.names == ("foo", "bar", "bar2")
    assert efs.orders == (
        UnknownFormOrder.FORM_ORDER_0,
        UnknownFormOrder.FORM_ORDER_2,
        UnknownFormOrder.FORM_ORDER_1,
    )


def test_getitem():
    """Check __getitem__ works."""
    efs = _ElementFormsSpecification(
        ("foo", UnknownFormOrder.FORM_ORDER_1), ("bar", UnknownFormOrder.FORM_ORDER_1)
    )
    assert efs[0] == ("foo", UnknownFormOrder.FORM_ORDER_1)
    assert efs[1] == ("bar", UnknownFormOrder.FORM_ORDER_1)
    with pytest.raises(IndexError):
        _ = efs[2]


def test_string_length_limit():
    """Check that it handles name being too long."""
    long_name = "a" * 40
    with pytest.raises(ValueError):
        _ElementFormsSpecification((long_name, UnknownFormOrder.FORM_ORDER_0))


def test_form_order_errors():
    """Check that it handles incorrect values for form order."""
    values = [
        -124,
        UnknownFormOrder.FORM_ORDER_0.value - 1,
        UnknownFormOrder.FORM_ORDER_0,
        UnknownFormOrder.FORM_ORDER_1,
        UnknownFormOrder.FORM_ORDER_2,
        UnknownFormOrder.FORM_ORDER_2.value + 1,
        21921,
    ]
    for val in values:
        if val in UnknownFormOrder:
            _ElementFormsSpecification(("test", val))
        else:
            with pytest.raises(ValueError):
                _ElementFormsSpecification(("test", val))


def test_type_errors():
    """Check that type errors are correctly raised."""
    with pytest.raises(TypeError):
        _ElementFormsSpecification((123, 5))  # name not a string
    with pytest.raises(ValueError):
        _ElementFormsSpecification(("name", "not-an-int"))  # order not int
    with pytest.raises(TypeError):
        _ElementFormsSpecification("single string, not tuple")


def test_empty_construction():
    """Check an empty constructor does not work."""
    with pytest.raises(TypeError):
        _ElementFormsSpecification()


def test_contains():
    """Check __contains__ method works."""
    specs = [
        ("foo", UnknownFormOrder.FORM_ORDER_0),
        ("bar", UnknownFormOrder.FORM_ORDER_2),
        ("bar2", UnknownFormOrder.FORM_ORDER_1),
    ]
    efs = _ElementFormsSpecification(*specs)

    for spec in specs:
        assert spec in efs

    assert ("one", "two", "three") not in specs


_TEST_ORDERS = ((1, 2), (3, 4), (4, 3), (4, 4))


@pytest.mark.parametrize(("n1", "n2"), _TEST_ORDERS)
def test_sizes(n1: int, n2: int):
    """Check form sizes are correct."""
    specs = [
        ("foo", UnknownFormOrder.FORM_ORDER_0),
        ("bar", UnknownFormOrder.FORM_ORDER_2),
        ("foo2", UnknownFormOrder.FORM_ORDER_0),
        ("bar3", UnknownFormOrder.FORM_ORDER_2),
        ("bar24", UnknownFormOrder.FORM_ORDER_1),
        ("bar512", UnknownFormOrder.FORM_ORDER_2),
        ("foo325", UnknownFormOrder.FORM_ORDER_0),
        ("bar275", UnknownFormOrder.FORM_ORDER_1),
        ("bar265", UnknownFormOrder.FORM_ORDER_1),
        ("bar26", UnknownFormOrder.FORM_ORDER_1),
    ]
    efs = _ElementFormsSpecification(*specs)

    for idx, (_, order) in enumerate(efs):
        uo = UnknownFormOrder(order)
        assert uo.full_unknown_count(n1, n2) == efs.form_size(idx, n1, n2)


@pytest.mark.parametrize(("n1", "n2"), _TEST_ORDERS)
def test_offsets(n1: int, n2: int):
    """Check form offsets are correct."""
    specs = [
        ("foo121", UnknownFormOrder.FORM_ORDER_0),
        ("bar2", UnknownFormOrder.FORM_ORDER_2),
        ("fooooo", UnknownFormOrder.FORM_ORDER_0),
        ("baar", UnknownFormOrder.FORM_ORDER_2),
        ("baraa2", UnknownFormOrder.FORM_ORDER_1),
        ("baaaaaar", UnknownFormOrder.FORM_ORDER_2),
        ("foooooooo", UnknownFormOrder.FORM_ORDER_0),
        ("baaaar2", UnknownFormOrder.FORM_ORDER_1),
        ("baaaaar2", UnknownFormOrder.FORM_ORDER_1),
        ("basr2", UnknownFormOrder.FORM_ORDER_1),
    ]
    efs = _ElementFormsSpecification(*specs)

    offset = 0
    offsets = []
    for _, order in efs:
        offsets.append(offset)
        uo = UnknownFormOrder(order)
        cnt = uo.full_unknown_count(n1, n2)
        offset += cnt

    for ie, offset in enumerate(offsets):
        assert offset == efs.form_offset(ie, n1, n2)


@pytest.mark.parametrize(("n1", "n2"), _TEST_ORDERS)
def test_total_size(n1: int, n2: int):
    """Check form total sizes are correct."""
    specs = [
        ("foasdo", UnknownFormOrder.FORM_ORDER_0),
        ("bhfar", UnknownFormOrder.FORM_ORDER_2),
        ("foo", UnknownFormOrder.FORM_ORDER_0),
        ("bvfar", UnknownFormOrder.FORM_ORDER_2),
        ("bar2", UnknownFormOrder.FORM_ORDER_1),
        ("baAr", UnknownFormOrder.FORM_ORDER_2),
        ("foooooooo", UnknownFormOrder.FORM_ORDER_0),
        ("baaaaar2", UnknownFormOrder.FORM_ORDER_1),
        ("bsfdar2", UnknownFormOrder.FORM_ORDER_1),
        ("badfr2", UnknownFormOrder.FORM_ORDER_1),
    ]
    efs = _ElementFormsSpecification(*specs)

    total_size = 0
    for _, order in efs:
        uo = UnknownFormOrder(order)
        cnt = uo.full_unknown_count(n1, n2)
        total_size += cnt

    assert total_size == efs.total_size(n1, n2)
