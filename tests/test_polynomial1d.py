"""Test polynomail basis class."""

import numpy as np
import pytest
from interplib._interp import Polynomial1D


def test_init_from_array():
    """Test that the object can be created from np.ndarray."""
    # create from numpy array
    poly1 = Polynomial1D(np.array([2.0, 1.0, 2]))
    # create from array of ints
    poly2 = Polynomial1D(np.array([2, 1, 2], np.uint8))

    assert np.all(poly2.coefficients == poly1.coefficients)


def test_init_from_list():
    """Test creating it from a list is same as from array."""
    test_list = [0.12, 31.2, -1 / 3.0, 2.0]
    poly1 = Polynomial1D(test_list)
    poly2 = Polynomial1D(np.array(test_list, np.float64))
    assert np.all(poly1.coefficients == poly2.coefficients)


def test_addition_float():
    """Test adding float to a polynomial."""
    test_array1 = np.array([0.2, -23.0, 2.41, 63])
    poly1 = Polynomial1D(test_array1)
    poly1 += 1.0
    test_array1[0] += 1.0
    assert np.all(poly1.coefficients == test_array1)


def test_addition_equal():
    """Test adding two polynomials of equal length."""
    test_array1 = np.array([0.2, -23.0, 2.41, 63])
    test_array2 = np.array([0.3, 0.0, 1, 3.6])
    poly1 = Polynomial1D(test_array1)
    poly2 = Polynomial1D(test_array2)
    combined = poly1 + poly2
    assert combined.coefficients == pytest.approx(test_array1 + test_array2)


def test_addition_unequal1():
    """Test adding two polynomials of different lengths."""
    test_array1 = np.array(
        [
            0.2,
            -23.0,
            2.41,
            63,
            2.10,
            11.201,
            2,
        ]
    )
    test_array2 = np.array([0.3, 0.0, 1, 3.6])
    poly1 = Polynomial1D(test_array1)
    poly2 = Polynomial1D(test_array2)
    combined = poly1 + poly2
    test_array1[: len(test_array2)] += test_array2
    assert combined.coefficients == pytest.approx(test_array1)


def test_addition_unequal2():
    """Test adding two polynomials of different lengths."""
    test_array1 = np.array([0.3, 0.0, 1, 3.6])
    test_array2 = np.array(
        [
            0.2,
            -23.0,
            2.41,
            63,
            2.10,
            11.201,
            2,
        ]
    )
    poly1 = Polynomial1D(test_array1)
    poly2 = Polynomial1D(test_array2)
    combined = poly1 + poly2
    test_array2[: len(test_array1)] += test_array1
    assert combined.coefficients == pytest.approx(test_array2)


def test_negation():
    """Test negating a polynomial."""
    test_array1 = np.array([0.3, 0.0, 1, 3.6])
    poly1 = Polynomial1D(test_array1)
    poly2 = -poly1
    assert np.all(poly2.coefficients == -test_array1)


def test_derivative():
    """Check if the derivative is correct."""
    test_array = np.array([1.2, 241.0, 2.50, -2.0, 554.22])
    poly = Polynomial1D(test_array)
    derivative = poly.derivative
    assert np.all(
        derivative.coefficients == (test_array * np.arange(test_array.shape[0]))[1:]
    )


def test_antiderivative():
    """Check if the antiderivative is correct."""
    test_array = np.array([1.2, 241.0, 2.50, -2.0, 554.22])
    poly = Polynomial1D(test_array)
    antiderivative = poly.antiderivative
    assert np.all(
        antiderivative.coefficients
        == np.pad(test_array / (np.arange(test_array.shape[0]) + 1), (1, 0))
    )


def test_multiplication():
    """Check if multiplication of two polynomials gives same results as product of two."""
    poly1 = Polynomial1D([0, 1, 41, 20, 100, 40])
    poly2 = Polynomial1D([-1, 2, -3.2, 10.0, 11])
    prod1 = poly1 * poly2
    prod2 = poly2 * poly1
    test_space = np.linspace(-1, +1, 11)
    assert prod1(test_space) == pytest.approx(poly1(test_space) * poly2(test_space))
    assert prod2(test_space) == pytest.approx(poly1(test_space) * poly2(test_space))


def test_setting_coefficients():
    """Manually set coefficients one by one."""
    poly1 = Polynomial1D([0, 1, 41, 20, 100])
    poly2 = Polynomial1D([-1, 2, -3.2, 10.0, 11])
    for i in range(len(poly2)):
        poly1[i] = poly2[i]
    assert np.all(poly1.coefficients == poly2.coefficients)


@pytest.mark.parametrize("order,ntest", ((1, 10), (3, 10), (10, 300)))
def test_polynomial_offset(order: int, ntest: int) -> None:
    """Check an offset polynomial acts as its original with offset."""
    np.random.seed(10928)
    poly = Polynomial1D(np.random.random_sample(order))
    offset = 1.0  # np.random.sample(1)[0]
    op = poly.offset_by(offset)
    xtest = np.linspace(-1, 1, ntest)
    assert pytest.approx(op(xtest)) == poly(xtest + offset)
