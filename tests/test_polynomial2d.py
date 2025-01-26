"""Check that 2D polynomials work as expected."""

import numpy as np
import pytest
from interplib import BasisProduct2D, Polynomial1D
from interplib.interp2d import Polynomial2D


def test_manual():
    """Check that a simple manual version works."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    y_manual = 1 + x2 * 0 + x2**2 * (x1) + x2**3 * (1)  # x1 * x2**2 - 2 * x1 + x2**3 + 1
    poly2d = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    y_auto = poly2d(x1, x2)
    assert pytest.approx(y_auto) == y_manual


def test_manual_partials():
    """Check that a simple manual version works for partial derivatives."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    # y(x1, x2) = 1 + x2 * 0 + x2**2 * (x1) + x2**3 * (1)
    dydx1_manual = x2**2
    dydx2_manual = 2 * x2 * (x1) + 3 * x2**2 * (1)
    poly2d = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    dydx1_auto = poly2d.partial(0)(x1, x2)
    dydx2_auto = poly2d.partial(1)(x1, x2)
    assert (
        pytest.approx(dydx1_auto) == dydx1_manual
        and pytest.approx(dydx2_auto) == dydx2_manual
    )


def test_manual_antiderivatives():
    """Check that a simple manual version works for anti-derivatives."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    # y(x1, x2) = 1 + x2 * 0 + x2**2 * (x1) + x2**3 * (1)
    # Y_x1(x1, x2) = x1 + x2**2 * (x1**2/2) + x2**3 * (x1)
    # Y_x2(x1, x2) = x2 + x2**3 * (x1/3) + x2**4 * (1/4)
    yx1_manual = x1 + x2**2 * (x1**2 / 2) + x2**3 * (x1)
    yx2_manual = x2 + x2**3 * (x1 / 3) + x2**4 * (1 / 4)
    poly2d = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    yx1_auto = poly2d.antiderivative(0)(x1, x2)
    yx2_auto = poly2d.antiderivative(1)(x1, x2)
    assert pytest.approx(yx1_auto) == yx1_manual and pytest.approx(yx2_auto) == yx2_manual


def test_manual_1d():
    """Check that passing 1D only works."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)

    poly2d = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    y_12 = poly2d(x1[0], None)(x2)
    y_21 = poly2d(None, x2[0])(x1)
    y_full_12 = poly2d(np.full_like(x2, x1[0]), x2)
    y_full_21 = poly2d(x1, np.full_like(x1, x2[0]))
    assert pytest.approx(y_full_12) == y_12 and pytest.approx(y_full_21) == y_21


def test_add():
    """Check that addition of two 2D polynomials works as expected."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    poly2d1 = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    poly2d2 = Polynomial2D(
        Polynomial1D((1, -2)),
        Polynomial1D((3, 1)),
        Polynomial1D((0, 4)),
        Polynomial1D((-2,)),
        Polynomial1D((-2,)),
        Polynomial1D((-3,)),
    )
    poly_sum = poly2d1 + poly2d2

    assert pytest.approx(poly_sum(x1, x2)) == poly2d2(x1, x2) + poly2d1(x1, x2)


def test_add_constant():
    """Check that addition of a constant to 2D polynomials works as expected."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    k = np.random.random_sample(1)[0]
    poly2d1 = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    poly_sum = k + poly2d1

    assert pytest.approx(poly_sum(x1, x2)) == k + poly2d1(x1, x2)


def test_mul():
    """Check that multiplication of two 2D polynomials works as expected."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    poly2d1 = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    poly2d2 = Polynomial2D(
        Polynomial1D((1, -2)),
        Polynomial1D((3, 1)),
        Polynomial1D((0, 4)),
        Polynomial1D((-2,)),
        Polynomial1D((-2,)),
        Polynomial1D((-3,)),
    )
    poly_sum = poly2d1 * poly2d2

    assert pytest.approx(poly_sum(x1, x2)) == poly2d2(x1, x2) * poly2d1(x1, x2)


def test_mul_constant():
    """Check that multiplication by constant works as expected."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    k = np.random.random_sample(1)[0]
    poly2d1 = Polynomial2D(
        Polynomial1D((1,)),
        Polynomial1D(()),
        Polynomial1D((0, 1)),
        Polynomial1D((1,)),
    )
    poly_sum = k * poly2d1

    assert pytest.approx(poly_sum(x1, x2)) == k * poly2d1(x1, x2)


def test_conversion():
    """Check that product is converted correctly."""
    np.random.seed(0)
    x1 = np.random.random_sample(512)
    x2 = np.random.random_sample(512)
    nodes = np.random.random_sample(4)
    product = BasisProduct2D.outer_product_basis(Polynomial1D.lagrange_nodal_basis(nodes))
    for lb in product:
        for b in lb:
            cov = b.as_polynomial()
            y_cov = cov(x1, x2)
            y_org = b(x1, x2)
            assert pytest.approx(y_cov) == y_org
