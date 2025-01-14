"""Test Bernstein polynomial implementation."""

import numpy as np
import pytest
from interplib import Bernstein1D, Polynomial1D


def test_manual_construction():
    """Check that the results are consistent with manually written polynomial."""
    a, b, c, d = -1.0, 3.0, -2.0, +0.5
    poly = Polynomial1D((d, c, b, a))
    bern = Bernstein1D.from_power_series(poly)
    test_samples = np.linspace(0, 1, 512)
    assert pytest.approx(poly(test_samples)) == bern(test_samples)


def test_derivative():
    """Check that derivative is correct."""
    a, b, c, d = -1.0, 3.0, -2.0, +0.5
    poly = Polynomial1D((d, c, b, a))
    bern = Bernstein1D.from_power_series(poly)
    dp = poly.derivative
    db = bern.derivative
    db2 = Bernstein1D.from_power_series(dp)
    assert db2 == db


def test_antiderivative():
    """Check that anitderivative is correct."""
    a, b, c, d = -1.0, 3.0, -2.0, +0.5
    poly = Polynomial1D((d, c, b, a))
    bern = Bernstein1D.from_power_series(poly)
    dp = poly.antiderivative
    db = bern.antiderivative
    db2 = Bernstein1D.from_power_series(dp)
    assert db2 == db


def test_fit():
    """Check that nodal fit is correct."""
    np.random.seed(0)
    x, y = np.random.random_sample(10), np.random.random_sample(10)
    bern = Bernstein1D.fit_nodal(x, y)
    assert pytest.approx(bern(x)) == y
