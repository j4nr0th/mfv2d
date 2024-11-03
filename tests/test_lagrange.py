"""Tests for Lagrange interpolation."""

import numpy as np
import pytest
from interplib._interp import Polynomial1D
from interplib.lagrange import (
    interp1d_2derivative_samples,
    interp1d_derivative_samples,
    interp1d_function_samples,
)


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_funciton_interpolation(order: int, test_samples: int) -> None:
    """Check a polynomial of some order is interpolated with enough samples."""
    np.random.seed(1512)
    p = Polynomial1D(np.random.random_sample(order + 1))
    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = p(nodes)
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_function_samples(test_nodes, nodes, values)
    print(p(test_nodes))
    assert pytest.approx(test_results) == p(test_nodes)


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_derivative_interpolation(order: int, test_samples: int) -> None:
    """Check a derivative of some order is interpolated with enough samples."""
    np.random.seed(1512)
    p = Polynomial1D(np.random.random_sample(order + 1))
    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = p(nodes)
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_derivative_samples(test_nodes, nodes, values)
    assert pytest.approx(test_results) == p.derivative(test_nodes)


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_2derivative_interpolation(order: int, test_samples: int) -> None:
    """Check a derivative of some order is interpolated with enough samples."""
    np.random.seed(1512)
    p = Polynomial1D(np.random.random_sample(order + 1))
    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = p(nodes)
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_2derivative_samples(test_nodes, nodes, values)
    assert pytest.approx(test_results) == p.derivative.derivative(test_nodes)
