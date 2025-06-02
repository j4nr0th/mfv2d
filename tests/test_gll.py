"""Check that Gauss-Legendre-Lobatto nodes."""

import numpy as np
import pytest
from mfv2d._mfv2d import compute_gll


@pytest.mark.parametrize("n", (1, 3, 5, 6, 8, 10, 20))
def test_weight_sum(n: int):
    """Check that the weights sum to 2."""
    _, w = compute_gll(n)
    assert np.sum(w) == pytest.approx(2.0)


@pytest.mark.parametrize("n", (0, 1, 2, 3, 5, 6, 8, 10, 20))
def test_weight_integration(n: int):
    """Check that integrals hold."""
    x, w = compute_gll(n)
    for p in range(max(n + 1, 2 * n - 2)):
        # Exact up to and including 2 * n - 3
        num = np.sum(x**p * w)
        assert (1 / (p + 1) - ((-1) ** (p + 1)) / (p + 1)) == pytest.approx(num)


@pytest.mark.parametrize("n", (0, 1, 2, 3, 5, 6, 8, 10, 20))
def test_independence(n: int):
    """Check that calling it multiple times is consistent."""
    x, w = compute_gll(n)
    for p in range(10 * (n + 1)):
        # Exact up to and including 2 * n - 3
        x1, w1 = compute_gll(n)
        assert np.allclose(x, x1)
        assert np.allclose(w, w1)


#     from time import perf_counter
#     from scipy.special import roots_legendre
#     np.random.seed(2509)
#     n = 100000
#     counts = np.random.randint(3, 15, n)
#     print(f"Reference time per call for {n} calls with random inputs.")
#     t0 = perf_counter()
#     for c in counts:
#         del c
#     t1 = perf_counter()
#     t_ref = (t1 - t0) / n
#     print(f"Reference time: {t_ref:e}s")
#
#     t0 = perf_counter()
#     for c in counts:
#         _, _ = compute_gll(c)
#     t1 = perf_counter()
#     t_jan = (t1 - t0) / n
#     print(f"My time: {t_jan:e} s ({t_jan - t_ref: e})")
#
#     t0 = perf_counter()
#     for c in counts:
#         _, _ = roots_legendre(c)
#     t1 = perf_counter()
#     t_spy = (t1 - t0) / n
#     print(f"Scipy: {t_spy:e} s ({t_spy - t_ref: e})")
