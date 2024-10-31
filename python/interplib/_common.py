"""Common internal Python functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ensure_array(a: npt.ArrayLike, dt: np.dtype | type) -> npt.NDArray:
    """Return the array which has the specified dtype."""
    if isinstance(a, np.ndarray) and a.dtype == dt:
        return a
    return np.array(a, dtype=dt)
