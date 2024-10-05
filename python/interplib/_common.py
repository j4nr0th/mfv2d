from __future__ import  annotations

from typing import overload

import numpy as np
import numpy.typing as npt


def ensure_array(a: npt.ArrayLike, dt: np.dtype) -> npt.NDArray:
    """Returns the array which has the specified dtype."""
    if isinstance(a, np.ndarray) and a.dtype == dt:
        return a
    return np.array(a, dtype=dt)
