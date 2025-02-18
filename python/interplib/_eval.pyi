"""C based evaluation of element matrices."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from interplib.kforms.eval import MatOpCode

_SerializedBasisCache = tuple[
    int,
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]

def compute_element_matrices(
    form_orders: Sequence[int],
    expressions: Sequence[Sequence[Sequence[MatOpCode | int | float] | None]],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint32],
    serialized_caches: Sequence[_SerializedBasisCache],
) -> tuple[npt.NDArray[np.float64]]:
    """Compuate element matrices based on the given instructions."""
    ...

def element_matrices(
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
    serialized_cache: _SerializedBasisCache,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute the element matrices."""
    ...

def check_bytecode(expression: list[MatOpCode | int | float], /) -> list[int | float]:
    """Convert bytecode to C-values, then back to Python."""
    ...

def check_incidence(
    x: npt.ArrayLike, /, order: int, in_form: int, transpose: bool, right: bool
) -> npt.NDArray[np.float64]:
    """Apply the incidence matrix to the input matrix."""
    ...
