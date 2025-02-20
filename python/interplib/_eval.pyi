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
    thread_stack_size: int = (1 << 24),
) -> tuple[npt.NDArray[np.float64]]:
    """Compuate element matrices based on the given instructions.

    Parameters
    ----------
    form_orders : Sequence of int
        Orders of the unknown differential forms.

    expressions : 2D matrix of (Sequence of (MatOpCode, int, and float) or None)
        Two dimensional matrix of instructions to compute the entry of the element matrix.
        It can be left as None, which means there is no contribution.

    pos_bl : (N, 2) array
        Array of position vectors for the bottom left corners of elements.

    pos_br : (N, 2) array
        Array of position vectors for the bottom right corners of elements.

    pos_tr : (N, 2) array
        Array of position vectors for the top right corners of elements.

    pos_tl : (N, 2) array
        Array of position vectors for the top left corners of elements.

    element_orders : (N,) array
        Array of orders of the elements. There must be an entry for this in
        the ``serialized_caches``.

    serialized_caches : Sequence of _SerializedBasisCache
        All the serialized caches to use for the elements. Only one is allowed
        per element order.

    thread_stack_size : int, default: 2 ** 24
        Default amount of memory allocated to each worker thread for the element they're
        working on.

    Returns
    -------
    tuple of N arrays
        Tuple of element matices.
    """
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
    """Convert bytecode to C-values, then back to Python.

    This function is meant for testing.
    """
    ...

def check_incidence(
    x: npt.ArrayLike, /, order: int, in_form: int, transpose: bool, right: bool
) -> npt.NDArray[np.float64]:
    """Apply the incidence matrix to the input matrix.

    This function is meant for testing.
    """
    ...
