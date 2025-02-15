"""C based evaluation of element matrices."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from interplib.kforms.eval import MatOpCode

def compute_element_matrices(
    expression: list[MatOpCode | int | float],
    pos_bl: npt.NDArray[np.float64],
    pos_br: npt.NDArray[np.float64],
    pos_tr: npt.NDArray[np.float64],
    pos_tl: npt.NDArray[np.float64],
    element_orders: npt.NDArray[np.uint64],
    cache_contents: Sequence[
        tuple[
            int,
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
    ],
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
    order: int,
    n_int: int,
    int_nodes: npt.NDArray[np.float64],
    node_precomp: npt.NDArray[np.float64],
    edge_00_precomp: npt.NDArray[np.float64],
    edge_01_precomp: npt.NDArray[np.float64],
    edge_11_precomp: npt.NDArray[np.float64],
    surface_precomp: npt.NDArray[np.float64],
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
