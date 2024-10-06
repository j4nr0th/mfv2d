import numpy as np
import numpy.typing as npt

def lagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...

def dlagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...

def d2lagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
