import numpy as np
import numpy.typing as npt

def test() -> str:...

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

def hermite(
    x: npt.NDArray[np.float64],
    bc1: tuple[float, float, float],
    bc2: tuple[float, float, float],
) -> npt.NDArray[np.float64]:...
