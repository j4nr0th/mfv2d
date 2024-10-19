from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt

from interplib._common import ensure_array


@dataclass(init=False, frozen=True)
class RadialBasisInterpolation:
    nodes: npt.NDArray[np.float64]
    basis: Callable[[npt.NDArray[np.float64]], npt.NDArray]

    def __init__(self, nodes: npt.NDArray[np.float64]) -> None:
        object.__setattr__(self, "nodes", nodes)

    def __call__(
        self,
        pos: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError()


# RBF using 1/((k * ||r||^2) + 1)


def _sirbf(
    x: npt.NDArray,
    k: np.float64,
    n: int,
    nodes: npt.NDArray[np.float64],
) -> npt.NDArray:
    return 1 / (
        k * np.linalg.norm(x[None, ...] - nodes[:, None, ...], axis=-1) ** n + 1
    )


@dataclass(init=False, frozen=True)
class SIRBF(RadialBasisInterpolation):
    """Radial basis interpolation using inverse square function."""
    k: np.float64

    def __init__(
        self,
        k: np.floating,
        nodes: npt.ArrayLike,
        values: npt.ArrayLike,
        positions: npt.ArrayLike|None = None,
        **kwargs,
    ) -> None:
        object.__setattr__(self, "k", np.float64(k))
        nds = ensure_array(nodes, np.float64)
        if nds.ndim < 2:
            nds = np.reshape(nds, (-1, 1))
        object.__setattr__(
            self,
            "basis",
            partial(_sirbf, k=np.float64(k), nodes=nds, n=2),
        )
        vals = ensure_array(values, np.float64)
        if vals.ndim < 2:
            vals = np.reshape(vals, (-1, 1))

        if vals.shape[0] != nds.shape[0]:
            raise ValueError(
                "Both nodes and values should have the number of entries."
            )

        if "_inverse" not in kwargs:
            if positions is None:
                pos = nds
            else:
                pos = ensure_array(positions, np.float64)
                if pos.ndim < 2:
                    pos = np.reshape(pos, (-1, 1))
            if vals.shape[0] != pos.shape[0]:
                raise ValueError(
                    "Both positions and values should have the same shape, "
                    "being at most 2d."
                )

            rbvs = self.basis(pos)
            object.__setattr__(self, "_inverse", np.linalg.inv(rbvs))
        else:
            object.__setattr__(self, "_inverse", kwargs["_inverse"])

        object.__setattr__(
            self,
            "_coeffs",
            np.reshape(self._inverse @ vals, (-1,))
        )

        super().__init__(nds)

    def __call__(
        self,
        pos: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        xv = ensure_array(pos, np.float64)
        x = xv
        if x.ndim < 2:
            x = np.reshape(x, (-1, 1))
        rbvs = self.basis(x)

        return np.sum(rbvs * self._coeffs[..., None], axis=0)


    def reinterpolate(self, new_vals: npt.ArrayLike) -> SIRBF:
        return SIRBF(self.k, self.nodes, np.empty(0), _inverse=self._inverse)


