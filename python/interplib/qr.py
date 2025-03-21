"""Code related to QR decomposition and solving."""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from interplib._mimetic import GivensRotation, SparseVector


class QMatrix:
    """Orthogonal matrix based on Givens rotations."""

    n: int
    rotations: tuple[GivensRotation, ...]

    def __init__(self, *rotations: GivensRotation) -> None:
        if not rotations:
            raise ValueError("No rotations were given.")
        self.rotations = rotations
        self.n = rotations[0].n
        for ir, r in enumerate(rotations):
            if r.n != self.n:
                raise ValueError(f"Rotation {ir} did not have the same dimension.")

    @overload
    def __matmul__(self, other: SparseVector) -> SparseVector: ...
    @overload
    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

    def __matmul__(
        self, other: SparseVector | npt.ArrayLike
    ) -> SparseVector | npt.NDArray[np.float64]:
        """Apply rotation to matrix or vector."""
        if isinstance(other, SparseVector):
            for r in self.rotations:
                other = r @ other

            return other

        a = np.asarray(other, np.float64)
        for r in self.rotations:
            a = r @ a

        return a


class CompositeQMatix:
    """QMatrix which first applies the child's QMatrices.

    When applying it to a vector, child matrices will be applied first on
    the sections of the input, then parent transformation will be applied.
    """

    children: tuple[tuple[slice, QMatrix | CompositeQMatix], ...]
    own: QMatrix | None

    def __init__(
        self, own: QMatrix | None = None, *children: QMatrix | CompositeQMatix
    ) -> None:
        self.own = own
        offset = 0
        cv: list[tuple[slice, QMatrix | CompositeQMatix]] = list()
        for c in children:
            end = offset + c.n
            cv.append((slice(offset, end), c))
            offset = end
        self.children = tuple(cv)

    @property
    def n(self) -> int:
        """Dimension of the system."""
        if self.own is not None:
            return self.own.n
        return sum(c.n for _, c in self.children)

    @overload
    def __matmul__(self, other: SparseVector) -> SparseVector: ...
    @overload
    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

    def __matmul__(
        self, other: SparseVector | npt.ArrayLike
    ) -> SparseVector | npt.NDArray[np.float64]:
        """Apply rotation to matrix or vector."""
        if isinstance(other, SparseVector):
            if other.n != self.n:
                raise ValueError("Dimension mismatch between input and result")
            cv: list[SparseVector] = [c @ other[s] for s, c in self.children]
            end = self.children[-1][0].stop
            if end != other.n:
                cv.append(other[end:])

            r = SparseVector.concatenate(*cv)
            if self.own is not None:
                r = self.own @ r
            return r
        a = np.asarray(other, np.float64)
        for s, c in self.children:
            a[s] = c @ a[s]
        # a = np.concatenate([c @ a[s] for s, c in self.children])
        if self.own is not None:
            a = self.own @ a
        return a
