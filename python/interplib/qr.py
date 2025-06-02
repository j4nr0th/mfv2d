"""Code related to QR decomposition and solving."""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from interplib._mimetic import GivensSeries, SparseVector


class CompositeQMatix:
    """QMatrix which first applies the child's QMatrices.

    When applying it to a vector, child matrices will be applied first on
    the sections of the input, then parent transformation will be applied.
    """

    children: tuple[tuple[slice, GivensSeries | CompositeQMatix], ...]
    own: GivensSeries | None

    def __init__(
        self, own: GivensSeries | None = None, *children: GivensSeries | CompositeQMatix
    ) -> None:
        self.own = own
        offset = 0
        cv: list[tuple[slice, GivensSeries | CompositeQMatix]] = list()
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

            cv: list[SparseVector] = list()
            for s, c in self.children:
                o = other[s]
                if o.count == 0:
                    cv.append(o)
                else:
                    co = c @ o
                    # print(f"{co=}, {o=}, {[g for g in c]}")
                    cv.append(co)

            end = self.children[-1][0].stop
            if end != other.n:
                cv.append(other[end:])

            r = SparseVector.concatenate(*cv)
            if self.own is not None:
                r = self.own @ r
            return r
        a = np.array(other, np.float64)
        for s, c in self.children:
            a[s] = c @ a[s]
        # a = np.concatenate([c @ a[s] for s, c in self.children])
        if self.own is not None:
            a = self.own @ a
        return a

    def apply(self, v: npt.NDArray[np.float64], /) -> None:
        """Apply in-place as fast as possible.

        This function works as fast as possible, without any allocations.

        Parameters
        ----------
        v : array
            One dimensional array to rotate.
        """
        for s, c in self.children:
            c.apply(v[s])
        if self.own:
            c.apply(v)
