"""Test files for decomposition, which will be re-coded in C later."""

import numpy as np
import numpy.typing as npt


class SingularLU:
    """Decomposition of a potentially singular matrix.

    The idea behind this process is that an LDU decomposition is performed,
    while if a zero is encountered on the pivot element, it is skipped. As
    such, it returns an "inverse", which will return a solution if it has
    no component in the null space of the matrix.

    The purpose of this function is to singular sub-blocks of matrices,
    which are then constrained by the use of Lagrange multipliers.

    Parameters
    ----------
    mat : array_like
        Square array-like object, which represents the system matrix.
    rtol : float
        Relative tolerance at which the pivot is considered to be zero.
    atol : float
        Absolute tolerance at which the pivot is considered to be zero.
    """

    _mem: npt.NDArray[np.float64]
    _n: int

    def __init__(
        self, mat: npt.ArrayLike, rtol: float = 1e-10, atol: float = 1e-14, /
    ) -> None:
        acrit = atol
        rcrit = rtol * np.max(np.abs(mat))
        if atol < 0 or rtol < 0:
            raise ValueError("Tolerances must be greater than or equal to zero.")

        mtx = np.array(mat, np.float64)
        if mtx.ndim != 2 or mtx.shape[0] != mtx.shape[1]:
            raise ValueError(
                "Input array was not a 2D square array, instead it had the shape "
                f"of {mtx.shape}"
            )
        del mat

        mem = np.zeros_like(mtx)
        n = mtx.shape[0]
        self._n = n

        for i in range(n):
            # Deal with a row of L
            for j in range(i):
                # Row i of L
                li = mem[i, : i - 1].flatten()
                # Column j of U
                cj = mem[: j + 1, j].flatten()
                pv = cj[-1]
                v = (
                    (mtx[i, j] - np.dot(li[:j], cj[:-1])) / pv
                    if np.abs(pv) > acrit and np.abs(pv) > rcrit
                    else 0.0
                )
                mem[i, j] = v

            # Deal with a column of U
            for j in range(i + 1):
                # Row j of L
                lj = mem[j, :j].flatten()
                # Column i of U
                ci = mem[:j, i].flatten()
                v = mtx[j, i] - np.dot(lj, ci)
                mem[j, i] = v

        self._mem = mem

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        """Create array copy."""
        if dtype != np.float64 or (copy is not None and not copy):
            raise TypeError("Can not create a copy or change type of the SingularLU.")
        return np.array(self._mem, copy=True)

    def __str__(self) -> str:
        """Create printable representation."""
        return f"SingualrLDU({self.n} x {self.n})"

    @property
    def n(self) -> int:
        """Dimension of the system."""
        return self._n

    def solve(
        self,
        x: npt.ArrayLike,
        out: npt.NDArray[np.float64] | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-14,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute a solution to the singular system.

        The solution computed will be exact if the decomposition was done on a full-rank
        matrix. If that was not the case, then a solution will be attempted, which may or
        may not give the correct result.

        Parameters
        ----------
        x : array_like
            Vector or matrix to which the inverse of the system should be applied.

        out : array, optional
            Output array which receives the result. Must have the same shape as ``x``,
            be of the correct type, and contiguous in memory.

        rtol : float
            Relative tolerance at which the pivot is considered to be zero.

        atol : float
            Absolute tolerance at which the pivot is considered to be zero.

        Returns
        -------
        array
            Either a newly allocated array, or if ``out`` parameter was specified, it is
            returned.
        """
        v = np.array(x)
        flatten = False

        if v.ndim == 1:
            flatten = True
            v = v.reshape((-1, 1))
        elif v.ndim != 2:
            raise ValueError("Input array must have 1 or 2 dimensions.")

        if v.shape[0] != self._n:
            raise ValueError(
                "Number of rows does not match the dimension of the system"
                f" (got {v.shape[0]} when expecting {self._n})"
            )

        acrit = atol
        rcrit = rtol * np.max(np.abs(v))
        if atol < 0 or rtol < 0:
            raise ValueError("Tolerances must be greater than or equal to zero.")

        if out is None:
            out = np.zeros_like(v, np.float64)
        elif out.shape != v.shape or out.dtype != np.float64:
            raise ValueError(
                "Output array must have same shape as the input array and the correct"
                f" data type (input was {v.shape} and {np.float64}, but got {out.shape}"
                f" and {out.dtype})."
            )
        assert out is not None
        out[...] = v[...]

        # Forward substitute to soleve for L
        for i in range(self._n):
            out[i, :] = v[i, :] - np.sum(self._mem[i, :i][:, None] * out[:i, :], axis=0)

        # Backward substitute to soleve for U
        for i in reversed(range(self._n)):
            s = np.sum(self._mem[i, i + 1 :][:, None] * out[i + 1 :, :], axis=0)
            pv = self._mem[i, i]
            if pv == 0:
                if np.abs(s) >= acrit or np.abs(s) > rcrit:
                    raise RuntimeWarning(
                        "Solving a system where the RHS has components in the nullspace"
                        " of the system."
                    )
                out[i, :] = out[i, :]
            else:
                out[i, :] = (out[i, :] - s) / pv

        if flatten:
            out = out.reshape((-1,))

        return out
