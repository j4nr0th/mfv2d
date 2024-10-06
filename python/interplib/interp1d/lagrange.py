from __future__ import annotations, absolute_import

import numpy as np
from numpy import typing as npt

from interplib._common import ensure_array

from interplib._interp import lagrange1d as _lagrange1d
from interplib._interp import dlagrange1d as _dlagrange1d
from interplib._interp import d2lagrange1d as _d2lagrange1d


def lagrange1d(x: npt.ArrayLike[np.floating], xp: npt.ArrayLike[np.floating], yp: npt.ArrayLike[np.floating]) -> npt.NDArray[np.float64]:
    """Interpolate value of function given its values at a number of nodes.

    Parameters
    ----------
    x : array_like
        An array of locations where the function should be evaluated at.
        Must not be outside the nodes ``xp``.
    xp : array_like
        An array of nodes where the function is evaluated.
    yp : array_like
        An array of values of functions at locations ``xp``.

    Returns
    -------
    array of float64
        Array with interpolated values at locations ``x``
    """
    real_x: npt.NDArray[np.float64] = ensure_array(x, np.float64)
    real_xp: npt.NDArray[np.float64] = ensure_array(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = ensure_array(yp, np.float64)

    if len(real_xp.shape) != 1 or len(real_yp.shape) != 1 or real_xp.shape != real_yp.shape:
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    sort_idx = np.argsort(real_xp)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _lagrange1d(flat_x, real_xp[sort_idx])

    return np.reshape(interp_mtx @ real_yp, shape=real_x.shape)


def dlagrange1d(x: npt.ArrayLike[np.floating], xp: npt.ArrayLike[np.floating], yp: npt.ArrayLike[np.floating]) -> npt.NDArray[np.float64]:
    """Interpolate derivative of function given its values at a number of nodes.

    Parameters
    ----------
    x : array_like
        An array of locations where the function should be evaluated at.
        Must not be outside the nodes ``xp``.
    xp : array_like
        An array of nodes where the function is evaluated.
    yp : array_like
        An array of values of functions at locations ``xp``.

    Returns
    -------
    array of float64
        Array with interpolated derivatives at locations ``x``
    """
    real_x: npt.NDArray[np.float64] = ensure_array(x, np.float64)
    real_xp: npt.NDArray[np.float64] = ensure_array(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = ensure_array(yp, np.float64)

    if len(real_xp.shape) != 1 or len(real_yp.shape) != 1 or real_xp.shape != real_yp.shape:
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    sort_idx = np.argsort(real_xp)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _dlagrange1d(flat_x, real_xp[sort_idx])

    return np.reshape(interp_mtx @ real_yp, shape=real_x.shape)


def d2lagrange1d(x: npt.ArrayLike[np.floating], xp: npt.ArrayLike[np.floating], yp: npt.ArrayLike[np.floating]) -> npt.NDArray[np.float64]:
    """Interpolate second derivative of function given its values at a number of nodes.

    Parameters
    ----------
    x : array_like
        An array of locations where the function should be evaluated at.
        Must not be outside the nodes ``xp``.
    xp : array_like
        An array of nodes where the function is evaluated.
    yp : array_like
        An array of values of functions at locations ``xp``.

    Returns
    -------
    array of float64
        Array with interpolated derivatives at locations ``x``
    """
    real_x: npt.NDArray[np.float64] = ensure_array(x, np.float64)
    real_xp: npt.NDArray[np.float64] = ensure_array(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = ensure_array(yp, np.float64)

    if len(real_xp.shape) != 1 or len(real_yp.shape) != 1 or real_xp.shape != real_yp.shape:
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    sort_idx = np.argsort(real_xp)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _d2lagrange1d(flat_x, real_xp[sort_idx])

    return np.reshape(interp_mtx @ real_yp, shape=real_x.shape)






