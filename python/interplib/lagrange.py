"""Functions related to Lagrange nodal interpolation.

Lagrange interpolation is a quick and dirty technique for global
interpolation. It works just fine, but beware of common issues
it can have, such as Runge phenomenon.
"""

from __future__ import absolute_import, annotations

import numpy as np
from numpy import typing as npt

from interplib._interp import d2lagrange1d as _d2lagrange1d
from interplib._interp import dlagrange1d as _dlagrange1d
from interplib._interp import lagrange1d as _lagrange1d


def lagrange_function_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike
) -> npt.NDArray[np.double]:
    """Compute interpolation matrix for function based on samples.

    Parameters
    ----------
    x : (N,) array_like
        Points where the function should be evaluated at.
    xp : (M,) array_like
        Points where the function samples will be given.

    Returns
    -------
    (N, M) ndarray
        Matrix which, if multiplied by ``f(xp)`` will give approximations to
        ``f(x)``, where ``f`` is a scalar function.
    """
    real_x: npt.NDArray[np.float64] = np.atleast_1d(np.asarray(x, np.float64))
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _lagrange1d(real_xp, flat_x)

    return interp_mtx


def lagrange_derivative_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute interpolation matrix for derivatives based on samples.

    Parameters
    ----------
    x : (N,) array_like
        Points where the function should be evaluated at.
    xp : (M,) array_like
        Points where the function samples will be given.

    Returns
    -------
    (N, M) ndarray
        Matrix which, if multiplied by ``f(xp)`` will give approximations to
        ``df/dx``, where ``f`` is a scalar function.
    """
    real_x: npt.NDArray[np.float64] = np.atleast_1d(np.asarray(x, np.float64))
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _dlagrange1d(real_xp, flat_x)

    return np.astype(interp_mtx, np.float64)


def lagrange_2derivative_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute interpolation matrix for second derivatives  based on samples.

    Parameters
    ----------
    x : (N,) array_like
        Points where the function should be evaluated at.
    xp : (M,) array_like
        Points where the function samples will be given.

    Returns
    -------
    (N, M) ndarray
        Matrix which, if multiplied by ``f(xp)`` will give approximations to
        ``d^2f/dx^2``, where ``f`` is a scalar function.
    """
    real_x: npt.NDArray[np.float64] = np.atleast_1d(np.asarray(x, np.float64))
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)

    flat_x = np.ravel(real_x, order="C")

    interp_mtx = _d2lagrange1d(flat_x, real_xp)

    return interp_mtx


def interp1d_function_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike, yp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Interpolate value of function given its values at nodes.

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
    real_x: npt.NDArray[np.float64] = np.asarray(x, np.float64)
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = np.asarray(yp, np.float64)

    if (
        len(real_xp.shape) != 1
        or len(real_yp.shape) != 1
        or real_xp.shape != real_yp.shape
    ):
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    interp_mtx = lagrange_function_samples(real_x, real_xp)

    return np.astype(np.reshape(interp_mtx @ real_yp, shape=real_x.shape), np.float64)


def interp1d_derivative_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike, yp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Interpolate derivative of function given its values at nodes.

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
    real_x: npt.NDArray[np.float64] = np.asarray(x, np.float64)
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = np.asarray(yp, np.float64)

    if (
        len(real_xp.shape) != 1
        or len(real_yp.shape) != 1
        or real_xp.shape != real_yp.shape
    ):
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    interp_mtx = lagrange_derivative_samples(real_x, real_xp)

    return np.astype(np.reshape(interp_mtx @ real_yp, shape=real_x.shape), np.float64)


def interp1d_2derivative_samples(
    x: npt.ArrayLike, xp: npt.ArrayLike, yp: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Interpolate second derivative of function given its values at nodes.

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
    real_x: npt.NDArray[np.float64] = np.asarray(x, np.float64)
    real_xp: npt.NDArray[np.float64] = np.asarray(xp, np.float64)
    real_yp: npt.NDArray[np.float64] = np.asarray(yp, np.float64)

    if (
        len(real_xp.shape) != 1
        or len(real_yp.shape) != 1
        or real_xp.shape != real_yp.shape
    ):
        raise ValueError("Both xp and yp must be flat arrays of equal length")

    interp_mtx = lagrange_2derivative_samples(real_x, real_xp)

    return np.astype(np.reshape(interp_mtx @ real_yp, shape=real_x.shape), np.float64)
