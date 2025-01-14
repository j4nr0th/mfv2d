.. currentmodule:: interplib

.. _interplib.basis1d:

.. toctree::
    :hidden:

    bernstein1d
    polynomial1d

1D Basis Functions
==================

Basis functions are a functions which allow for representation of other functions.
The :class:`Basis1D` acts as a base class which specifies the bare necesseties which
must be provided by such functions.

.. autoclass:: Basis1D
    :members:

    .. py:method:: __call__(x: array_like, /) -> array

        Evaluate the basis function at the specified positions.

The types of 1D basis functions currently provided are:

- :ref:`power series polynomials <interplib.polynomial1d>`
- :ref:`bernstein polynomials <interplib.bernstein1d>`
