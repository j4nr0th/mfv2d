.. currentmodule:: interplib

.. _interplib.polynomial1d:

1D Polynomial Basis
===================

The most basic representation of a basis function is the :class:`Polynomial1D`. This
is a class which is defined by a set of coefficients
:math:`A = \left\{ a_0, \dots, a_n \right\}` as:

.. math::

    p(x) = \sum\limits_{i = 0}^n a_i x^i

The downside of these is that they become numerically unstable once their degree becomes
too large (above 10 or so), as at that point the terms at either side of the power series
start to dominate the result. As such, they are not the best choice when polynomials of
high degrees are used.

They do however have their advantages:

- Derivative and Antiderivative are quick to analytically evaluate,
- Multiplication, division, and remainder are simple,
- Conversion to and from different representations are easy.

.. autoclass:: Polynomial1D
    :members:
