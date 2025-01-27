.. currentmodule:: interplib

.. _interplib.splines1d:


1D Splines
==========

Splines are functions which are defined piecewise on an interval. There are two
main types of splines provided in :mod:`interplib`:

- Integer splines, implemented in :class:`Spline1Di`,
- Regular splines, implemented in :class:`Spline1D`.

Note that these use same power series representation as :class:`Polynomial1D` and
as such have the same weaknesses and strengths.

To construct splines based on either nodal values or element averages, convenience
functions are provided. All of them use use of :class:`SplineBC` type to define the
boundary conditions.

.. autoclass:: SplineBC
    :members:

.. autoclass:: Spline1D
    :members:

    .. .. autofunction:: nodal_interpolating_spline

    .. .. autofunction:: element_interpolating_spline

.. autoclass:: Spline1Di
    :members:

    .. autofunction:: nodal_interpolating_splinei

    .. autofunction:: element_interpolating_splinei
