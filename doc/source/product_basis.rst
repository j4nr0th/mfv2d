.. currentmodule:: interplib

.. _interplib.product_basis:


Product Basis
=============

In higher dimensions, the simplest way to create a set of basis functions is to use the
set(s) of :ref:`1D basis <interplib.basis1d>` and take their products with each other
as the new basis. The advantage of such a choice of basis is the fact that it allows for
very natural link with different geometrical forms.

.. autoclass:: BasisProduct2D
    :members:
