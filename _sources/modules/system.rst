.. currentmodule:: mfv2d.system

system
======

This system contains the type :class:`KFormSystem`, used to hold the equations
described by :class:`mfv2d.kform.KEquation`.

.. autoclass:: KFormSystem


There is also the wrapper around the C type
:class:`mfv2d._mfv2d.ElementFormSpecification` provided by
:class:`ElementFormSpecification`. It provides a nice constructor,
as well as wrappers that allow to use types from :class:`mfv2d.kform`
instead of "raw" Python strings and integers. Some utility methods
are also provided.

.. autoclass:: ElementFormSpecification
