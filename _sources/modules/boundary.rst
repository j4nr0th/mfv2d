.. currentmodule:: mfv2d.boundary

boundary
========

This submodule contains types that describe strong boundary condtitions. For
now, only steady boundary conditions are used, though there are already
types for the unsteady ones.

The main class that would be in use at the moment would be
:class:`BoundaryCondition2DSteady`. It is derived from
:class:`BoundaryCondition2D`, which is also the parent for
:class:`BoundaryCondition2DUnsteady`, but these are not (yet) supported,
so they are not exported by the module.

.. autoclass:: BoundaryCondition2DSteady

.. autoclass:: BoundaryCondition2DUnsteady

.. autoclass:: BoundaryCondition2D
