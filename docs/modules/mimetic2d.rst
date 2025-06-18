.. currentmodule:: mfv2d.mimetic2d

mimetic2d
=========

Most of the module is in the process of getting refactored into others.
Previously, code was written with element objects, which were used to
hold element relations. These are now all handled by
:class:`mfv2d.element.ElementCollection`.

Incidence Functions
-------------------

While these are not used internally, when it comes to testing these functions
play a very important role. They are not fast nor are they efficient, their
main purpose is to be used in tests to validate that operations done
in C are correct. If they were to be rewritten to allow different polynomial
order in each dimension, it would allow the code to deal with mixed-order
elements.


Generating Matrices
~~~~~~~~~~~~~~~~~~~

These directly generate the incidence matrices.

.. autofunction:: incidence_10

.. autofunction:: incidence_21


Applying Matrices
~~~~~~~~~~~~~~~~~

Since C code does not explicitly compute incidnece matrices unless necessary,
these are a lot more commonly used. Here they are written very explicitly,
so that they can be translated into C almost line by line.

.. autofunction:: apply_e10

.. autofunction:: apply_e10_t

.. autofunction:: apply_e10_r

.. autofunction:: apply_e10_rt

.. autofunction:: apply_e21

.. autofunction:: apply_e21_t

.. autofunction:: apply_e21_r

.. autofunction:: apply_e21_rt


Mesh Geometry
-------------

Mesh geometry is handled by the :class:`Mesh2D` type. This uses
:class:`mfv2d._mfv2d.Manifold2D` to store its topology and its dual topology,
but also contains geometry. It is also equiped with helper methods for
obtaining elements and plotting with :mod:`pyvista`.

.. autoclass:: Mesh2D


Caching Basis
-------------

Since only a handful of basis and integration orders are ever used in the solve
and creating :class:`mfv2d._mfv2d.Basis1D` and
:class:`mfv2d._mfv2d.IntegrationRule1D` both is not extremely cheap, the
:class:`FemCache` is introduced to deal with caching them. It does not
deal with caching :class:`mfv2d._mfv2d.Basis2D` objects, since they are
just containers for two objects and so are not a significant cost to newly
construct each time.

.. autoclass:: FemCache
