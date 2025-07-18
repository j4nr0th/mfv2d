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


Element Sides
-------------

Many functions have to perform operations on boundaries of the elements. To
make this easier to type check, the type :class:`ElementSide` is introduced.
This is an :class:`enum.IntEnum` subtype with only four values and is used
everywhere when a side of an element is an input of any function.

.. autoclass:: ElementSide
    :no-inherited-members:
    :member-order: bysource


To help identify what side a :class:`Line` with a specific index
of a :class:`Surface` is from, the function
:func:`find_surface_boundary_id_line` is given.

.. autofunction:: find_surface_boundary_id_line



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


Mesh Geometry
-------------

Since :class:`mfv2d._mfv2d.Mesh` is meant as a container, constructing
it is repetative and cumbersome. As such, the function :func:`mesh_create`
is intended to be used for to create a new mesh from position and
connectivity data.

.. autofunction:: mesh_create


Lagrange Multipliers and Constraints
------------------------------------

Since continuity and strong boundary conditions are enforced throught
Lagrange multipliers, there are types to represent these relations
efficiently. The base building block is the
:class:`ElementConstraint` type, which describes the degrees of
freedom and coefficients involved in a specific constraint. These
may then be combined into :class:`Constraint` type which associates
one or more :class:`ElementConstraint` with their right hand side value.

Note that :class:`ElementConstraint` are also used to convey weak boundary
condition information, however, in that case the coefficients in the
:class:`ElementConstraint` object represent contributions to the right
hand side.

.. autoclass:: ElementConstraint

.. autoclass:: Constraint


Degree of Freedom Counts
------------------------

There are some utility functions provided for computing number of
elemet degrees of freedom or lagrange multipliers. This is done
by :func:`compute_leaf_dof_counts`.

.. autofunction:: compute_leaf_dof_counts


Element Geometry
----------------

Some functions are also provided for computing geometry of elements.
This includes computing the Jacobian matrix (with :func:`jacobian`),
as well as computing the physical coordinates :math:`(x, y)`
as functions of the reference domain coordinates :math:`(\xi, \eta)`,
since it is assumed these are bilinear (hence the :func:`bilinear_interpolate`
function).

.. autofunction:: jacobian

.. autofunction:: bilinear_interpolate


Element Projection
------------------

This submodule also contains functions for projection of arbitrary
functions on the element as either primal or dual degrees of freedom.
Note that the dual projection is faster, since the primal has to
be followed by a multiplication of an inverse of the mass matrix
for the specific :math:`k`-form.

.. autofunction:: element_primal_dofs

.. autofunction:: element_dual_dofs


Reconstruction
--------------

To be able to return result for a solver as VTK file or even
for being able to compute interior product, there has to be
functionality to compute pointwise values for a :math:`k`-form
given its element degrees of freedom. This is handled by the
:func:`reconstruct` function. To help with it, :func:`vtk_lagrange_ordering`
is provided, since VTK does not like sensible order of unknowns.

.. autofunction:: reconstruct

.. autofunction:: vtk_lagrange_ordering
