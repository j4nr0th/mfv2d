.. currentmodule:: mfv2d.element

element
=======

This submodule defines operations and utilities for dealing with
elements. This means both organizing them, as well as using that
data for iterating over them for computations.


At the core of organization is the :class:`ArrayCom` type, which
holds information necessary to coordinate operations over all
elements. For now that is just the number of all elements, though
in the future, this is probably where OpenMPI data would be stored.

.. autoclass:: ArrayCom

Storing Data
------------

From :class:`ArrayCom` object element arrays can be created. These
contain data for each of the elements. The base for these objects is
the :class:`ElementArrayBase` type.

.. autoclass:: ElementArrayBase

The first subtype of :class:`ElementArrayBase` is the
:class:`FixedElementArray`. This stores data as arrays of equal
shape for each element.

.. autoclass:: FixedElementArray

The second subtype is :class:`FlexibleElementArray`. This type
stores data as arrays with the same number of dimensions, but
different shapes for each element. As such these arrays also
contain a :class:`FixedElementArray`, which acts as storage
for the shapes of individual element arrays.

.. autoclass:: FlexibleElementArray

The last subtype is :class:`ObjectElementArray`, which is
designed to hold one object of a specific type for each
element. If the element entry is not set before accessing it,
an exception is raised.

.. autoclass:: ObjectElementArray


Organizing Elements
-------------------

Mesh elements are organized into an :class:`ElementCollection`.
This collection holds as :class:`ArrayCom` object for all arrays
related to this collection. Other important data it holds is
related to parent-child relations, corners of individual elements
and the orders of polynomials for these.

It also has some convenience methods to obtain indices of degrees
of freedom on the boundary of an element, or an order of a specific
element.

.. autoclass:: ElementCollection


Executing per Element
---------------------

One of key requirements for the solver is to be able to execute
functions for each element. This is done by calling either the
:func:`call_per_element_fix`, :func:`call_per_element_flex`,
or :func:`call_per_leaf_obj`
functions, which work for functions whose results are either
fixed size, flexible size, or objects respectively.

.. autofunction:: call_per_element_fix

.. autofunction:: call_per_element_flex

.. autofunction:: call_per_leaf_obj

There are also two specialized variant of the :func:`call_per_element_flex`
function, named :func:`call_per_root_flex` and :func:`call_per_leaf_flex`.
As can probably inferred from their names, these are called only for
root or leaf elements.

.. autofunction:: call_per_root_flex

.. autofunction:: call_per_leaf_flex


Element Utilities
-----------------

There are some utility functions provided for computing number of
elemet degrees of freedom or lagrange multipliers. These include
their single-element variats :func:`_compute_element_dofs` as well as the
exported :func:`compute_dof_sizes`.

.. autofunction:: _compute_element_dofs

.. autofunction:: compute_dof_sizes


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

Element Geometry
----------------

Some functions are also provided for computing geometry of elements.
This includes computing the Jacobian matrix (with :func:`jacobian`),
as well as computing the physical coordinates :math:`(x, y)`
as functions of the reference domain coordinates :math:`(\xi, \eta)`
(:func:`poly_x` and :func:`poly_y` respectively).


.. autofunction:: jacobian

.. autofunction:: poly_x

.. autofunction:: poly_y


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
:func:`reconstruct` function.

.. autofunction:: reconstruct
