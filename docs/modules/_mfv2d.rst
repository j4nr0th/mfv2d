.. currentmodule:: mfv2d._mfv2d

_mfv2d
======

Internal module which includes functions written in C. These are mostly
intended for speed, but some types are written in C for the exprese purpose
of making it easier to use in other C functions.

Polynomials
-----------

Since basis require Lagrange polynomials and their derivatives to be
computed often, these are implemented in C, where the most efficient algorithms
also preserve accuracy for large degrees of these polynomials.

Another commonly required operation is computing Gauss-Legendre-Lobatto nodes
and associated integration weights.

.. autofunction:: lagrange1d

.. autofunction:: dlagrange1d

.. autofunction:: compute_gll


There is also a function to evaluate Legendre polynomials. This is intended to
be used for computing projections on Legendre hierarchical basis.

.. autofunction:: compute_legendre

For conversion of the Legendre basis coefficients into Legendre integral basis
coefficients, the function :func:`legendre_l2_to_h1_coefficients` is provided.

.. autofunction:: legendre_l2_to_h1_coefficients


Basis
-----

In order to define a set of basis, which can be integrated, an integration
rule is defined by an :class:`IntegrationRule1D`. This is essentially wrapped
result of :func:`compute_gll`, but as an object.

.. autoclass:: IntegrationRule1D
    :members:


Based on an integration rule, one-dimensional basis can be defined with the
:class:`Basis1D` type. This

.. autoclass:: Basis1D
    :members:

Two dimensional basis are constructed from tensor products of one-dimensional
basis. As such the :class:`Basis2D` type is merely a container for two
one-dimensional basis.

.. autoclass:: Basis2D
    :members:


Topology
--------

Toplogy information is used when constructing continuity constraints on degrees
of freedom. As such, these types allow for very quick and precice lookup of
neighbours, computations of duals, and other required operations.

The base building block is the :class:`GeoID` type, which is a combination of
a geometrical object's index and an indicator of its orientation. Often time
functions will allow :class:`GeoID` arguments to be replaced by 1-based integer
index, with its sign indicating orientation.

The index of a :class:`GeoID` can also be considered invalid. For functions
that allow integer along with :class:`GeoID` this would correspond to passing
the value of 0. In that case a value would be equivalent to ``False`` in
any boolean context.

.. autoclass:: GeoID
    :members:

A :class:`Line` is a one-dimensional topological object that connects two
points with their respective :class:`GeoID` indices. For a dual line,
point indices indicate indices of primal surfaces valid index, with its
orientation indicating whether or not the line was in the surface in its
positive or negative orientation.

While for primal lines, both points have valid indices, dual lines may not.
This happens when a primal line is on the external boundary of the manifold.
In that case, the side which has no valid index is comming from bordering on
no surfaces on that side.

.. autoclass:: Line
    :members:


The last of the topological primitives is the :class:`Surface`. Surfaces
consist of an arbitraty number of lines. This is because dual surfaces
corresponding to nodes on the corners of a mesh will in almost all cases
have a number of lines different than four.

.. autoclass:: Surface
    :members:

Surfaces can together form a :class:`Manifold2D` object. This is a
collection of surfaces which supports most of needed topological
operations. It is used as the main workhorse of
:class:`mfv2d.mimetic2d.Mesh2D` functionality.

.. autoclass:: Manifold2D
    :members:


Evaluating Terms
----------------

One of key operations that need to be supported in order to solve a
system is either computation of element matrices, or computing what
is their product with the current solution vector. The instruction
translation and generation is handled by the :mod:`mfv2d.kforms` module,
so as far as :mod:`mfv2d._mfv2d` code is concerned, it receives parsed
bytecode it just needs to execute.

One of the most important functions in the entire module is
:func:`compute_element_matrix`. This can be used to generate individual
element matrices for the system, which can then be combined into the
global system matrix.

.. autofunction:: compute_element_matrix

If evaluation of the product of the solution vector with the matrix is
needed, instead of computing it with :func:`compute_element_matrix`, it can
instead be obtained from :func:`compute_element_vector`. This is especially
useful for non-linear terms, since they would require matrices to be evaulated
at every iteration.

.. autofunction:: compute_element_vector


Projections
-----------

There are two functions related to projection of degrees of freedom. These are
concerned with two different types of projection. First is the less intuitive
one and is the the projection between primal and dual degrees of freedom.
This is done by the :func:`compute_element_mass_matrix` function, which does
exactly as its name would suggest.

.. autofunction:: compute_element_mass_matrix


The second kind of projection is from a lower order :class:`Basis2D` to
higher order :class:`Basis2D` or vice versa. This is performed by the
matrices computed by :func:`compute_element_projector`. If its resulting
projection matirx is transposed, it will instead act as the projector for
dual degrees of freedom.

.. autofunction:: compute_element_projector
