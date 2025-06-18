.. currentmodule:: mfv2d.solve_system

solve_system
============

This submodule contains all the basic building blocks which are needed to
create the solver in :mod:`mfv2d.solve_system_2d`. Some of these should
be replaced or removed, as they are outdated, but it works for now.


Computing the Explicit Terms
----------------------------

To compute :class:`mfv2d.kform.KExplicit` terms of the
:class:`mfv2d.kform.KFormSystem` the function :func:`compute_element_rhs`
is used, which is to be passed to :func:`mfv2d.element.call_per_leaf_flex`
to compute forcing terms. Under the hood it just calls :func:`_extract_rhs_2d`
for each element and then joins them together. This function in turn calls
:func:`rhs_2d_element_projection`, since boundary projections are evaluated
separately.

.. autofunction:: compute_element_rhs

.. autofunction:: _extract_rhs_2d

.. autofunction:: rhs_2d_element_projection


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


Continuity
----------

Continuity between elements is enforced throught creating :class:`Constraint`
values. Since continuity may be between elements which have different orders
on the edge where they border one another, the coefficients in these
constraints may not be ones and zeros. As such, there is a function
:func:`continuity_matrices`, which returns a matrix with coefficients
that can be used for these relations. It uses Python's
:func:`functools.cache` annotation, so calling it repetedly should not be
slow.

.. autofunction:: continuity_matrices


The internal functions used to generate these relations are
:func:`_continuity_element_1_forms`, :func:`_continuity_element_0_forms_inner`,
and :func:`_continuity_element_0_forms_corner`, which are used by other
functions.

.. autofunction:: _continuity_element_1_forms

.. autofunction:: _continuity_element_0_forms_inner

.. autofunction:: _continuity_element_0_forms_corner


Another function which is used to generate continuity coefficients is
:func:`continuity_child_matrices`. This is used for continuity between
elements.

.. autofunction:: continuity_child_matrices


These are then in turn used by :func:`_continuity_parent_child_nodes`
and :func:`_continuity_parent_child_edges`, which generate equations for
parent-child continuity of 0- and 1-forms respectively. Together they
are wrapped in the :func:`_parent_child_equations` function.


.. autofunction:: _continuity_parent_child_nodes

.. autofunction:: _continuity_parent_child_edges

.. autofunction:: _parent_child_equations


Matrix Assembly
---------------

Global system matrix is created by :func:`assemble_matrix`, which creates
individual root element matrices. These are created by calling
:func:`_compute_element_matrix`, which recursevly collect child element
matrices and adds Lagrange multipliers resulting from continuity between
children and parent.

.. autofunction:: assemble_matrix

.. autofunction:: _compute_element_matrix


Vector Assembly
---------------

Global system vector is similarly created by :func:`assemble_vector`. It creates
individual root element vectors by by calling :func:`_compute_element_vector`,
which recursevly collect child element matrices and padds to account for
Lagrange multipliers resulting from continuity between children and parent.

.. autofunction:: assemble_vector

.. autofunction:: _compute_element_vector


Forcing Assembly
----------------

Last of the assembly routines is the :func:`assemble_forcing` function. This
computes root element forcing (value of the given expression given solution)
by recursivly calling :func:`_compute_element_forcing`, then adding values
resulting from Lagrange multiplier relations.

.. autofunction:: assemble_forcing

.. autofunction:: _compute_element_forcing


Global Continuity
-----------------

For generating the continuity :class:`Constraint` values for global
continuity, the function
:func:`mesh_continuity_constraints` is used. It uses quite a few
helper functions, such as :func:`_top_level_continuity_0` for generating
top level continuity of 0-forms and :func:`_top_level_continuity_1` for
1-form continuity.


.. autofunction:: mesh_continuity_constraints

.. autofunction:: _top_level_continuity_0

.. autofunction:: _top_level_continuity_1


Boundary Conditions
-------------------

For generating the boundary condtition :class:`Constraint` and
:class:`ElementConstraint` values, the function
:func:`mesh_boundary_conditions` is used, which returns the strong and
weak boundary condition results.

Internally it calls :func:`_element_weak_boundary_condition` and
:func:`_element_strong_boundary_condition` for each boundary edge
and element, depending on which (if any) is specified on that edge.


.. autofunction:: mesh_boundary_conditions

.. autofunction:: _element_weak_boundary_condition

.. autofunction:: _element_strong_boundary_condition


Global Reconstruction
---------------------

Based on :func:`mfv2d.element.reconstruct` a global reconstruction
of the solution for all leaf elements is implemented with the
:func:`reconstruct_mesh_from_solution` function. It returns a
:class:`pyvista.UnstructuredGrid`, which can be added to the list
of outputs.


.. autofunction:: reconstruct_mesh_from_solution


Time March Support
------------------

Since for time marching certain quantities must be extracted from
the non-constraint equations, supporting functions are provided here. Namely,
:func:`_extract_time_carry` is called on each element, which
:func:`extract_carry` wraps. To determine what are the indices of degrees
of freedom to carry :func:`find_time_carry_indices` is used.

.. autofunction:: extract_carry

.. autofunction:: _extract_time_carry

.. autofunction:: find_time_carry_indices


Leaf Calculations
-----------------

For computing leaf element matrices and forcing, :func:`compute_leaf_matrix`
and :func:`compute_leaf_vector` are available. These pretty much just
coordinate calls to :func:`mfv2d._mfv2d.compute_element_matrix` and
:func:`mfv2d._mfv2d.compute_element_vector`.

.. autofunction:: compute_leaf_matrix

.. autofunction:: compute_leaf_vector

Since these also need vector field information for any interior products,
functions to compute are also given:
:func:`compute_element_vector_fields_nonlin` and its wrapper
:func:`compute_element_vector_fields`.


.. autofunction:: compute_element_vector_fields_nonlin

.. autofunction:: compute_element_vector_fields


The Actual Solver
-----------------

The actual solver that runs to solve the (potentially non-linear) system
is implemented in the :func:`non_linear_solve_run` function.

.. autofunction:: non_linear_solve_run


Solver Settings
~~~~~~~~~~~~~~~

To configure the solver, several settings dataclasses are provided. These
each handle a different aspect of the solver and most have some default
values.

.. autoclass:: SystemSettings

.. autoclass:: SolverSettings

.. autoclass:: RefinementSettings

.. autoclass:: TimeSettings


Solver Statistics
~~~~~~~~~~~~~~~~~

To return feedback on how the solver ran and more information about the system
it just solved, :class:`SolutionStatistics` type is provided. As such, an
object of this type is returned when the solver runs.

.. autoclass:: SolutionStatistics
