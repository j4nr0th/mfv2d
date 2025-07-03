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



Matrix Assembly
---------------

Global system matrix is created by :func:`assemble_matrix`, which creates
individual merges leaf element matrices. This used to be more involved,
but now we are in better times.

.. autofunction:: assemble_matrix



Vector Assembly
---------------

Global system vector is similarly created by :func:`assemble_vector`. It used
to be slow, complicated, and painful, but it's all ogre now.

.. autofunction:: assemble_vector


Forcing Assembly
----------------

Last of the assembly routines is the :func:`assemble_forcing` function. This
assembles together the leaf element forcing (value of the given expression
given solution).

.. autofunction:: assemble_forcing


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
