.. currentmodule:: mfv2d.solve_system

solve_system
============

This submodule contains all the basic building blocks which are needed to
create the solver in :mod:`mfv2d.solve_system_2d`.

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
the non-constraint equations, supporting functions are provided here.
To determine what are the indices of degrees
of freedom to carry :func:`find_time_carry_indices` is used.

.. autofunction:: find_time_carry_indices


With time marching element degrees of freedom have to often be mapped
from primal to dual or the other way around. To support this,
:func:`compute_element_primal_from_dual` and :func:`compute_element_dual` can
be used.

.. autofunction:: compute_element_primal_from_dual

.. autofunction:: compute_element_dual


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

.. autoclass:: TimeSettings


Solver Statistics
~~~~~~~~~~~~~~~~~

To return feedback on how the solver ran and more information about the system
it just solved, :class:`SolutionStatistics` type is provided. As such, an
object of this type is returned when the solver runs.

.. autoclass:: SolutionStatistics
