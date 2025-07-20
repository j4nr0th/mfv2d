.. currentmodule:: mfv2d.refinement

refinement
==========

This submodule's purpose is to provide support functions for mesh refinement,
namely the ones involved in post-solver refinement. Currently heavily work in
progress.


Utilities
---------

For computing Legendre decomposition of functions for error estimation, the
function :func:`_compute_legendre_coefficients` is provided.


.. autofunction:: _compute_legendre_coefficients


Refinement Settings
-------------------

Refinement settings are specified using the type :class:`RefinementSettings`.
There are a few additional types introduced to help type hint and define
all parameters (for example :class:`ErrorCalculationFunction`).

.. autoclass:: RefinementSettings

.. autoclass:: ErrorCalculationFunction

Refinement Limits
~~~~~~~~~~~~~~~~~

To specify when the refineemnt should stop, several different dataclasses are
available. :class:`RefinementLimitUnknownCount` limits it based on number of
new degrees of freedom that are introduced,
:class:`RefinementLimitElementCount` is based on the number of elements
refined, added, and :class:`RefinementLimitErrorValue` is based on the
error value of elements.

.. autoclass:: RefinementLimitUnknownCount

.. autoclass:: RefinementLimitElementCount

.. autoclass:: RefinementLimitErrorValue



Doing the Refinement
--------------------

After the solver obtains a solution for the problem, refinement is performed
by invoking the function :func:`perform_mesh_refinement`. This takes values
computed by the solver, as well as all that is specified through
:class:`RefinementSettings`.

.. autofunction:: perform_mesh_refinement
