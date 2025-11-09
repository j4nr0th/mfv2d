.. currentmodule:: mfv2d.refinement

refinement
==========

This submodule's purpose is to provide support functions for mesh refinement,
namely the ones involved in post-solver refinement. Currently heavily work in
progress.


Utilities
---------

For computing Legendre decomposition of functions for error estimation, the
function :func:`compute_legendre_coefficients` is provided. It is called by
:func:`compute_legendre_error_estimates`, which is used to estimate the
:math:`L^2` error estimate and the approximate increase in the :math:`L^2`
error due to order drop with :math:`h`-refinement.


.. autofunction:: compute_legendre_error_estimates

.. autofunction:: compute_legendre_coefficients


Refinement Settings
-------------------

Refinement settings are specified using the type :class:`RefinementSettings`.
The two most important members of it are the ``error_estimate`` and
the ``refinement_limit`` members.

.. autoclass:: RefinementSettings



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


Error Estimates
~~~~~~~~~~~~~~~

Error can be specified in different ways. These differ in time and accuracy.

.. autoclass:: ErrorEstimateCustom

.. autoclass:: ErrorEstimateLocalInverse

.. autoclass:: ErrorEstimateL2OrderReduction

.. autoclass:: ErrorEstimateExplicit

.. autoclass:: ErrorEstimateVMS


Each of these has its respective error estimation function:

- :class:`ErrorEstimateCustom` has :func:`error_estimate_with_custom_estimator`
- :class:`ErrorEstimateLocalInverse` has
  :func:`error_estimate_with_local_inversion`
- :class:`ErrorEstimateL2OrderReduction` has
  :func:`error_estimate_with_order_reduction`
- :class:`ErrorEstimateExplicit` has
  :func:`error_estimate_with_explicit_solution`
- :class:`ErrorEstimateVMS` has :func:`error_estimate_with_vms`

.. autofunction:: error_estimate_with_custom_estimator

.. autofunction:: error_estimate_with_local_inversion

.. autofunction:: error_estimate_with_order_reduction

.. autofunction:: error_estimate_with_explicit_solution

.. autofunction:: error_estimate_with_vms



Doing the Refinement
--------------------

After the solver obtains a solution for the problem, refinement is performed
by invoking the function :func:`perform_mesh_refinement`. This takes values
computed by the solver, as well as all that is specified through
:class:`RefinementSettings`. After it calls the error estimation function
of the error estimate specified, :func:`refine_mesh_based_on_error` is called
to perform the refinement based on the computed error measures.

.. autofunction:: perform_mesh_refinement

.. autofunction:: refine_mesh_based_on_error
