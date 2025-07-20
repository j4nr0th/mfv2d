.. currentmodule:: mfv2d.progress

progress
========

This is a small submodule which contains code to make printing progress
reports and feedback pretty.

To report on the state of the solver, the :class:`ProgressTracker` type
is used, which contains state about where the solver stated, when it
should stop and what is the current state. It also has methods that
allow it to have pretty terminal output.

.. autoclass:: ProgressTracker


For reporting on error distribution and distribution of element orders,
the type :class:`HistogramFormat` is used. It can behave oddly for very
low number of elements.

.. autoclass:: HistogramFormat
