.. currentmodule:: mfv2d.progress

progress
========

This is a small submodule which serves no other purpose other than to
fill the command line with colored garbage an emojis to indicate
how screwed your solution is getting over the course of iterations.

About the only noteworthy thing in it is the type :class:`ProgressTracker`,
which contains state about where the solver stated, when it should stop
and what is the current state. It also has methods that allow it to
have pretty terminal output.

.. autoclass:: ProgressTracker
