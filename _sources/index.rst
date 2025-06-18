.. mfv2d documentation master file, created by
   sphinx-quickstart on Sun Jun 15 15:18:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mfv2d documentation
===================

MFV2D aims to be a 2D differential equation solver based on mimetic spectral
element method (MSEM), as outlined by :cite:author:`gerritsma_mimetic_2018`.
The solver is capable of working on semi-structured deformed quadrilateral
meshes. These meshes can be locally refined hierarchically by dividing
elements, polynomially by increasing order of any elements, or both combined.

In the future the goal is to incorporate variational multi-scale (VMS) theory
based on work by :cite:author:`shrestha_optimal_2025`.

The package is written partially in C and partially in Python. The idea
behind it is that only performance critical code ought to be written in C,
as writing, testing, and debugging it is significantly harder than for Python.
As such, make sure to first profile the code to determine if it should be
rewritten to C, then introduce enough tests to be sure any errors that can
happen are not the result of the newly added C code. Also make sure to run
these tests with valgrind :cite:`valgrind` to check for any memory
corruption or leaks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basis
   modules

.. bibliography::
