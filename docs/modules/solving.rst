.. currentmodule:: mfv2d.solving

solving
=======

This sub-module contains functions and types that are used to implement
different solvers for the system of equations. They rely on the types
implemented in C to speed the most time consuming parts of the calculations.

The key idea behind all these is to write it in a way that would make it
straight forward to port to a distributed memory architecture. As such,
data and operations are made in a way which requires minimal communication
between the different element data.


Support Types
-------------

A wrapper around :class:`mfv2d._mfv2d.LinearSystem` is provided by
:class:`LinearSystem`. This provides an easier to deal with constructor,
as well as a few convenience methods.

.. autoclass:: LinearSystem


To make writing some solvers easier :class:`FullVector` is provided.
This type just packs the dense parts represented by
:class:`mfv2d._mfv2d.DenseVector` and the trace parts
:class:`mfv2d._mfv2d.TraceVector` together and provides methods
these two provide.

.. autoclass:: FullVector


Core Solvers
------------

The solvers implemented in this sub-module all follow the phylosophy that
they should be type-agnostic. As such, they all take the system along with
all functions needed to operate on it. This allows the same solver to be
used for different vector and system types by just writing a few short
wrapper functions.

.. autofunction:: gmres_general

.. autofunction:: cg_general

.. autofunction:: pcg_general


System Solvers
--------------

Solvers which actually solve the system are based on the core solvers
and use them at some point, though the specific way the do it can differ.

GMRES
~~~~~

GMRES may be slow and memory hungry, but it is guaranteed to give you the
best solution possible from a Krylov method solver. It solves the system
by using the :class:`FullVector` and solve it all at once.

.. autofunction:: solve_gmres_iterative


CG
~~~

Conjugate gradient is the optimal method for symmetric positive definite (SPD)
matrices. Unfortunately, mixed problems (mixed Poisson, Navier-Stokes) give
rise to indefinete problems, which means that CG will likely terminate due to
the degeneration of the Krylov subspace.

.. autofunction:: solve_cg_iterative


PCG
~~~

Preconditioned conjugate gradient can deal with the problems of CG solver
by applying a preconditioner to the system. This allows for both faster
convergence and nicer behavior with indefinete systems. It uses block-Jacobi
preconditioner.

.. autofunction:: solve_pcg_iterative


Schur's Complement
~~~~~~~~~~~~~~~~~~

Schur's complement is the best performing solver I have written thus far.
The idea is to use Gaussian eleimination to obtain the system for Lagrange
multipliers using the equation :eq:`eq-schur-complement-1`, where
:math:`\vec{\lambda}` are the Lagrange multipliers, :math:`\mathbf{N}`
the matrix that enforces the constraints, :math:`\mathbf{A}` the
block diagonal part, and the :math:`\vec{y}` and :math:`\vec{\phi}`
being the dense and trace forcing respectively.

While the matrix :math:`\mathbf{N} \mathbf{A}^{-1} \mathbf{N}^T` is dense,
the is solved iteratively using PCG in a matrix-free manner. This is because
the dense solve would scale with the cube of the degrees of freedom.

.. math::
    :label: eq-schur-complement-1

    \mathbf{N} \mathbf{A}^{-1} \mathbf{N}^T \vec{\lambda} = \mathbf{N} \mathbf{A}^{-1} \vec{y} - \phi
