.. currentmodule:: mfv2d.eval

eval
====

This submodule is responsible for conversion of system from its abstract syntax
tree (AST) representation, which is the job of :mod:`mfv2d.kform` submodule,
into a form which can be understood and executed by C code in
:mod:`mfv2d._mfv2d`.

This process is done in the following steps:

1. First, the AST is converted into Python bytecode-like objects
   (subtypes of :class:`MatOp`).
2. Resulting expression is simplified as much as possible by
   the :func:`simplify_expression` function.
3. Simplified expression is translated into C compatible values
   (sequence of :class:`int`, :class:`float`, and
   :class:`MatOpCode` values).

After this is done for each of the blocks of the in the system,
the two-dimensional square sequence can be passed to the
:func:`mfv2d._mfv2d.compute_element_matrix` or
:func:`mfv2d._mfv2d.compute_element_vector` functions.


Matrix Operations
-----------------

For the first step in instruction generation, subtypes of
the :class:`MatOp` type are used. These are all documented
bellow.

.. autoclass:: MatOp

.. autoclass:: Identity

.. autoclass:: MassMat

.. autoclass:: Incidence

.. autoclass:: Push

.. autoclass:: MatMul

.. autoclass:: Scale

.. autoclass:: Sum

.. autoclass:: InterProd


AST Translation
---------------

The translation from the AST form into the Python bytecode
is performed by :func:`translate_equation` function. This
is a thin wrapper around :func:`_translate_equation` function,
which produces very sub-optimal bytecode. As such, the function
will (when specified) call :func:`simplify_expression` to simplify
the bytecode before returning.


.. autofunction:: translate_equation

.. autofunction:: _translate_equation


To check if the AST was correctly translated, or when debugging,
the expected result can be obtained by a call to
:func:`print_eval_procedure` which will convert the iterable of
:class:`MatOp` into a :class:`str`.

.. autofunction:: print_eval_procedure


Bytecode Simplifictaion
-----------------------

Simplifictaion of the bytecode is handled by the :func:`simplify_expression`
function. It mainly focuses on eliminating :class:`Identity` operations,
fusing together :class:`Scale`, or applying :class:`MassMat` to its
invers.

.. autofunction:: simplify_expression


Conversion to C Bytecode
------------------------

Conversion into "C-friendly" bytecode is done by converting the :class:`MatOp`
values into :class:`MatOpCode`, :class:`int`, or :class:`float` values.

.. autoclass:: MatOpCode
    :no-inherited-members:
    :member-order: bysource

The actual translation is handled by the :func:`translate_to_c_instructions`
function.

.. autofunction:: translate_to_c_instructions


Putting it All Togehter
-----------------------

Since the steps and related function and types are never used outside of
converting AST code into C bytecode, the functionality is wrapped
by the :func:`translate_system` function, which performs the conversion
of an entire :class:`mfv2d.kform.KFormSystem` into the bytecode.

.. autofunction:: translate_system


This is abstracted further into a :class:`CompiledSystem` type, which
automatically extracts linear, non-linear, right implicit, and left
implicit terms, which is what the solver actually needs.

.. autoclass:: CompiledSystem
