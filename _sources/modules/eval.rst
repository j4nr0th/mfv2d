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

.. autoclass:: Scale

.. autoclass:: Sum

.. autoclass:: InterProd


AST Translation
---------------

The translation from the AST form into the Python bytecode
is performed by :func:`translate_implicit_ksum` function. This
calls :func:`_translate_inner_prod` function, which in turn relies on
:func:`_translate_form`. The bytecode these produce may be sub-optimal.
As such, the function will (when specified) call
:func:`simplify_expression` to simplify the bytecode before returning.
This mainly has do deal with eliminating matrices followed by their inverse.

.. autofunction:: translate_implicit_ksum

.. autofunction:: _translate_inner_prod

.. autofunction:: _translate_form


To cache these translation results and to be able to explicitly
choose only linear, nonlinear, or explicit terms the type
:class:`CompiledSystem` is provided.

.. autoclass:: CompiledSystem


To check if the AST was correctly translated, or when debugging,
the expected result can be obtained by a call to
:func:`system_as_string` which will convert the system into instructions,
then print the resulting (simplified) code, exactly as the evaluation code
would. To support this function, support functions
:func:`_translate_expr_to_str`,
:func:`_translate_codes_to_str`, :func:`bytecode_matrix_as_rows`, and
:func:`explicit_ksum_as_string` are available.

.. autofunction:: system_as_string

.. autofunction:: _translate_expr_to_str

.. autofunction:: _translate_codes_to_str

.. autofunction:: bytecode_matrix_as_rows

.. autofunction:: explicit_ksum_as_string


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
function. Some basic type hits are introduced to allow for type hinting the
output of :func:`translate_to_c_instructions`.

.. autofunction:: translate_to_c_instructions
