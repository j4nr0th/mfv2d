.. currentmodule:: mfv2d.kform

kform
=====

This submodule is provides types which allow for creating a description
of the system to solve using Python's operator overloading. It also
allows for caching any syntax errors with it. It could be even further
improved in the future if `Python's generic typing <https://typing.python.org/en/latest/reference/generics.html>`_
would be used.


Different :math:`k`-Form Types
------------------------------

Most of this module's contents consists of types that allow describing
aforementioned :math:`k`-form relations. These are all subtypes of the
:class:`Term` type. It only describes how to describe/print a term and
nothing more.

.. autoclass:: Term


Next is the basic :class:`KForm` type. In addition to its label it also
has its order. This can have a value of 0, 1, or 2. It has quite a few
useful operators implemented.

.. autoclass:: KForm
    :no-inherited-members:


To describe order of a form older functions made use of just taking the
integers and assuming they were in the desired range. Newly written code should
instead make use of :class:`UnknownFormOrder` type, which is an enum.

.. autoclass:: UnknownFormOrder
    :no-inherited-members:

For a system, these :class:`UnknownFormOrder` can be stored in
:class:`UnknownOrderings`. This is just a glorified :class:`tuple`.

.. autoclass:: UnknownOrderings
    :no-inherited-members:


To denote variables that should be solved for, :class:`KFormUnknown`
are used. They also can be used for non-linear interior products
using the ``^`` operator.

.. autoclass:: KFormUnknown
    :no-inherited-members:


Each unknown form can also be used to create a weight form with
the type :class:`KWeight` throught the :meth:`KFormUnknown.weight`
method. These are used for forming interior products, as well as
forming either element projections
(described by :class:`KElementProjection`) or boundary projections
for weak boundary conditions (described by :class:`KBoundaryProjection`).

.. autoclass:: KWeight
    :no-inherited-members:


:math:`k`-forms resulting from different operations also each have
a different type:

- For the Hodge operator, :class:`KHodge` type is used
- For the derivative :class:`KFormDerivative` is used
- For the interor product with a vector function :class:`KInteriorProduct`
- For the interor product with an unknown form :class:`KInteriorProductNonlinear`

.. autoclass:: KHodge
    :no-inherited-members:

.. autoclass:: KFormDerivative
    :no-inherited-members:

.. autoclass:: KInteriorProduct
    :no-inherited-members:

.. autoclass:: KInteriorProductNonlinear
    :no-inherited-members:


Forming Expressions
-------------------

From these basic building blocks, :class:`TermEvaluatable` objects can
be created. These represent an expression, which can be evaluated,
provided degrees of freedom of associated :class:`KFormUnknown` are known.
Quite a few operators are implemented on it, which allow for scaling,
adding, or subtracting these together.

.. autoclass:: TermEvaluatable
    :no-inherited-members:


The most basic of these is the :class:`KInnerProduct`, which represents
an inner product between a weight form and an unknown form.

.. autoclass:: KInnerProduct
    :no-inherited-members:

Next is the family of :class:`KExplicit` terms. These do not depend on any
unknown forms and can as such be evaluated explicity. These also may only
appear on the right side of any :class:`KEquation` formed.

.. autoclass:: KExplicit
    :no-inherited-members:


First of the :class:`KExplicit` subtypes is the :class:`KElementProjection`,
which is used to represent the :class:`L^2` projection on the element.

.. autoclass:: KElementProjection
    :no-inherited-members:


Second is the the :class:`KBoundaryProjection`, which represents a boundary
integral for an element. It is used to describe weak boundary conditions.

.. autoclass:: KBoundaryProjection
    :no-inherited-members:


As the last :class:`TermEvaluatable` subtype there is :class:`KSum` type.
This represents a linear combination of other :class:`TermEvaluatable`
types. It is actually a linear combination of the terms, as each can
have its own coefficient. When two :class:`KSum` objects are added together,
they are concatenated, so :class:`KSum` should never contain another
:class:`KSum`. All terms in the sum must use the exact same weight form.

.. autoclass:: KSum
    :no-inherited-members:

Equations and Systems
---------------------

When two expressions are related as being equal with the ``==`` operator,
the result is a :class:`KEquation`. It consists of a left and right side.
The left must contain at least one :class:`KInnerProduct` term and no
:class:`KExplicit` terms. Both sides must also use a matching weight form.

.. autoclass:: KEquation

A collection of equations form a :class:`KFormSystem`. This type also provides
useful utilities for identifying different terms, extracting vector fileds,
weights, unknowns, and others.

.. autoclass:: KFormSystem
