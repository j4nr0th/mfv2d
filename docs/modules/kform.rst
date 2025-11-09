.. currentmodule:: mfv2d.kform

kform
=====

This submodule is provides types which allow for creating a description
of the system to solve using Python's operator overloading. It also
allows for caching any syntax errors with it. It could be even further
improved in the future if `Python's generic typing <https://typing.python.org/en/latest/reference/generics.html>`_
will ever support inheritence for generic types.


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


To denote variables that should be solved for, :class:`KFormUnknown`
are used. They also can be used for non-linear interior products
using the ``*`` operator.

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

- For the derivative :class:`KFormDerivative` is used
- For the interor product with a vector function :class:`KInteriorProduct`
- For the interor product with an unknown form
  :class:`KInteriorProductLowered`

.. autoclass:: KFormDerivative
    :no-inherited-members:

.. autoclass:: KInteriorProduct
    :no-inherited-members:

.. autoclass:: KInteriorProductLowered
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


Supporting Functions
--------------------

Some minor support functions for parsing and dealing with :math:`k`-forms
are also provided. These used to be methods, but having them as functions
is easier to deal with. And to refactor, since all implementations for
different types are in the same spot (God bless Python's ``match`` statement).

.. autofunction:: extract_base_form

.. autofunction:: extract_unknown_forms

.. autofunction:: check_form_linear


Operations
----------

Different operators are supported by different :math:`k`-forms.
The overview is shown in the table bellow.


.. list-table::
    :header-rows: 1

    * - operation
      - required type(s)
      - required order(s)
      - operator
      - result
    * - derivative
      - :class:`KForm`
      - (0, 1)
      - ``<KForm>.derivative``
      - :class:`KFormDerivative`
    * - interior product (linear)
      - (callable, :class:`KForm`)
      - (1, 2)
      - ``func * <KForm>``
      - :class:`KInteriorProduct`
    * - interior product (non-linear)
      - (:class:`KForm`, :class:`KForm`)
      - (1, (1, 2))
      - ``<KForm> * <KForm>``
      - :class:`KInteriorProductLowered`
    * - element projection
      - (:class:`KWeight`, callable)
      - Any
      - ``<KWeight> @ callable``
      - :class:`KElementProjection`
    * - boundary projection
      - (:class:`KWeight`, callable)
      - 0, 1
      - ``<KWeight> ^ callable``
      - :class:`KBoundaryProjection`
    * - inner product
      - (:class:`KForm`, :class:`KForm`)
      - (:math:`k`, :math:`k`)
      - ``<KForm> @ <KForm>``
      - :class:`KInnerProduct`
    * - scale
      - :class:`Term`
      - None
      - ``k * <Term>``
      - :class:`KSum`
    * - add
      - (:class:`Term`, :class:`Term`)
      - None
      - ``<Term> + <Term>``
      - :class:`KSum`
    * - sub
      - (:class:`Term`, :class:`Term`)
      - None
      - ``<Term> - <Term>``
      - :class:`KSum`
