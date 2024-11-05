.. currentmodule:: interplib

***********
Polynomials
***********

.. _polynomials:

A polynomial is simply a function which can be represented as a linear combination
of monomials. The most straight forward representation is a polynomial that is written
out in power basis as:

.. math::

    p(x) = \sum\limits_{n=0}^N a_n x^n

In interplib, such representation of a polynomial is offered by the :class:`Polynomial1D`
class. While convenient, it should be noted, that such representation quickly starts
to suffer from numerical stability issues and the loss of accuracy, as whenever
:math:`x \neq 1` the highest terms will either tend close to zero, or explode. This can
be very worrying when trying to compute a polynomial for which most terms would be close
to canceling each other out.

To remidy such cases, Bernstein basis can be employed instead to write the polynomial.
While introducing a small performance penalty, the trade-off becomes quickly noticeable
as the order of polynomials increases.
