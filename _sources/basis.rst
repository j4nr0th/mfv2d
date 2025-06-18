Mimetic Spectral Elements Method
================================

In mfv2d, the MSEM method is formulated using the hybridized approach.
This means that the equations are expressed on each element on its own,
with continuity being enforced later through Lagrange multipliers.
This approach allows for easier implementation of local refinement.

Basis functions used depend on the space in which the unknown is
represented in. There are three spaces which can be used:

.. math::

    {H}\left(\mathrm{curl}\right) \xrightarrow{\mathrm{curl}} {H}\left(\mathrm{div}\right)
    \xrightarrow{\mathrm{div}} L^2

Each of these spaces represent differential :math:`k`-forms of 0-th, 1-st,
and 2-nd order respectively. These also have (implicit) dual spaces, with
which they form the double de Rahm complex:

.. math::

    \tilde{L}^2 \xleftarrow{\mathrm{curl}} \tilde{H}\left(\mathrm{curl}\right)
    \xleftarrow{-\mathrm{grad}} \tilde{H}\left(\mathrm{grad}\right)

Mapping between these is done using the Hodge operator, which is computed from
basis inner-products described by :cite:author:`jain_construction_2020`. As in
that paper, the basis functions for these spaces are formed by tensor product
grids of nodal Lagrange polynomials and their derivatives.

Basis
-----

One dimensional basis are simply Lagrange basis polynomials, defined on
a set of Gauss-Lobatto-Legendre nodes. These nodes can be computed by using
the
