.. currentmodule:: mfv2d.continuity

continuity
==========

This is a submodule which contains functions used to deal with continuity.
This concerns connecting element boundaries together and generating
continuity equations.


Functions used to connect elements together on edges or corners are
:func:`connect_corner_based` and :func:`connect_edge_center` for 0-forms,
and :func:`connect_edge_based` for both 0- and 1-forms.

.. autofunction:: connect_corner_based

.. autofunction:: connect_edge_center

.. autofunction:: connect_edge_based

These are all used for innner element connectivity function
:func:`connect_element_inner`.

.. autofunction:: connect_element_inner

This all culminates in :func:`connect_elements`, which generates all
constraints for mesh continuity.

.. autofunction:: connect_elements

There is also an internal function used in this module.

.. autofunction:: _find_surface_boundary_id_node
