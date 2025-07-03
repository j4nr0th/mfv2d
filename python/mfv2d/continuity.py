"""Functions that generate continuity constraints.

There are two main types of continuity constraints, based on
where they are comming from:

- intra-element
- inter-element

When it comes to how the application process is done, there
are also two different types:

- edge-based for 1-forms and 0-forms on edge interior,
- node-based for 0-forms on corners,
"""

import numpy as np

from mfv2d.element import (
    Constraint,
    ElementCollection,
    ElementConstraint,
    FixedElementArray,
    UnknownFormOrder,
    UnknownOrderings,
    element_node_children_on_side,
    get_corner_dof,
    get_side_dofs,
    get_side_order,
)
from mfv2d.mimetic2d import ElementSide, Mesh2D, find_surface_boundary_id_line


def _find_surface_boundary_id_node(
    mesh: Mesh2D, surf_idx: int, node_idx: int
) -> ElementSide:
    """Find what boundary begins with the node with a given index is in the surface.

    Parameters
    ----------
    mesh : Mesh2D
        Mesh the surface is a part of.

    surf_idx : int
        Index of the surface.

    node_idx : int
        Index of the node.

    Returns
    -------
    ElementSide
        Side of the element which begins with the given node. If the node is not
        in the surface an exception is raised.
    """
    s = mesh.primal.get_surface(surf_idx + 1)
    for line_id, bnd_id in zip(iter(s), ElementSide, strict=True):
        line = mesh.primal.get_line(line_id)
        if line.begin.index == node_idx:
            return bnd_id

    raise ValueError(f"Node with index {node_idx=} is not in the surface {surf_idx=}.")


def connect_corner_based(
    elements: ElementCollection,
    *pairs: tuple[int, ElementSide],
) -> list[Constraint]:
    """Create constraints for 0-forms on the corner.

    Parameters
    ----------
    elements : ElementCollection
        Element collection to which the two elements belong to.

    *pairs : tuple of (int, ElementSide)
        Elements that should be connected as pairs of element and side
        that the corner is the beginning for.

    Returns
    -------
    list of Constraints
        Constraints which enforce continuity between the elements for 0-forms.
    """
    n = len(pairs)
    constraints: list[Constraint] = list()

    e1, s1 = pairs[0]
    l1, d1 = get_corner_dof(elements, e1, s1)
    for i in range(n - 1):
        e2, s2 = pairs[i + 1]
        l2, d2 = get_corner_dof(elements, e2, s2)

        constraints.append(
            Constraint(
                0.0,
                ElementConstraint(
                    l1, np.array([d1], np.uint32), np.array([+1], np.float64)
                ),
                ElementConstraint(
                    l2, np.array([d2], np.uint32), np.array([-1], np.float64)
                ),
            )
        )
        e1, s1 = e2, s2
        l1, d1 = l2, d2

    return constraints


def connect_edge_center(
    elements: ElementCollection, e1: int, e2: int, side: ElementSide
) -> list[Constraint]:
    """Connect center of edges for two elements with 0-forms.

    Parameters
    ----------
    elements : ElementCollection
        Element collection to which the elements belong to.

    e1 : int
        Index of the first element.

    e2 : int
        Index of the second element.

    side : ElementSide
        Side of the elements that is to be connected.

    Returns
    -------
    list of Constraints
        Constraints which enforce continuity between the two elements for
        0-forms in the middle of the side.
    """
    constraints = connect_corner_based(elements, (e1, side.next), (e2, side))
    c1 = elements.child_array[e1]
    c2 = elements.child_array[e2]

    if len(c1):
        c11, c12 = element_node_children_on_side(side, c1)
        constraints += connect_edge_center(elements, c11, c12, side)

    if len(c2):
        c21, c22 = element_node_children_on_side(side, c2)
        constraints += connect_edge_center(elements, c21, c22, side)

    return constraints


def connect_edge_based(
    elements: ElementCollection,
    e1: int,
    s1: ElementSide,
    e2: int,
    s2: ElementSide,
    form_order: UnknownFormOrder,
) -> list[Constraint]:
    """Create constraints for 0-forms or 1-forms on edges.

    Parameters
    ----------
    elements : ElementCollection
        Element collection to which the two elements belong to.

    e1 : int
        Index of the first element.

    s1 : ElementSide
        Side of the first element that is connected to the second.

    e2 : int
        Index of the second element.

    s2 : ElementSide
        Side of the second element that is connected to the second.

    form_order : UnknownFormOrder
        Order of the unknown form to handle. Note that only 0-forms and
        1-forms are valid, since 2-forms have no boundary DoFs.

    Returns
    -------
    list of Constraints
        Constraints which enforce continuity between the two elements for
        0-forms or 1-forms.
    """
    assert (
        form_order == UnknownFormOrder.FORM_ORDER_0
        or form_order == UnknownFormOrder.FORM_ORDER_1
    )
    c1 = elements.child_array[e1]
    c2 = elements.child_array[e2]
    constraints: list[Constraint] = list()
    if len(c1) and len(c2):
        c11, c12 = element_node_children_on_side(s1, c1)
        c21, c22 = element_node_children_on_side(s2, c2)
        constraints_1 = connect_edge_based(elements, c11, s1, c22, s2, form_order)
        constraints_2 = connect_edge_based(elements, c12, s1, c21, s2, form_order)
        constraints_3: list[Constraint] = list()
        if form_order == UnknownFormOrder.FORM_ORDER_0:
            constraints_3 = connect_corner_based(
                elements,
                (c11, s1.next),
                (c12, s1),
                (c22, s2),
                (c21, s2.next),
            )

        return constraints_1 + constraints_2 + constraints_3

    elif form_order == UnknownFormOrder.FORM_ORDER_0:
        # Connect the corner of two children if two are needed
        if len(c1):
            c11, c12 = element_node_children_on_side(s1, c1)
            constraints += connect_edge_center(elements, c11, c12, s1)

        elif len(c2):
            c21, c22 = element_node_children_on_side(s2, c2)
            constraints += connect_edge_center(elements, c21, c22, s2)

    order_1 = get_side_order(elements, e1, s1)
    order_2 = get_side_order(elements, e2, s2)

    highest_order = max(order_1, order_2)

    # Do not do the corners for 0-forms
    dofs_1 = get_side_dofs(elements, e1, s1, form_order, highest_order)
    dofs_2 = get_side_dofs(elements, e2, s2, form_order, highest_order)

    if form_order == UnknownFormOrder.FORM_ORDER_0:
        dofs_1 = dofs_1[1:-1]
        dofs_2 = dofs_2[1:-1]

    if form_order == UnknownFormOrder.FORM_ORDER_0:
        sign = -1

    elif form_order == UnknownFormOrder.FORM_ORDER_1:
        sgn1 = 1 - (s1.value & 2)  # +1 for B, L and -1 for R, T
        sgn2 = 1 - (s2.value & 2)  # +1 for B, L and -1 for R, T
        sign = sgn1 * sgn2  # +1 when same orientation, otherwise -1

    else:
        assert False

    # Flip the order of DoFs for the second one, since the boundary is
    # the other way around.
    for d1, d2 in zip(dofs_1, reversed(dofs_2), strict=True):
        constraints.append(
            Constraint(
                0.0,
                *d1.element_constraints,
                *(
                    # Flip sign of coefficients for the second one
                    ElementConstraint(dof.i_e, dof.dofs, sign * dof.coeffs)
                    for dof in d2.element_constraints
                ),
            )
        )

    return constraints


def connect_element_inner(
    elements: ElementCollection,
    element: int,
    form_order: UnknownFormOrder,
) -> list[Constraint]:
    """Generate constraints for continuity within the element.

    Parameters
    ----------
    elements : ElementCollection
        Element collection the element belongs to.

    element : int
        Index of element to deal with.

    form_order : UnknownFormOrder
        Order of the form to use.

    Returns
    -------
    list of Constraint
        Constraints which enforce the continuity within the element.

    """
    children = elements.child_array[element]
    if not len(children):
        # Leaf has no need for constraints
        return list()

    c_bl: int
    c_br: int
    c_tr: int
    c_tl: int

    c_bl, c_br, c_tr, c_tl = children

    # Inner constraints
    child_constraints: list[Constraint] = sum(
        (connect_element_inner(elements, c, form_order) for c in children), start=list()
    )

    # Edge connections between child elements
    edge_constraints = (
        connect_edge_based(
            elements,
            c_bl,
            ElementSide.SIDE_RIGHT,
            c_br,
            ElementSide.SIDE_LEFT,
            form_order,
        )
        + connect_edge_based(
            elements,
            c_br,
            ElementSide.SIDE_TOP,
            c_tr,
            ElementSide.SIDE_BOTTOM,
            form_order,
        )
        + connect_edge_based(
            elements,
            c_tr,
            ElementSide.SIDE_LEFT,
            c_tl,
            ElementSide.SIDE_RIGHT,
            form_order,
        )
        + connect_edge_based(
            elements,
            c_tl,
            ElementSide.SIDE_BOTTOM,
            c_bl,
            ElementSide.SIDE_TOP,
            form_order,
        )
    )

    corner_constraint: list[Constraint] = list()
    if form_order == UnknownFormOrder.FORM_ORDER_0:
        # Add corner constraint if 0-form
        corner_constraint = connect_corner_based(
            elements,
            (c_bl, ElementSide.SIDE_TOP),
            (c_br, ElementSide.SIDE_LEFT),
            (c_tr, ElementSide.SIDE_BOTTOM),
            (c_tl, ElementSide.SIDE_RIGHT),
        )

    return child_constraints + edge_constraints + corner_constraint


def connect_elements(
    elements: ElementCollection,
    unknowns: UnknownOrderings,
    mesh: Mesh2D,
    dof_offsets: FixedElementArray[np.uint32],
) -> list[Constraint]:
    """Generate constraints for all elements and unknowns.

    Parameters
    ----------
    elements : ElementCollection
        Element collection that contains all the elements.

    unknowns : UnknownOrderings
        Orders of unknown forms defined for all elements.

    mesh : Mesh2D
        Mesh with primal and dual topology of root elements.

    dof_offsets : FixedElementArray[np.uint32]
        Array of offsets for degrees of freedom in the elements.

    Returns
    -------
    list of Constraint
        List with constraints which enforce continuity between degrees of freedom
        for unknown forms defined between all elements.
    """
    has_0 = any(form == UnknownFormOrder.FORM_ORDER_0 for form in unknowns.form_orders)
    has_1 = any(form == UnknownFormOrder.FORM_ORDER_1 for form in unknowns.form_orders)

    intra_element_0: list[Constraint] = list()  # for 0-forms
    intra_element_1: list[Constraint] = list()  # for 1-forms

    # Generate all intra-element constraints
    for surf_index in range(mesh.primal.n_surfaces):
        ie = elements.root_indices[surf_index]
        if has_0:
            intra_element_0 += connect_element_inner(
                elements, ie, UnknownFormOrder.FORM_ORDER_0
            )
        if has_1:
            intra_element_1 += connect_element_inner(
                elements, ie, UnknownFormOrder.FORM_ORDER_1
            )

    # Generate inter-element constraints. This has two parts (if there are any 0-forms).
    #   First are the edge-based connections
    inter_edge_0: list[Constraint] = list()  # for 0-forms
    inter_edge_1: list[Constraint] = list()  # for 1-forms
    for edge_index in range(mesh.primal.n_lines):
        dual_line = mesh.dual.get_line(edge_index + 1)
        idx1 = dual_line.begin
        idx2 = dual_line.end

        if not idx1 or not idx2:
            # Boundary line, we do not constrain here, leave it for BCs
            continue

        e1 = int(elements.root_indices[idx1.index])
        e2 = int(elements.root_indices[idx2.index])

        surf_1 = mesh.primal.get_surface(idx1)
        surf_2 = mesh.primal.get_surface(idx2)

        side_1 = find_surface_boundary_id_line(surf_1, edge_index)
        side_2 = find_surface_boundary_id_line(surf_2, edge_index)

        if has_0:
            inter_edge_0 += connect_edge_based(
                elements, e1, side_1, e2, side_2, UnknownFormOrder.FORM_ORDER_0
            )

        if has_1:
            inter_edge_1 += connect_edge_based(
                elements, e1, side_1, e2, side_2, UnknownFormOrder.FORM_ORDER_1
            )

    # Next are the corner-based ones
    inter_corner_0: list[Constraint] = list()
    if has_0:
        # Check now, since 1-forms do not need these
        for node_index in range(mesh.primal.n_points):
            dual_surf = mesh.dual.get_surface(node_index + 1)

            # find all elements in this dual surface
            element_indices: list[int] = list()
            for dual_line_id in iter(dual_surf):
                # Iterate over dual lines in the mesh
                dual_line = mesh.dual.get_line(dual_line_id)
                primal_line = mesh.primal.get_line(dual_line_id)
                # According to my code, this should be true
                assert primal_line.begin.index == node_index
                e_idx = dual_line.begin
                if not e_idx:
                    # Boundary, so skip
                    continue
                element_indices.append(e_idx.index)

            if len(element_indices) == 1:
                # Corner of the actual mesh.
                continue

            inter_corner_0 += connect_corner_based(
                elements,
                *(
                    (
                        elements.root_indices[ie],
                        _find_surface_boundary_id_node(mesh, ie, node_index),
                    )
                    for ie in element_indices
                ),
            )

    # Combine all of these
    combined_0 = intra_element_0 + inter_edge_0 + inter_corner_0
    combined_1 = intra_element_1 + inter_edge_1

    # Now make adjustments as needed to correctly offset
    real_constraints: list[Constraint] = list()

    # First does not need offsets, since there is nothing before it
    for i_form, form in enumerate(unknowns.form_orders):
        if form == UnknownFormOrder.FORM_ORDER_0:
            base = combined_0
        elif form == UnknownFormOrder.FORM_ORDER_1:
            base = combined_1
        else:
            # No need to do anything here
            assert form == UnknownFormOrder.FORM_ORDER_2
            continue

        if i_form != 0:
            # Offset DoFs in the base constraints
            real_constraints += [
                Constraint(
                    0.0,
                    *(
                        ElementConstraint(
                            ec.i_e, ec.dofs + dof_offsets[ec.i_e][i_form], ec.coeffs
                        )
                        for ec in constraint.element_constraints
                    ),
                )
                for constraint in base
            ]
        else:
            real_constraints += base

    return real_constraints
