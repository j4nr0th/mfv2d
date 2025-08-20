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

from __future__ import annotations

from collections.abc import Sequence
from itertools import accumulate

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from mfv2d._mfv2d import Mesh, compute_gll, lagrange1d
from mfv2d.boundary import BoundaryCondition2DSteady, mesh_boundary_conditions
from mfv2d.kform import KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import (
    Constraint,
    ElementConstraint,
    ElementSide,
    FemCache,
    element_boundary_dofs,
    element_node_children_on_side,
    find_surface_boundary_id_line,
    get_side_order,
)
from mfv2d.system import ElementFormSpecification, KFormSystem


def _find_surface_boundary_id_node(
    mesh: Mesh, surf_idx: int, node_idx: int
) -> ElementSide:
    """Find what boundary begins with the node with a given index is in the surface.

    Parameters
    ----------
    mesh : Mesh
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


def _get_corner_dof(mesh: Mesh, element: int, side: ElementSide, /) -> tuple[int, int]:
    """Get element index and degree of freedom index for the corner of the element.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side which begins with the corner that should be obtained.

    Returns
    -------
    element_id : int
        Index of the (leaf) element to which the corner belongs to.

    dof_index : int
        Index of the (0-form) degree of freedom which is in that corner.
    """
    children = mesh.get_element_children(element)
    if children is None:
        # Actual leaf
        order_1, order_2 = mesh.get_leaf_orders(element)

        if side == ElementSide.SIDE_BOTTOM:
            idx = 0
        elif side == ElementSide.SIDE_RIGHT:
            idx = order_1
        elif side == ElementSide.SIDE_TOP:
            idx = (order_1 + 1) * order_2 + order_1
        elif side == ElementSide.SIDE_LEFT:
            idx = order_2 * (order_1 + 1)
        else:
            raise ValueError(f"Invalid side given by {side=}")

        return (element, idx)

    child = children[side.value - 1]

    return _get_corner_dof(mesh, child, side)


def _get_side_dof_nodes(
    mesh: Mesh,
    element: int,
    side: ElementSide,
    order: UnknownFormOrder,
    /,
) -> list[ElementConstraint]:
    """Get equations for obtaining DoFs on the side for the element.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side of the element to get the DoFs for.

    order : UnknownFormOrder
        Order of unknown forms for which to get the form orders from. It
        can only be ``UnknownFormOrder.FORM_ORDER_0`` or
        ``UnknownFormOrder.FORM_ORDER_1``, since 2-forms do not have any
        boundary DoFs.

    Returns
    -------
    constraints : list of ElementConstraint
        Specifiecation of which *leaf* elements are involved. These
        are also ordered. The ``coeff`` member specifies positions
        of these in the element.
    """
    children = mesh.get_element_children(element)
    if children is not None:
        c1: int
        c2: int
        c1, c2 = element_node_children_on_side(side, children)

        dofs1 = _get_side_dof_nodes(mesh, c1, side, order)
        dofs2 = _get_side_dof_nodes(mesh, c2, side, order)

        if order == UnknownFormOrder.FORM_ORDER_0:
            # Do not include the last row (DoF shared between the two)
            # since it overlaps with M1. Otherwise it overconstrains.
            dofs2[0] = ElementConstraint(
                dofs2[0].i_e, dofs2[0].dofs[1:], dofs2[0].coeffs[1:]
            )
        elif order == UnknownFormOrder.FORM_ORDER_1:
            # Still have to remove coeffs (since that stands for child nodes!)
            dofs2[0] = ElementConstraint(dofs2[0].i_e, dofs2[0].dofs, dofs2[0].coeffs[1:])
        else:
            assert False

        combined_dofs = [
            ElementConstraint(dof.i_e, dof.dofs, (dof.coeffs - 1) / 2) for dof in dofs1
        ] + [ElementConstraint(dof.i_e, dof.dofs, (dof.coeffs + 1) / 2) for dof in dofs2]

        return combined_dofs

    # This is a leaf
    n1, n2 = mesh.get_leaf_orders(element)

    indices = element_boundary_dofs(side, order, n1, n2)

    side_orders = mesh.get_leaf_orders(element)
    side_order = side_orders[(side.value - 1) & 1]

    return [
        ElementConstraint(
            mesh.get_leaf_index(element), indices, compute_gll(side_order)[0]
        )
    ]


def _get_side_dofs(
    mesh: Mesh,
    element: int,
    side: ElementSide,
    form_order: UnknownFormOrder,
    output_order: int | None = None,
    /,
) -> tuple[Constraint, ...]:
    """Get DoFs on the boundary in terms of leaf element DoFs.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

    element : int
        Element which to get DoF equations for.

    side : ElementSide
        Side of the element to get the DoFs for.

    order : UnknownFormOrder
        Order of unknown forms for which to get the form orders from. It
        can only be ``UnknownFormOrder.FORM_ORDER_0`` or
        ``UnknownFormOrder.FORM_ORDER_1``, since 2-forms do not have any
        boundary DoFs.

    Returns
    -------
    tuple of Constraint
        Tuple of constraints, each of which specifies how the "normal" DoFs
        on the element's boundary may be constructed from DoFs of the element.
    """
    self_order = get_side_order(mesh, element, side)

    # If output order is not specified, use own order
    if output_order is None:
        output_order = self_order

    if mesh.get_element_children(element) is None and output_order == self_order:
        # fast track for leaf elements with no projection, since it should be identity
        indices = element_boundary_dofs(side, form_order, *mesh.get_leaf_orders(element))
        out_c = tuple(
            Constraint(
                0.0,
                ElementConstraint(
                    mesh.get_leaf_index(element),
                    np.array([idx], np.uint32),
                    np.ones(1, np.float64),
                ),
            )
            for idx in indices
        )
        return out_c

    dofs = _get_side_dof_nodes(mesh, element, side, form_order)

    self_nodes = compute_gll(self_order)[0]
    input_nodes = np.concatenate([dof.coeffs for dof in dofs])

    # Values of output basis (axis 1) at input points (axis 0)
    # nodal_basis_vals and edge_basis_vals are maps from parent dofs to child dofs
    nodal_basis_vals = lagrange1d(self_nodes, input_nodes)
    if form_order == UnknownFormOrder.FORM_ORDER_0:
        m = np.linalg.inv(nodal_basis_vals)

    elif form_order == UnknownFormOrder.FORM_ORDER_1:
        diffs = nodal_basis_vals[:-1, :] - nodal_basis_vals[+1:, :]
        edge_basis_vals = np.stack(
            [x for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))],
            axis=-1,
            dtype=np.float64,
        )
        m = np.linalg.inv(edge_basis_vals)

    elif form_order == UnknownFormOrder.FORM_ORDER_2:
        raise ValueError("2-forms have no boundary DoFs.")

    else:
        raise ValueError(f"Invalid for order {form_order=}.")

    if self_order != output_order:
        # Values of output basis (axis 1) at input points (axis 0)
        # nodal_basis_vals and edge_basis_vals are maps from parent dofs to child dofs
        output_nodes = compute_gll(output_order)[0]
        map_nodal_basis_vals = lagrange1d(self_nodes, output_nodes)
        if form_order == UnknownFormOrder.FORM_ORDER_0:
            m = map_nodal_basis_vals @ m

        elif form_order == UnknownFormOrder.FORM_ORDER_1:
            diffs = map_nodal_basis_vals[:-1, :] - map_nodal_basis_vals[+1:, :]
            map_edge_basis_vals = np.stack(
                [
                    x
                    for x in accumulate(diffs[..., i] for i in range(diffs.shape[-1] - 1))
                ],
                axis=-1,
                dtype=np.float64,
            )
            m = map_edge_basis_vals @ m

        elif form_order == UnknownFormOrder.FORM_ORDER_2:
            raise ValueError("2-forms have no boundary DoFs.")

        else:
            raise ValueError(f"Invalid for order {form_order=}.")

    constraints: list[Constraint] = list()

    vrow: npt.NDArray[np.float64]
    for vrow in m:
        col_offset = 0
        elem_constraints: list[ElementConstraint] = list()

        for elem_dofs in dofs:
            cnt = elem_dofs.dofs.size
            element_constraint = ElementConstraint(
                elem_dofs.i_e, elem_dofs.dofs, vrow[col_offset : col_offset + cnt]
            )
            col_offset += cnt
            elem_constraints.append(element_constraint)

        assert col_offset == vrow.size
        constraint = Constraint(0.0, *elem_constraints)
        constraints.append(constraint)

    return tuple(constraints)


def connect_corner_based(
    mesh: Mesh,
    *pairs: tuple[int, ElementSide],
) -> list[Constraint]:
    """Create constraints for 0-forms on the corner.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the elements are in.

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
    l1, d1 = _get_corner_dof(mesh, e1, s1)
    for i in range(n - 1):
        e2, s2 = pairs[i + 1]
        l2, d2 = _get_corner_dof(mesh, e2, s2)

        constraints.append(
            Constraint(
                0.0,
                ElementConstraint(
                    mesh.get_leaf_index(l1),
                    np.array([d1], np.uint32),
                    np.array([+1], np.float64),
                ),
                ElementConstraint(
                    mesh.get_leaf_index(l2),
                    np.array([d2], np.uint32),
                    np.array([-1], np.float64),
                ),
            )
        )
        e1, s1 = e2, s2
        l1, d1 = l2, d2

    return constraints


def connect_edge_center(
    mesh: Mesh, e1: int, e2: int, side: ElementSide
) -> list[Constraint]:
    """Connect center of edges for two elements with 0-forms.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

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
    constraints = connect_corner_based(mesh, (e1, side.next), (e2, side))
    c1 = mesh.get_element_children(e1)
    c2 = mesh.get_element_children(e2)

    if c1 is not None:
        c11, c12 = element_node_children_on_side(side, c1)
        constraints += connect_edge_center(mesh, c11, c12, side)

    if c2 is not None:
        c21, c22 = element_node_children_on_side(side, c2)
        constraints += connect_edge_center(mesh, c21, c22, side)

    return constraints


def connect_edge_based(
    elements: Mesh,
    e1: int,
    s1: ElementSide,
    e2: int,
    s2: ElementSide,
    form_order: UnknownFormOrder,
) -> list[Constraint]:
    """Create constraints for 0-forms or 1-forms on edges.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

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
    c1 = elements.get_element_children(e1)
    c2 = elements.get_element_children(e2)
    constraints: list[Constraint] = list()
    if c1 is not None and c2 is not None:
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
        if c1 is not None:
            c11, c12 = element_node_children_on_side(s1, c1)
            constraints += connect_edge_center(elements, c11, c12, s1)

        elif c2 is not None:
            c21, c22 = element_node_children_on_side(s2, c2)
            constraints += connect_edge_center(elements, c21, c22, s2)

    order_1 = get_side_order(elements, e1, s1)
    order_2 = get_side_order(elements, e2, s2)

    highest_order = max(order_1, order_2)

    # Do not do the corners for 0-forms
    dofs_1 = _get_side_dofs(elements, e1, s1, form_order, highest_order)
    dofs_2 = _get_side_dofs(elements, e2, s2, form_order, highest_order)

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
    mesh: Mesh,
    element: int,
    form_order: UnknownFormOrder,
) -> list[Constraint]:
    """Generate constraints for continuity within the element.

    Parameters
    ----------
    mesh : Mesh
        Mesh in which the element is in.

    element : int
        Index of element to deal with.

    form_order : UnknownFormOrder
        Order of the form to use.

    Returns
    -------
    list of Constraint
        Constraints which enforce the continuity within the element.

    """
    children = mesh.get_element_children(element)
    if children is None:
        # Leaf has no need for constraints
        return list()

    c_bl, c_br, c_tr, c_tl = children

    # Inner constraints
    child_constraints: list[Constraint] = sum(
        (connect_element_inner(mesh, c, form_order) for c in children), start=list()
    )

    # Edge connections between child elements
    edge_constraints = (
        connect_edge_based(
            mesh,
            c_bl,
            ElementSide.SIDE_RIGHT,
            c_br,
            ElementSide.SIDE_LEFT,
            form_order,
        )
        + connect_edge_based(
            mesh,
            c_br,
            ElementSide.SIDE_TOP,
            c_tr,
            ElementSide.SIDE_BOTTOM,
            form_order,
        )
        + connect_edge_based(
            mesh,
            c_tr,
            ElementSide.SIDE_LEFT,
            c_tl,
            ElementSide.SIDE_RIGHT,
            form_order,
        )
        + connect_edge_based(
            mesh,
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
            mesh,
            (c_bl, ElementSide.SIDE_TOP),
            (c_br, ElementSide.SIDE_LEFT),
            (c_tr, ElementSide.SIDE_BOTTOM),
            (c_tl, ElementSide.SIDE_RIGHT),
        )

    return child_constraints + edge_constraints + corner_constraint


def connect_elements(
    form_specs: ElementFormSpecification, mesh: Mesh
) -> list[Constraint]:
    """Generate constraints for all elements and unknowns.

    Parameters
    ----------
    form_specs : ElementFormSpecification
        Orders of unknown forms defined for all elements.

    mesh : Mesh
        Mesh with primal and dual topology of root elements.

    Returns
    -------
    list of Constraint
        List with constraints which enforce continuity between degrees of freedom
        for unknown forms defined between all elements.
    """
    has_0 = any(form == UnknownFormOrder.FORM_ORDER_0 for form in form_specs.orders)
    has_1 = any(form == UnknownFormOrder.FORM_ORDER_1 for form in form_specs.orders)

    intra_element_0: list[Constraint] = list()  # for 0-forms
    intra_element_1: list[Constraint] = list()  # for 1-forms

    # Generate all intra-element constraints
    for surf_index in range(mesh.primal.n_surfaces):
        if has_0:
            intra_element_0 += connect_element_inner(
                mesh, surf_index, UnknownFormOrder.FORM_ORDER_0
            )
        if has_1:
            intra_element_1 += connect_element_inner(
                mesh, surf_index, UnknownFormOrder.FORM_ORDER_1
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

        surf_1 = mesh.primal.get_surface(idx1)
        surf_2 = mesh.primal.get_surface(idx2)

        side_1 = find_surface_boundary_id_line(surf_1, edge_index)
        side_2 = find_surface_boundary_id_line(surf_2, edge_index)

        if has_0:
            inter_edge_0 += connect_edge_based(
                mesh,
                idx1.index,
                side_1,
                idx2.index,
                side_2,
                UnknownFormOrder.FORM_ORDER_0,
            )

        if has_1:
            inter_edge_1 += connect_edge_based(
                mesh,
                idx1.index,
                side_1,
                idx2.index,
                side_2,
                UnknownFormOrder.FORM_ORDER_1,
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
                mesh,
                *(
                    (ie, _find_surface_boundary_id_node(mesh, ie, node_index))
                    for ie in element_indices
                ),
            )

    # Combine all of these
    combined_0 = intra_element_0 + inter_edge_0 + inter_corner_0
    combined_1 = intra_element_1 + inter_edge_1

    # Now make adjustments as needed to correctly offset
    real_constraints: list[Constraint] = list()

    # First does not need offsets, since there is nothing before it
    for i_form, form in enumerate(form_specs.orders):
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
                            ec.i_e,
                            ec.dofs
                            + form_specs.form_offset(
                                i_form, *mesh.get_leaf_orders(ec.i_e)
                            ),
                            ec.coeffs,
                        )
                        for ec in constraint.element_constraints
                    ),
                )
                for constraint in base
            ]
        else:
            real_constraints += base

    return real_constraints


def add_system_constraints(
    system: KFormSystem,
    mesh: Mesh,
    basis_cache: FemCache,
    constrained_forms: Sequence[tuple[float, KFormUnknown]],
    boundary_conditions: Sequence[BoundaryCondition2DSteady],
    leaf_indices: Sequence[int],
    element_offset: npt.NDArray[np.uint32],
    linear_vectors: Sequence[npt.NDArray[np.float64]] | None,
) -> tuple[sp.csr_array | None, npt.NDArray[np.float64]]:
    """Compute constraints for the system and vectors with weak boundary conditions."""
    constrained_form_constaints: dict[KFormUnknown, Constraint] = dict()
    form_specs = system.unknown_forms
    for k, form in constrained_forms:
        i_unknown = system.unknown_forms.index(form)
        constrained_form_constaints[form] = Constraint(
            k,
            *(
                ElementConstraint(
                    i,
                    form_specs.form_offset(i_unknown, *orders)
                    + np.arange(
                        form_specs.form_size(i_unknown, *orders),
                        dtype=np.uint32,
                    ),
                    np.ones(form_specs.form_size(i_unknown, *orders)),
                )
                for (i, orders) in (
                    (i, mesh.get_leaf_orders(leaf_idx))
                    for i, leaf_idx in enumerate(leaf_indices)
                )
            ),
        )

    if boundary_conditions is None:
        boundary_conditions = list()

    strong_bc_constraints, weak_bc_constraints = mesh_boundary_conditions(
        [eq.right for eq in system.equations],
        form_specs,
        mesh,
        [
            [bc for bc in boundary_conditions if bc.form == eq.weight.base_form]
            for eq in system.equations
        ],
        basis_cache,
    )

    continuity_constraints = connect_elements(form_specs, mesh)

    constraint_rows: list[npt.NDArray[np.uint32]] = list()
    constraint_cols: list[npt.NDArray[np.uint32]] = list()
    constraint_coef: list[npt.NDArray[np.float64]] = list()
    constraint_vals: list[float] = list()
    # Continuity constraints
    ic = 0
    for constraint in continuity_constraints:
        constraint_vals.append(constraint.rhs)
        # print(f"Continuity constraint {ic=}:")
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        #     print(ec)
        # print("")
        ic += 1

    # Form constraining
    for form in constrained_form_constaints:
        constraint = constrained_form_constaints[form]
        constraint_vals.append(constraint.rhs)
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        ic += 1

    # Strong BC constraints
    for ec in strong_bc_constraints:
        offset = int(element_offset[ec.i_e])
        for ci, cv in zip(ec.dofs, ec.coeffs, strict=True):
            constraint_rows.append(np.array([ic]))
            constraint_cols.append(np.array([ci + offset]))
            constraint_coef.append(np.array([1.0]))
            constraint_vals.append(float(cv))

            ic += 1

    # Weak BC constraints/additions
    if linear_vectors is not None:
        for ec in weak_bc_constraints:
            linear_vectors[ec.i_e][ec.dofs] += ec.coeffs

    if constraint_coef:
        lagrange_mat = sp.csr_array(
            (
                np.concatenate(constraint_coef),
                (
                    np.concatenate(constraint_rows, dtype=np.intp),
                    np.concatenate(constraint_cols, dtype=np.intp),
                ),
            )
        )
        lagrange_mat.resize((ic, element_offset[-1]))
        lagrange_vec = np.array(constraint_vals, np.float64)
    else:
        lagrange_mat = None
        lagrange_vec = np.zeros(0, np.float64)
    return lagrange_mat, lagrange_vec
