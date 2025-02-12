"""Functionality related to creating a full system of equations."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms as kform
from interplib._interp import compute_gll
from interplib._mimetic import GeoID, Surface
from interplib.mimetic.mimetic2d import Element2D, Mesh2D, element_system


def edge_dof_indices_from_line(
    elm: Element2D, s: Surface, i: int
) -> npt.NDArray[np.uint32]:
    """Find degree of freedom indices based on the line."""
    if s[0].index == i:
        return elm.boundary_edge_bottom
    if s[1].index == i:
        return elm.boundary_edge_right
    if s[2].index == i:
        return elm.boundary_edge_top
    if s[3].index == i:
        return elm.boundary_edge_left
    raise ValueError("Line is not in the element.")


def node_dof_indices_from_line(
    elm: Element2D, s: Surface, i: int
) -> npt.NDArray[np.uint32]:
    """Find degree of freedom indices based on the line."""
    if s[0].index == i:
        return elm.boundary_nodes_bottom
    if s[1].index == i:
        return elm.boundary_nodes_right
    if s[2].index == i:
        return elm.boundary_nodes_top
    if s[3].index == i:
        return elm.boundary_nodes_left
    raise ValueError("Line is not in the element.")


def solve_system_2d(
    system: kform.KFormSystem,
    mesh: Mesh2D,
    rec_order: int,
    boundaray_conditions: Sequence[kform.BoundaryCondition2DStrong] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[kform.KFormUnknown, npt.NDArray[np.float64]],
]:
    """Solve the system on the specified mesh.

    Parameters
    ----------
    system : kforms.KFormSystem
        System of equations to solve.
    mesh : Mesh2D
        Mesh on which to solve the system on.
    rec_order : int
        Order of reconstruction returned.

    Returns
    -------
    x : array
        Array of x positions where the reconstructed values were computed.
    y : array
        Array of y positions where the reconstructed values were computed.
    reconstructions : dict of kforms.KFormUnknown to array
        Reconstructed solution for unknowns. The number of points on the element
        where these reconstructions are computed is equal to ``(rec_order + 1) ** 2``.
    """
    # Check that inputs make sense.
    for primal in system.unknown_forms:
        if primal.order > 2:
            raise ValueError(
                f"Can not solve the system on a 2D mesh, as it contains a {primal.order}"
                "-form."
            )
    # Check boundary conditions are sensible
    if boundaray_conditions is not None:
        for bc in boundaray_conditions:
            if bc.form not in system.unknown_forms:
                raise ValueError(
                    f"Boundary conditions specify form {bc.form}, which is not in the"
                    " system"
                )
            if np.any(bc.indices >= mesh.primal.n_lines):
                raise ValueError(
                    f"Boundary condition on {bc.form} specifies lines which are"
                    " outside not in the mesh (highest index specified was "
                    f"{np.max(bc.indices)}, but mesh has {mesh.primal.n_points}"
                    " lines)."
                )

    cont_indices_edges: list[int] = []
    cont_indices_nodes: list[int] = []
    for form in system.unknown_forms:
        if form.order == 2:
            continue
        idx = system.unknown_forms.index(form)
        if form.order == 1:
            cont_indices_edges.append(idx)
        elif form.order == 0:
            cont_indices_nodes.append(idx)
        else:
            assert False

    # Obtain system information
    n_elem = mesh.n_elements
    element_orders = np.full(n_elem, mesh.order)
    sizes_weight, sizes_unknown = system.shape_2d(element_orders)
    offset_weight, offset_unknown = system.offsets_2d(element_orders)
    del sizes_weight, offset_weight
    unknown_element_offset = np.pad(np.cumsum(sizes_unknown), (1, 0))
    # weight_element_offset = np.pad(np.cumsum(sizes_weight), (1, 0))

    # Make element matrices and vectors
    elements = tuple(mesh.get_element(ie) for ie in range(n_elem))
    element_outputs = tuple(element_system(system, e) for e in elements)
    element_matrix: list[npt.NDArray[np.float64]] = [e[0] for e in element_outputs]
    element_vectors: list[npt.NDArray[np.float64]] = [e[1] for e in element_outputs]

    # Sanity check
    # Apply lagrange multipliers for continuity

    entries: list[
        tuple[
            npt.NDArray[np.integer],
            npt.NDArray[np.integer],
            npt.NDArray[np.integer | np.floating],
        ]
    ] = list()

    lagrange_idx = unknown_element_offset[-1]  # current_h
    # Loop over dual lines
    # NOTE: assuming you can make extending lists and incrementing the index atomic, this
    # loop can be parallelized.

    # Continuity of 1-forms
    if cont_indices_edges:
        for il in range(mesh.dual.n_lines):
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end
            if not idx_self or not idx_neighbour:
                continue

            # print(f"Linking {idx_self.index} to {idx_neighbour.index}.")

            offset_self = unknown_element_offset[idx_self.index]
            offset_other = unknown_element_offset[idx_neighbour.index]
            # d_offset_self = weight_element_offset[idx_self.index]
            # d_offset_other = weight_element_offset[idx_neighbour.index]
            # For each variable which must be continuous, get locations in left and right
            for var_idx in cont_indices_edges:
                # Left one is from the first DoF of that variable
                self_var_offset = offset_unknown[var_idx][idx_self.index]
                # d_self_var_offset = offset_weight[var_idx][idx_self.index]
                # Right one is from the first DoF of that variable
                other_var_offset = offset_unknown[var_idx][idx_neighbour.index]
                # d_other_var_offset = offset_weight[var_idx][idx_neighbour.index]

                # row_off_self = d_offset_self + d_self_var_offset
                # row_off_other = d_offset_other + d_other_var_offset
                col_off_self = offset_self + self_var_offset
                col_off_other = offset_other + other_var_offset

                s_other = mesh.primal.get_surface(idx_neighbour)
                e_other = elements[idx_neighbour.index]

                s_self = mesh.primal.get_surface(idx_self)
                e_self = elements[idx_self.index]

                dofs_other = np.flip(edge_dof_indices_from_line(e_other, s_other, il))
                ds = edge_dof_indices_from_line(e_self, s_self, il)
                assert dofs_other.size == ds.size
                n_lag = ds.size

                # If this is atomic, this works in parallel
                begin_idx = lagrange_idx
                lagrange_idx += n_lag
                # End atomic

                entries.append(
                    (
                        np.concatenate((col_off_self + ds, col_off_other + dofs_other)),
                        np.tile(np.arange(n_lag) + begin_idx, 2),
                        np.concatenate((np.ones(n_lag), -np.ones(n_lag))),
                    )
                )

    # Continuity of 0-forms on the non-corner DoFs
    if cont_indices_nodes and mesh.order > 1:
        for il in range(mesh.dual.n_lines):
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end

            if not idx_neighbour or not idx_self:
                continue

            # print(f"Linking {idx_self.index} to {idx_neighbour.index}.")

            offset_self = unknown_element_offset[idx_self.index]
            offset_other = unknown_element_offset[idx_neighbour.index]
            # For each variable which must be continuous, get locations in left and right
            for var_idx in cont_indices_nodes:
                # Left one is from the first DoF of that variable
                self_var_offset = offset_unknown[var_idx][idx_self.index]
                # Right one is from the first DoF of that variable
                other_var_offset = offset_unknown[var_idx][idx_neighbour.index]

                col_off_self = offset_self + self_var_offset
                col_off_other = offset_other + other_var_offset

                s_other = mesh.primal.get_surface(idx_neighbour)
                e_other = elements[idx_neighbour.index]

                s_self = mesh.primal.get_surface(idx_self)
                e_self = elements[idx_self.index]

                dofs_other = np.flip(
                    node_dof_indices_from_line(e_other, s_other, il)[1:-1]
                )
                ds = node_dof_indices_from_line(e_self, s_self, il)[1:-1]
                assert dofs_other.size == ds.size
                n_lag = ds.size

                # If this is atomic, this works in parallel
                begin_idx = lagrange_idx
                lagrange_idx += n_lag
                # End atomic

                entries.append(
                    (
                        np.concatenate((col_off_self + ds, col_off_other + dofs_other)),
                        np.tile(np.arange(n_lag) + begin_idx, 2),
                        np.concatenate((np.ones(n_lag), -np.ones(n_lag))),
                    )
                )

    # Continuity of 0-forms on the corner DoFs
    if cont_indices_nodes:
        for i_surf in range(mesh.dual.n_surfaces):
            dual_surface = mesh.dual.get_surface(i_surf + 1)
            closed = True
            valid: list[int] = list()
            for i_ln in range(len(dual_surface)):
                id_line = dual_surface[i_ln]
                dual_line = mesh.dual.get_line(id_line)
                idx_neighbour = dual_line.begin
                idx_self = dual_line.end
                if not idx_neighbour or not idx_self:
                    closed = False
                else:
                    valid.append(i_ln)

            if closed:
                valid.pop()

            for i_ln in valid:
                id_line = dual_surface[i_ln]
                dual_line = mesh.dual.get_line(id_line)
                idx_neighbour = dual_line.begin
                idx_self = dual_line.end

                if not idx_neighbour or not idx_self:
                    continue

                offset_self = unknown_element_offset[idx_self.index]
                offset_other = unknown_element_offset[idx_neighbour.index]

                for var_idx in cont_indices_nodes:
                    # Left one is from the first DoF of that variable
                    self_var_offset = offset_unknown[var_idx][idx_self.index]
                    # Right one is from the first DoF of that variable
                    other_var_offset = offset_unknown[var_idx][idx_neighbour.index]

                    col_off_self = offset_self + self_var_offset
                    col_off_other = offset_other + other_var_offset

                    s_other = mesh.primal.get_surface(idx_neighbour)
                    e_other = elements[idx_neighbour.index]

                    s_self = mesh.primal.get_surface(idx_self)
                    e_self = elements[idx_self.index]

                    dofs_other = node_dof_indices_from_line(
                        e_other, s_other, id_line.index
                    )[-1]
                    ds = node_dof_indices_from_line(e_self, s_self, id_line.index)[0]

                    # If this is atomic, this works in parallel
                    begin_idx = lagrange_idx
                    lagrange_idx += 1
                    # End atomic

                    entries.append(
                        (
                            np.array((col_off_self + ds, col_off_other + dofs_other)),
                            np.array((begin_idx, begin_idx)),
                            np.array((1, -1)),
                        )
                    )

    num_lagrange_coeffs = lagrange_idx - unknown_element_offset[-1]
    if num_lagrange_coeffs > 0:
        element_vectors.append(np.zeros(num_lagrange_coeffs))

    # Strong boundary conditions
    if boundaray_conditions is not None:
        bc_entries: list[
            tuple[
                npt.NDArray[np.integer],
                npt.NDArray[np.integer],
                npt.NDArray[np.integer | np.floating],
            ]
        ] = list()
        bc_rhs: list[npt.NDArray[np.float64]] = []
        set_nodes: set[int] = set()
        for bc in boundaray_conditions:
            self_var_offset = offset_unknown[system.unknown_forms.index(bc.form)]

            for idx in bc.indices:
                dof_offsets: npt.NDArray[np.integer] | None = None
                surf_id: GeoID | None = None
                x0: float
                x1: float
                y0: float
                y1: float

                dual_line = mesh.dual.get_line(idx + 1)
                if dual_line.begin and dual_line.end:
                    raise ValueError(
                        f"Boundary condition for {bc.form} was specified for"
                        f" line {idx}, which is not on the boundary."
                    )
                surf_id = dual_line.begin if dual_line.begin else dual_line.end
                primal_surface = mesh.primal.get_surface(surf_id)
                assert len(primal_surface) == 4

                for i_side in range(4):
                    if primal_surface[i_side].index == idx:
                        break
                assert i_side != 4
                elm = elements[surf_id.index]
                if i_side == 0:
                    if bc.form.order == 0:
                        dof_offsets = elm.boundary_nodes_bottom
                    else:
                        dof_offsets = elm.boundary_edge_bottom
                elif i_side == 1:
                    if bc.form.order == 0:
                        dof_offsets = elm.boundary_nodes_right
                    else:
                        dof_offsets = elm.boundary_edge_right
                elif i_side == 2:
                    if bc.form.order == 0:
                        dof_offsets = elm.boundary_nodes_top
                    else:
                        dof_offsets = elm.boundary_edge_top
                elif i_side == 3:
                    if bc.form.order == 0:
                        dof_offsets = elm.boundary_nodes_left
                    else:
                        dof_offsets = elm.boundary_edge_left
                else:
                    assert False
                assert dof_offsets is not None
                assert surf_id is not None

                dof_offsets += np.astype(
                    self_var_offset[surf_id.index]
                    + unknown_element_offset[surf_id.index],
                    np.uint32,
                )
                assert dof_offsets is not None

                primal_line = mesh.primal.get_line(primal_surface[i_side])
                x0, y0 = mesh.positions[primal_line.begin.index, :]
                x1, y1 = mesh.positions[primal_line.end.index, :]

                comp_nodes = elm.nodes_1d
                xv = (x1 + x0) / 2 + (x1 - x0) / 2 * comp_nodes
                yv = (y1 + y0) / 2 + (y1 - y0) / 2 * comp_nodes

                vals = np.empty_like(dof_offsets, np.float64)
                if bc.form.order == 0:
                    vals[:] = bc.func(xv, yv)

                    if primal_line.begin.index in set_nodes:
                        vals = vals[1:]
                        dof_offsets = dof_offsets[1:]
                    else:
                        set_nodes.add(primal_line.begin.index)

                    if primal_line.end.index in set_nodes:
                        vals = vals[:-1]
                        dof_offsets = dof_offsets[:-1]
                    else:
                        set_nodes.add(primal_line.end.index)

                elif bc.form.order == 1:
                    # TODO: this might be more efficiently done as some sort of projection
                    lnds, wnds = compute_gll(2 * elm.order)
                    for i in range(bc.form.order):
                        xc = (xv[i + 1] + xv[i]) / 2 + (xv[i + 1] - xv[i]) / 2 * lnds
                        yc = (yv[i + 1] + yv[i]) / 2 + (yv[i + 1] - yv[i]) / 2 * lnds
                        dx = (xv[i + 1] - xv[i]) / 2
                        dy = (yv[i + 1] - yv[i]) / 2
                        if i_side == 0:
                            normal = np.array((-dy, dx))
                        elif i_side == 1:
                            normal = np.array((dy, -dx))
                        elif i_side == 2:
                            normal = np.array((dy, -dx))
                        elif i_side == 3:
                            normal = np.array((-dy, dx))
                        else:
                            assert False
                        fvals = bc.func(xc, yc)
                        fvals = fvals[..., 0] * normal[0] + fvals[..., 1] * normal[1]
                        vals[i] = np.sum(fvals * wnds)
                else:
                    assert False

                n_lag = vals.size
                assert n_lag == dof_offsets.size
                if n_lag:
                    rows = np.arange(n_lag) + lagrange_idx
                    lagrange_idx += n_lag

                    bc_entries.append((rows, dof_offsets, np.ones_like(dof_offsets)))
                    bc_rhs.append(vals)

        entries.extend(bc_entries)
        element_vectors.extend(bc_rhs)

    num_lagrange_coeffs = lagrange_idx - unknown_element_offset[-1]
    sys_mat = sp.block_diag(element_matrix)
    if entries:
        # element_vectors.append(np.zeros(num_lagrange_coeffs))

        # Transpose
        l1 = [e[1] for e in entries]
        l2 = [e[0] for e in entries]
        l3 = [e[2] for e in entries]
        mat_rows = np.concatenate(l1 + l2)
        mat_cols = np.concatenate(l2 + l1)
        mat_vals = np.concatenate(l3 + l3)

        lagrange = sp.coo_array((mat_vals, (mat_rows, mat_cols)), dtype=np.float64)
        # Make the big matrix
        sys_mat.resize(lagrange.shape)

        # Create the system matrix and vector
        matrix = sp.csr_array(sys_mat + lagrange)
    else:
        matrix = sp.csr_array(sys_mat)
    vector = np.concatenate(element_vectors, dtype=np.float64)

    # from matplotlib import pyplot as plt

    # plt.spy(matrix)
    # plt.show()
    # with open("my_mat.dat", "w") as f_out:
    #     np.savetxt(f_out, sys_mat.toarray())
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.spy(matrix)
    # plt.show()

    # Solve the system
    # print(matrix.toarray())
    solution = sla.spsolve(matrix, vector)

    recon_nodes_1d = np.linspace(-1, +1, rec_order + 1)

    xvals: list[npt.NDArray[np.float64]] = list()
    yvals: list[npt.NDArray[np.float64]] = list()

    # Prepare to build up the 1D Splines
    build: dict[kform.KFormUnknown, list[npt.NDArray[np.float64]]] = {
        form: [] for form in system.unknown_forms
    }

    # Loop over element
    for ie, elm in enumerate(elements):
        # Extract element DoFs
        element_dofs = solution[
            unknown_element_offset[ie] : unknown_element_offset[ie + 1]
        ]

        ex = elm.poly_x(recon_nodes_1d[None, :], recon_nodes_1d[:, None])
        ey = elm.poly_y(recon_nodes_1d[None, :], recon_nodes_1d[:, None])

        xvals.append(ex.flatten())
        yvals.append(ey.flatten())

        # Loop over each of the primal forms
        for idx, form in enumerate(system.unknown_forms):
            form_offset = offset_unknown[idx][ie]
            form_offset_end = offset_unknown[idx + 1][ie]
            form_dofs = element_dofs[form_offset:form_offset_end]

            # Reconstruct unknown
            v = elm.reconstruct(
                form.order, form_dofs, recon_nodes_1d[None, :], recon_nodes_1d[:, None]
            )
            shape = (-1, 2) if form.order == 1 else (-1,)
            build[form].append(np.reshape(v, shape))

    out: dict[kform.KFormUnknown, npt.NDArray[np.float64]] = dict()

    # Build the output splines
    for form in build:
        out[form] = np.stack(build[form], dtype=np.float64)

    return (
        np.stack(xvals, dtype=np.float64),
        np.stack(yvals, dtype=np.float64),
        out,
    )
