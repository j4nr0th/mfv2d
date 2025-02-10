"""Functionality related to creating a full system of equations."""

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms as kform
from interplib._mimetic import Surface
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

    cont_indices_edges: list[int] = []
    cont_indices_nodes: list[int] = []
    # d_cont_indices: list[int] = []
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
            # primal_surface = mesh.primal.get_surface(ie + 1)
            # dofs_self_edge = (
            #     elm.boundary_edge_bottom,
            #     elm.boundary_edge_right,
            #     elm.boundary_edge_top,
            #     elm.boundary_edge_left,
            # )
            # dofs_self_nodes = (
            #     elm.boundary_nodes_bottom,
            #     elm.boundary_nodes_right,
            #     elm.boundary_nodes_top,
            #     elm.boundary_nodes_left,
            # )
            dual_line = mesh.dual.get_line(il + 1)
            idx_neighbour = dual_line.begin
            idx_self = dual_line.end

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

            for i_ln in range(len(dual_surface) - 1):
                id_line = dual_surface[i_ln]
                dual_line = mesh.dual.get_line(il)
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
                    )[0]
                    ds = node_dof_indices_from_line(e_self, s_self, id_line.index)[-1]

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
    sys_mat = sp.block_diag(element_matrix)
    if num_lagrange_coeffs > 0:
        element_vectors.append(np.zeros(num_lagrange_coeffs))

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

    # plt.spy(matrix.toarray())
    # plt.show()
    # with open("my_mat.dat", "w") as f_out:
    #     np.savetxt(f_out, matrix.toarray())
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

        xvals.append(ex)
        yvals.append(ey)

        # Loop over each of the primal forms
        for idx, form in enumerate(system.unknown_forms):
            form_offset = offset_unknown[idx][ie]
            form_offset_end = offset_unknown[idx + 1][ie]
            form_dofs = element_dofs[form_offset:form_offset_end]

            # Reconstruct unknown
            v = elm.reconstruct(
                form.order, form_dofs, recon_nodes_1d[None, :], recon_nodes_1d[:, None]
            )

            build[form].append(v)

    out: dict[kform.KFormUnknown, npt.NDArray[np.float64]] = dict()

    # Build the output splines
    for form in build:
        out[form] = np.concatenate(build[form], dtype=np.float64)

    return (
        np.concatenate(xvals, dtype=np.float64),
        np.concatenate(yvals, dtype=np.float64),
        out,
    )
