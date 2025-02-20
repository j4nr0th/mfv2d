"""Functionality related to creating a full system of equations."""

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms as kform
from interplib._eval import compute_element_matrices
from interplib._interp import compute_gll
from interplib._mimetic import GeoID, Surface
from interplib.kforms.eval import _ctranslate, translate_equation
from interplib.mimetic.mimetic2d import BasisCache, Element2D, Mesh2D, element_system


def endpoints_from_line(
    elm: Element2D, s: Surface, i: int
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    """Get endpoints of the element boundary based on the line index.

    Returns
    -------
    int
        Direction of the normal. Positive means it contributes to gradient.
    (float, float)
        Beginning of the line corresponding to -1 on the reference element.

    (float, float)
        End of the line corresponding to +1 on the reference element.
    """
    if s[0].index == i:
        return (+1, elm.bottom_left, elm.bottom_right)
    if s[1].index == i:
        return (-1, elm.bottom_right, elm.top_right)
    if s[2].index == i:
        return (-1, elm.top_right, elm.top_left)
    if s[3].index == i:
        return (+1, elm.top_left, elm.bottom_left)
    raise ValueError("Line is not in the element.")


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
    # workers: int | None = None,
    # new_evaluation: bool = True,
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
    boundary_conditions: Sequence of kforms.BoundaryCondition2DStrong, optional
        Sequence of boundary conditions to be applied to the system.
    new_evaluation: bool, default: True
        Use newer evaluation based on a virtual stack machine. This can then be
        ported to C.

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
    strong_boundary_edges: dict[kform.KFormUnknown, list[npt.NDArray[np.uint64]]] = {}
    for primal in system.unknown_forms:
        if primal.order > 2:
            raise ValueError(
                f"Can not solve the system on a 2D mesh, as it contains a {primal.order}"
                "-form."
            )
        strong_boundary_edges[primal] = []

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
            strong_boundary_edges[bc.form].append(bc.indices)

    strong_indices = {
        form: np.concatenate(strong_boundary_edges[form])
        if len(strong_boundary_edges[form])
        else np.array([])
        for form in strong_boundary_edges
    }
    del strong_boundary_edges

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
    cache: dict[int, BasisCache] = dict()
    for e in elements:
        if e.order in cache:
            continue
        cache[e.order] = BasisCache(e.order, 3 * e.order)
    bytecodes = [translate_equation(eq.left, simplify=True) for eq in system.equations]

    t0 = perf_counter()
    element_outputs = tuple(
        element_system(system, e, cache[e.order], None) for e in elements
    )
    t1 = perf_counter()
    print(f"Element matrices old way: {t1 - t0} seconds.")
    codes = []
    for bite in bytecodes:
        row: list[list | None] = []
        for f in system.unknown_forms:
            if f in bite:
                row.append(_ctranslate(*bite[f]))
            else:
                row.append(None)
        codes.append(row)

    bl = np.array([e.bottom_left for e in elements])
    br = np.array([e.bottom_right for e in elements])
    tr = np.array([e.top_right for e in elements])
    tl = np.array([e.top_left for e in elements])
    orde = np.array([e.order for e in elements], np.uint32)
    t0 = perf_counter()
    second_matrices = compute_element_matrices(
        [f.order for f in system.unknown_forms],
        codes,
        bl,
        br,
        tr,
        tl,
        orde,
        [cache[o].c_serialization for o in cache],
    )
    t1 = perf_counter()
    del bl, br, tr, tl, orde
    print(f"Element matrices new way: {t1 - t0} seconds.")

    # from interplib._eval import element_matrices
    element_matrix: list[npt.NDArray[np.float64]] = [e[0] for e in element_outputs]

    for e, mat1, mat2 in zip(elements, element_matrix, second_matrices, strict=True):
        # print(np.max(np.max(mat1 - mat2)))
        # ce = cache[e.order]
        # m0, m1, m2, i0, i1, i2 = element_matrices(
        #     e.bottom_left[0],
        #     e.bottom_right[0],
        #     e.top_right[0],
        #     e.top_left[0],
        #     e.bottom_left[1],
        #     e.bottom_right[1],
        #     e.top_right[1],
        #     e.top_left[1],
        #     ce.c_serialization,
        # )
        # assert np.allclose(m0, e.mass_matrix_node(ce))
        # assert np.allclose(m1, e.mass_matrix_edge(ce))
        # assert np.allclose(m2, e.mass_matrix_surface(ce))
        # assert np.allclose(i0, np.linalg.inv(e.mass_matrix_node(ce)))
        # assert np.allclose(i1, np.linalg.inv(e.mass_matrix_edge(ce)))
        # assert np.allclose(i2, np.linalg.inv(e.mass_matrix_surface(ce)))
        assert mat1.shape == mat2.shape
        assert np.allclose(mat2, element_system(system, e, cache[e.order], None)[0])
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

    from interplib.kforms.kform import KBoundaryProjection

    # Weak boundary conditions
    for eq in system.equations:
        rhs = eq.right
        for c, kp in rhs.pairs:
            if not isinstance(kp, KBoundaryProjection) or kp.func is None or c == 0:
                continue
            w_form = kp.weight.base_form
            edges = mesh.boundary_indices
            if w_form in strong_indices:
                edges = np.astype(
                    np.setdiff1d(edges, strong_indices[w_form]),  # type: ignore
                    dtype=np.int32,
                    copy=False,
                )
            if edges.size == 0:
                continue
            for edge in edges:
                dual_line = mesh.dual.get_line(edge + 1)
                primal_line = mesh.primal.get_line(edge + 1)
                if dual_line.begin:
                    id_surf = dual_line.begin
                elif dual_line.end:
                    id_surf = dual_line.end
                else:
                    assert False
                e = elements[id_surf.index]
                basis_cache = cache[e.order]
                primal_surface = mesh.primal.get_surface(id_surf)
                ndir, p0, p1 = endpoints_from_line(e, primal_surface, edge)
                dx = (p1[0] - p0[0]) / 2
                xv = (p1[0] + p0[0]) / 2 + dx * basis_cache.int_nodes_1d
                dy = (p1[1] - p0[1]) / 2
                yv = (p1[1] + p0[1]) / 2 + dy * basis_cache.int_nodes_1d
                f_vals = kp.func(xv, yv)
                if w_form.order == 0:
                    dofs = node_dof_indices_from_line(e, primal_surface, edge)
                    # Tangental integral of function with the 0 basis
                    basis = basis_cache.nodal_1d
                    f_vals = (
                        f_vals[..., 0] * dx + f_vals[..., 1] * dy
                    ) * basis_cache.int_weights_1d

                elif w_form.order:
                    dofs = edge_dof_indices_from_line(e, primal_surface, edge)
                    # Integral with the normal basis
                    basis = basis_cache.edge_1d
                    f_vals *= basis_cache.int_weights_1d * ndir  # * np.hypot(dx, dy)

                else:
                    assert False

                vals = c * np.sum(f_vals[..., None] * basis, axis=0)
                element_vectors[id_surf.index][dofs] += vals

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
            if not form.is_primal:
                mass: npt.NDArray[np.float64]
                if form.order == 0:
                    mass = elm.mass_matrix_surface(cache[elm.order])
                elif form.order == 1:
                    mass = elm.mass_matrix_edge(cache[elm.order])
                elif form.order == 2:
                    mass = elm.mass_matrix_node(cache[elm.order])
                else:
                    assert False
                form_dofs = np.linalg.solve(mass, form_dofs)
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
