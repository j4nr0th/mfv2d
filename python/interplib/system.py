"""Functionality related to creating a full system of equations."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# from matplotlib import pyplot as plt  # TODO: remove
from scipy import sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms as kform
from interplib._interp import Polynomial1D, Spline1D
from interplib.mimetic.mimetic1d import Mesh1D, element_system


def solve_system_1d(
    system: kform.KFormSystem,
    mesh: Mesh1D,
    continuous: Sequence[kform.KFormUnknown],
    bcs_left: kform.BoundaryCondition1D | None = None,
    bcs_right: kform.BoundaryCondition1D | None = None,
) -> dict[kform.KFormUnknown, Spline1D]:
    """Solve the system on the specified mesh."""
    # Check that inputs make sense.
    for primal in system.unknown_forms:
        if primal.order > 1:
            raise ValueError(
                f"Can not solve the system on a 1D mesh, as it contains a {primal.order}"
                "-form."
            )
    # Check that the boundary conditions make sense
    if bcs_left is not None:
        if isinstance(bcs_left, kform.BoundaryCondition1DStrong):
            for form in bcs_left.forms:
                if form not in system.unknown_forms:
                    raise ValueError(
                        f"Left boundary condition uses a form {form}, which is not in any"
                        f" of the equations (which have forms {system.unknown_forms})."
                    )
        elif isinstance(bcs_left, kform.BoundaryCondition1DWeak):
            if bcs_left.form not in system.weak_forms:
                raise ValueError(
                    f"Form in the left weak condition ({bcs_left.form}) does not appear"
                    f" in the system {system.weak_forms}."
                )

    if bcs_right is not None:
        if isinstance(bcs_right, kform.BoundaryCondition1DStrong):
            for form in bcs_right.forms:
                if form not in system.unknown_forms:
                    raise ValueError(
                        f"Right boundary condition uses a form {form}, which is not in"
                        " any of the equations (which have forms"
                        f" {system.unknown_forms})."
                    )
        elif isinstance(bcs_right, kform.BoundaryCondition1DWeak):
            if bcs_right.form not in system.weak_forms:
                raise ValueError(
                    f"Form in the right weak condition ({bcs_right.form}) does not"
                    f" appear in the system {system.weak_forms}."
                )

    cont_indices: list[int] = []
    # d_cont_indices: list[int] = []
    for form in continuous:
        try:
            cont_indices.append(system.unknown_forms.index(form))
        except ValueError:
            raise ValueError(
                f"Can not enforce continuity on {form}, as it is not a form in the system"
                " of equations."
            ) from None

    # Obtain system information
    n_elem = mesh.primal.n_lines
    sizes_dual, sizes_primal = system.shape_1d(mesh.element_orders)
    offset_dual, offset_primal = system.offsets_1d(mesh.element_orders)
    element_offset = np.pad(np.cumsum(sizes_primal), (1, 0))
    dual_element_offset = np.pad(np.cumsum(sizes_dual), (1, 0))

    # Make element matrices and vectors
    element_outputs = tuple(
        element_system(system, mesh.get_element(ie)) for ie in range(n_elem)
    )
    element_matrix: list[npt.NDArray[np.float64]] = [e[0] for e in element_outputs]
    element_vectors: list[npt.NDArray[np.float64]] = [e[1] for e in element_outputs]

    # Sanity check
    # Apply lagrange multipliers for continuity
    mat_vals: list[int | float] = []
    mat_rows: list[int] = []
    mat_cols: list[int] = []

    lagrange_idx = element_offset[-1]  # current_h
    # Loop over dual elements
    # NOTE: assuming you can make extending lists and incrementing the index atomic, this
    # loop can be parallelized.
    for ie in range(n_elem + 1):
        dual = mesh.get_dual(ie)
        if not dual.begin or not dual.end:
            # This is a boundary of some sort
            continue
        left_element_idx = dual.begin.index
        right_element_idx = dual.end.index

        offset_left = element_offset[left_element_idx]
        offset_right = element_offset[right_element_idx]
        d_offset_left = dual_element_offset[left_element_idx]
        d_offset_right = dual_element_offset[right_element_idx]
        # For each variable which must be continuous, get locations in left and right
        for var_idx in cont_indices:
            # Left one is the first DoF of that variable
            left_var_offset = offset_primal[var_idx][left_element_idx]
            d_left_var_offset = offset_dual[var_idx][left_element_idx]
            # Right one is the last DoF of that variable
            right_var_offset = offset_primal[var_idx + 1][right_element_idx] - 1
            d_right_var_offset = offset_dual[var_idx + 1][right_element_idx] - 1
            mat_vals.extend((+1, -1, +1, -1))
            mat_rows.extend(
                (
                    lagrange_idx,
                    lagrange_idx,
                    d_offset_left + d_left_var_offset,
                    d_offset_right + d_right_var_offset,
                )
            )
            mat_cols.extend(
                (
                    offset_left + left_var_offset,
                    offset_right + right_var_offset,
                    lagrange_idx,
                    lagrange_idx,
                )
            )
        lagrange_idx += 1
    element_vectors.append(np.zeros(lagrange_idx - element_offset[-1]))  # current_h))

    # Apply lagrange multipliers for strong boundary conditions
    coeffs: list[float]
    dof_indices: list[int]
    if bcs_left is not None:
        if isinstance(bcs_left, kform.BoundaryCondition1DStrong):
            base_offset = element_offset[0]
            coeffs = []
            dof_indices = []
            for form in bcs_left.forms:
                coeffs.append(bcs_left.forms[form])
                form_index = system.unknown_forms.index(form)
                form_offset = offset_primal[form_index][0] + 0
                dof_indices.append(form_offset + base_offset)
            element_vectors.append(np.array([bcs_left.value]))
            mat_vals += coeffs + coeffs
            mat_cols += dof_indices + [lagrange_idx] * len(coeffs)
            mat_rows += [lagrange_idx] * len(coeffs) + dof_indices
            lagrange_idx += 1

        elif isinstance(bcs_left, kform.BoundaryCondition1DWeak):
            for ie, eq in enumerate(system.equations):
                for p_form in eq.weak_forms:
                    if bcs_left.form != p_form:
                        continue

                    form_offset = offset_dual[ie][0] + 0
                    element_vectors[0][form_offset] -= bcs_left.value

            form_index = system.unknown_forms.index(bcs_left.form)
            form_offset = offset_primal[form_index][0] + 0
            element_vectors[0][form_offset] -= bcs_left.value

        else:
            assert False

    if bcs_right is not None:
        if isinstance(bcs_right, kform.BoundaryCondition1DStrong):
            base_offset = element_offset[-2]
            coeffs = []
            dof_indices = []
            for form in bcs_right.forms:
                coeffs.append(bcs_right.forms[form])
                form_index = system.unknown_forms.index(form)
                form_offset = offset_primal[form_index + 1][-1] - 1
                dof_indices.append(form_offset + base_offset)
            element_vectors.append(np.array([bcs_right.value]))
            mat_vals += coeffs + coeffs
            mat_cols += dof_indices + [lagrange_idx] * len(coeffs)
            mat_rows += [lagrange_idx] * len(coeffs) + dof_indices
            del coeffs
            lagrange_idx += 1

        elif isinstance(bcs_right, kform.BoundaryCondition1DWeak):
            for ie, eq in enumerate(system.equations):
                for p_form in eq.weak_forms:
                    if bcs_right.form != p_form:
                        continue

                    form_offset = offset_dual[ie + 1][n_elem - 1] - 1
                    element_vectors[n_elem - 1][form_offset] += bcs_right.value

        else:
            assert False

    # this transposes it
    # indices1 = mat_rows + mat_cols
    # indices2 = mat_cols + mat_rows
    # mat_vals += mat_vals

    lagrange = sp.coo_array(
        (np.array(mat_vals), np.array((mat_rows, mat_cols), np.uint64)), dtype=np.float64
    )
    # Make the big matrix
    sys_mat = sp.block_diag(element_matrix)
    sys_mat.resize(lagrange.shape)

    # Create the system matrix and vector
    matrix = sp.csr_array(sys_mat + lagrange)
    vector = np.concatenate(element_vectors, dtype=np.float64)

    # with open("my_mat.dat", "w") as f_out:
    #     np.savetxt(f_out, matrix.toarray())
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.spy(matrix)
    # plt.show()

    # Solve the system
    # print(matrix.toarray())
    solution = sla.spsolve(matrix, vector)

    # Prepare to build up the 1D Splines
    build: dict[kform.KFormUnknown, list[npt.NDArray[np.float64]]] = {
        form: [] for form in system.unknown_forms
    }

    max_coeffs = np.max(mesh.element_orders) + 1
    # Loop over element
    for ie in range(n_elem):
        element = mesh.get_element(ie)
        # Extract element DoFs
        element_dofs = solution[element_offset[ie] : element_offset[ie + 1]]

        # Loop over each of the primal forms
        for idx, form in enumerate(system.unknown_forms):
            basis: tuple[Polynomial1D, ...]
            # Pick the basis
            if form.order == 0:
                basis = element.node_basis
            elif form.order == 1:
                basis = element.edge_basis
            else:
                assert False, "Should have been checked at the beginning."
            form_offset = offset_primal[idx][ie]
            form_offset_end = offset_primal[idx + 1][ie]
            form_dofs = element_dofs[form_offset:form_offset_end]
            # Make the element polynomial
            polynomial: Polynomial1D
            polynomial = sum(p * d for p, d in zip(basis, form_dofs))
            # Offset and scale it to domain [0, 1], the put it into the build dict
            k = polynomial.coefficients
            bad_len = max_coeffs - k.size
            if bad_len != 0:
                build[form].append(np.pad(k, (0, max_coeffs - k.size)))
            else:
                build[form].append(k)

    out: dict[kform.KFormUnknown, Spline1D] = dict()
    nodes = mesh.positions
    # Build the output splines
    for form in build:
        coefficients = np.array(build[form], np.float64)
        out[form] = Spline1D(nodes, coefficients)

    return out
