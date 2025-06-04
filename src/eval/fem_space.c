#include "fem_space.h"

MFV2D_INTERNAL eval_result_t fem_space_2d_create(const fem_space_1d_t *space_h, const fem_space_1d_t *space_v,
                                                 const quad_info_t *const quad, fem_space_2d_t **p_out,
                                                 const allocator_callbacks *allocator)
{

    const unsigned rows = space_h->n_pts;
    const unsigned cols = space_v->n_pts;

    fem_space_2d_t *const out = allocate(allocator, sizeof *out + sizeof *out->jacobian * rows * cols);
    if (!out)
        return EVAL_FAILED_ALLOC;
    out->space_2 = *space_v;
    out->space_1 = *space_h;

    const double x0 = quad->x0;
    const double x1 = quad->x1;
    const double x2 = quad->x2;
    const double x3 = quad->x3;
    const double y0 = quad->y0;
    const double y1 = quad->y1;
    const double y2 = quad->y2;
    const double y3 = quad->y3;

    for (unsigned row = 0; row < rows; ++row)
    {
        const double eta = space_v->pnts[row];
        const double dx_dxi = ((x1 - x0) * (1 - eta) + (x2 - x3) * (1 + eta)) / 4;
        const double dy_dxi = ((y1 - y0) * (1 - eta) + (y2 - y3) * (1 + eta)) / 4;
        for (unsigned col = 0; col < cols; ++col)
        {
            const double xi = space_h->pnts[col];
            const double dx_deta = ((x3 - x0) * (1 - xi) + (x2 - x1) * (1 + xi)) / 4;
            const double dy_deta = ((y3 - y0) * (1 - xi) + (y2 - y1) * (1 + xi)) / 4;
            const double determinant = dx_dxi * dy_deta - dx_deta * dy_dxi;
            out->jacobian[row * cols + col] =
                (jacobian_t){.j00 = dx_dxi, .j01 = dy_dxi, .j10 = dx_deta, .j11 = dy_deta, .det = determinant};
        }
    }

    *p_out = out;
    return EVAL_SUCCESS;
}

static double node_basis_value(const fem_space_2d_t *space, unsigned i_basis, unsigned j_basis, unsigned i_point,
                               unsigned j_point)
{
    if (ASSERT(i_point < space->space_1.n_pts, "Point %u is out of range for basis 1 with %u points", i_point,
               space->space_1.n_pts) ||
        ASSERT(j_point < space->space_2.n_pts, "Point %u is out of range for basis 2 with %u points", j_point,
               space->space_2.n_pts) ||
        ASSERT(i_basis <= space->space_1.order,
               "Basis %u is out of range for nodal basis set 1 with %u basis functions", i_basis,
               space->space_1.order + 1) ||
        ASSERT(j_basis <= space->space_2.order,
               "Basis %u is out of range for nodal basis set 2 with %u basis functions", j_basis,
               space->space_2.order + 1))
    {
        // Shit
        exit(EXIT_FAILURE);
        return 0;
    }

    const double *const basis_1 = space->space_1.node + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.node + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double edge_h_basis_value(const fem_space_2d_t *space, unsigned i_basis, unsigned j_basis, unsigned i_point,
                                 unsigned j_point)
{
    if (ASSERT(i_point < space->space_1.n_pts, "Point %u is out of range for basis 1 with %u points", i_point,
               space->space_1.n_pts) ||
        ASSERT(j_point < space->space_2.n_pts, "Point %u is out of range for basis 2 with %u points", j_point,
               space->space_2.n_pts) ||
        ASSERT(i_basis <= space->space_1.order,
               "Basis %u is out of range for nodal basis set 1 with %u basis functions", i_basis,
               space->space_1.order + 1) ||
        ASSERT(j_basis < space->space_2.order, "Basis %u is out of range for edge basis set 2 with %u basis functions",
               j_basis, space->space_2.order))
    {
        // Shit
        exit(EXIT_FAILURE);
        return 0;
    }

    const double *const basis_1 = space->space_1.node + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.edge + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double edge_v_basis_value(const fem_space_2d_t *space, unsigned i_basis, unsigned j_basis, unsigned i_point,
                                 unsigned j_point)
{
    if (ASSERT(i_point < space->space_1.n_pts, "Point %u is out of range for basis 1 with %u points", i_point,
               space->space_1.n_pts) ||
        ASSERT(j_point < space->space_2.n_pts, "Point %u is out of range for basis 2 with %u points", j_point,
               space->space_2.n_pts) ||
        ASSERT(i_basis < space->space_1.order, "Basis %u is out of range for edge basis set 1 with %u basis functions",
               i_basis, space->space_1.order) ||
        ASSERT(j_basis <= space->space_2.order,
               "Basis %u is out of range for nodal basis set 2 with %u basis functions", j_basis,
               space->space_2.order + 1))
    {
        // Shit
        exit(EXIT_FAILURE);
        return 0;
    }

    const double *const basis_1 = space->space_1.edge + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.node + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double surf_basis_value(const fem_space_2d_t *space, unsigned i_basis, unsigned j_basis, unsigned i_point,
                               unsigned j_point)
{
    if (ASSERT(i_point < space->space_1.n_pts, "Point is out of range for basis 1") ||
        ASSERT(j_point < space->space_2.n_pts, "Point is out of range for basis 2") ||
        ASSERT(i_basis < space->space_1.order, "Basis is out of range for basis 1") ||
        ASSERT(j_basis < space->space_2.order, "Basis is out of range for basis 2"))
        return 0;

    const double *const basis_1 = space->space_1.edge + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.edge + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

MFV2D_INTERNAL eval_result_t compute_mass_matrix_node(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                      const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = (space_h->order + 1) * (space_v->order + 1);
    const unsigned cols = (space_h->order + 1) * (space_v->order + 1);

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return EVAL_FAILED_ALLOC;

    for (unsigned i_weight = 0; i_weight < space_v->order + 1; ++i_weight)
    {
        for (unsigned j_weight = 0; j_weight < space_h->order + 1; ++j_weight)
        {
            const unsigned idx_weight = j_weight + i_weight * (space_h->order + 1);
            for (unsigned i_basis = 0; i_basis < space_v->order + 1; ++i_basis)
            {
                for (unsigned j_basis = 0; j_basis < space_h->order + 1; ++j_basis)
                {
                    const unsigned idx_basis = j_basis + i_basis * (space_h->order + 1);

                    double v = 0;
                    for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
                    {
                        for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                        {
                            v += node_basis_value(space, i_basis, j_basis, i_point, j_point) *
                                 node_basis_value(space, i_weight, j_weight, i_point, j_point) *
                                 space->jacobian[j_point + i_point * space_h->n_pts].det * space_h->wgts[j_point] *
                                 space_v->wgts[i_point];
                        }
                    }
                    // Exploit the symmetry of the matrix.
                    out.data[idx_basis * rows + idx_weight] = out.data[idx_weight * cols + idx_basis] = v;
                }
            }
        }
    }

    *p_out = out;
    return EVAL_SUCCESS;
}

MFV2D_INTERNAL eval_result_t compute_mass_matrix_edge(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                      const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = space_h->order * (space_v->order + 1) + (space_h->order + 1) * space_v->order;
    const unsigned cols = space_h->order * (space_v->order + 1) + (space_h->order + 1) * space_v->order;

    const size_t mem_size = sizeof(double) * rows * cols;
    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, mem_size)};
    if (!out.data)
        return EVAL_FAILED_ALLOC;

    // Edge basis are a pain in the ass to compute, since they are actually representing
    // two different vector components. As such, matrix has 4 blocks:
    // - vertical inner product with vertical (block 00),
    // - vertical inner product with horizontal (block 01),
    // - horizontal inner product with vertical (block 10),
    // - horizontal inner product with horizontal (block 11).
    //
    // The symmetry can be exploited as such:
    // - Blocks 00 and 11 are symmetrical,
    // - Blocks 01 and 10 are not symmetrical but are transposes of each other.

    // k00 = (j11**2 + j10**2) / det
    // k11 = (j01**2 + j00**2) / det
    // k01 = (j01 * j11 + j00 * j10) / det

    const unsigned n_v_basis = (space_v->order + 1) * space_h->order;
    const unsigned n_h_basis = space_v->order * (space_h->order + 1);

    // Block 11
    for (unsigned i_weight = 0; i_weight < space_v->order; ++i_weight)
        for (unsigned j_weight = 0; j_weight < space_h->order + 1; ++j_weight)
            for (unsigned i_basis = 0; i_basis < space_v->order; ++i_basis)
                for (unsigned j_basis = 0; j_basis < space_h->order + 1; ++j_basis)
                {
                    const unsigned idx_weight = j_weight + i_weight * (space_h->order + 1);
                    const unsigned idx_basis = j_basis + i_basis * (space_h->order + 1);
                    double v = 0;
                    for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
                    {
                        for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                        {
                            const double val_basis = edge_v_basis_value(space, i_basis, j_basis, i_point, j_point);
                            const double val_weight = edge_v_basis_value(space, i_weight, j_weight, i_point, j_point);
                            const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                            const double jac_term = (jac->j00 * jac->j00 + jac->j01 * jac->j01) / jac->det;
                            v += val_basis * val_weight * jac_term * space_h->wgts[j_point] * space_v->wgts[i_point];
                        }
                    }
                    // Exploit the symmetry of the matrix.
                    CHECK_MEMORY_BOUNDS(mem_size, (idx_basis + n_v_basis) * rows + (idx_weight + n_v_basis),
                                        sizeof(double));
                    CHECK_MEMORY_BOUNDS(mem_size, (idx_weight + n_v_basis) * cols + (idx_basis + n_v_basis),
                                        sizeof(double));
                    // out.data[(idx_basis + n_v_basis) * rows + (idx_weight + n_v_basis)] =
                    out.data[(idx_weight + n_v_basis) * cols + (idx_basis + n_v_basis)] = v;
                }

    // Block 00
    for (unsigned i_weight = 0; i_weight < space_v->order + 1; ++i_weight)
        for (unsigned j_weight = 0; j_weight < space_h->order; ++j_weight)
            for (unsigned i_basis = 0; i_basis < space_v->order + 1; ++i_basis)
                for (unsigned j_basis = 0; j_basis < space_h->order; ++j_basis)
                {
                    const unsigned idx_weight = j_weight + i_weight * space_h->order;
                    const unsigned idx_basis = j_basis + i_basis * space_h->order;
                    double v = 0;
                    for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
                    {
                        for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                        {
                            const double val_basis = edge_h_basis_value(space, i_basis, j_basis, i_point, j_point);
                            const double val_weight = edge_h_basis_value(space, i_weight, j_weight, i_point, j_point);
                            const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                            const double jac_term = (jac->j11 * jac->j11 + jac->j10 * jac->j10) / jac->det;
                            v += val_basis * val_weight * jac_term * space_h->wgts[j_point] * space_v->wgts[i_point];
                        }
                    }
                    // Exploit the symmetry of the matrix.
                    CHECK_MEMORY_BOUNDS(mem_size, idx_basis * rows + idx_weight, sizeof(double));
                    CHECK_MEMORY_BOUNDS(mem_size, idx_weight * cols + idx_basis, sizeof(double));
                    // out.data[idx_basis * rows + idx_weight] =
                    out.data[idx_weight * cols + idx_basis] = v;
                }

    // Block 01
    for (unsigned i_weight = 0; i_weight < space_v->order + 1; ++i_weight)
        for (unsigned j_weight = 0; j_weight < space_h->order; ++j_weight)
            for (unsigned i_basis = 0; i_basis < space_v->order; ++i_basis)
                for (unsigned j_basis = 0; j_basis < space_h->order + 1; ++j_basis)
                {
                    const unsigned idx_weight = j_weight + i_weight * (space_h->order);
                    const unsigned idx_basis = j_basis + i_basis * (space_h->order + 1);
                    double v = 0;
                    for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
                    {
                        for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                        {
                            const double val_basis = edge_v_basis_value(space, i_basis, j_basis, i_point, j_point);
                            const double val_weight = edge_h_basis_value(space, i_weight, j_weight, i_point, j_point);
                            // (j01 * j11 + j00 * j10) / det
                            const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                            const double jac_term = (jac->j01 * jac->j11 + jac->j00 * jac->j10) / jac->det;
                            v += val_basis * val_weight * jac_term * space_h->wgts[j_point] * space_v->wgts[i_point];
                        }
                    }
                    // Exploit the symmetry of the matrix.
                    CHECK_MEMORY_BOUNDS(mem_size, idx_weight * rows + (idx_basis + n_v_basis), sizeof(double));
                    CHECK_MEMORY_BOUNDS(mem_size, (idx_basis + n_v_basis) * cols + idx_weight, sizeof(double));
                    out.data[idx_weight * rows + (idx_basis + n_v_basis)] =
                        out.data[(idx_basis + n_v_basis) * cols + idx_weight] = v;
                }

    *p_out = out;
    return EVAL_SUCCESS;
}

MFV2D_INTERNAL eval_result_t compute_mass_matrix_surf(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                      const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = space_h->order * space_v->order;
    const unsigned cols = space_h->order * space_v->order;

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return EVAL_FAILED_ALLOC;

    for (unsigned i_weight = 0; i_weight < space_v->order; ++i_weight)
    {
        for (unsigned j_weight = 0; j_weight < space_h->order; ++j_weight)
        {
            const unsigned idx_weight = j_weight + i_weight * space_h->order;
            for (unsigned i_basis = 0; i_basis < space_v->order; ++i_basis)
            {
                for (unsigned j_basis = 0; j_basis < space_h->order; ++j_basis)
                {
                    const unsigned idx_basis = j_basis + i_basis * space_h->order;

                    double v = 0;
                    for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
                    {
                        for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                        {
                            v += surf_basis_value(space, i_basis, j_basis, i_point, j_point) *
                                 surf_basis_value(space, i_weight, j_weight, i_point, j_point) /
                                 space->jacobian[j_point + i_point * space_h->n_pts].det * space_h->wgts[j_point] *
                                 space_v->wgts[i_point];
                        }
                    }
                    // Exploit the symmetry of the matrix.
                    out.data[idx_basis * rows + idx_weight] = out.data[idx_weight * cols + idx_basis] = v;
                }
            }
        }
    }

    *p_out = out;
    return EVAL_SUCCESS;
}

MFV2D_INTERNAL
const char compute_element_mass_matrices_docstr[] = "";

MFV2D_INTERNAL
PyObject *compute_element_mass_matrices(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyObject *corners = NULL;
    int order_1, order_2;
    PyObject *basis_1_nodal = NULL;
    PyObject *basis_1_edge = NULL;
    PyObject *weights_1 = NULL;
    PyObject *nodes_1 = NULL;
    PyObject *basis_2_nodal = NULL;
    PyObject *basis_2_edge = NULL;
    PyObject *weights_2 = NULL;
    PyObject *nodes_2 = NULL;

    static char *kwlist[12] = {"corners", "order_1",       "order_2",      "basis_1_nodal", "basis_1_edge", "weights_1",
                               "nodes_1", "basis_2_nodal", "basis_2_edge", "weights_2",     "nodes_2",      NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiiOOOOOOOO", kwlist,
                                     &corners,       // np.ndarray
                                     &order_1,       // int
                                     &order_2,       // int
                                     &basis_1_nodal, // np.ndarray
                                     &basis_1_edge,  // np.ndarray
                                     &weights_1,     // np.ndarray
                                     &nodes_1,       // np.ndarray
                                     &basis_2_nodal, // np.ndarray
                                     &basis_2_edge,  // np.ndarray
                                     &weights_2,     // np.ndarray
                                     &nodes_2        // np.ndarray
                                     ))
    {
        return NULL;
    }
    PyArrayObject *const corners_array = (PyArrayObject *)PyArray_FromAny(
        corners, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!corners_array)
        return NULL;

    if (PyArray_SIZE(corners_array) != 8)
    {
        PyErr_Format(PyExc_ValueError, "Expected 4 corners with two coordinates (8 points), got %d",
                     (int)PyArray_SIZE(corners_array));
        Py_DECREF(corners_array);
        return NULL;
    }
    quad_info_t quad;
    memcpy(&quad, PyArray_DATA(corners_array), sizeof quad);
    Py_DECREF(corners_array);
    fem_space_1d_t space_1, space_2;
    if (system_1d_from_python(order_1, nodes_1, weights_1, basis_1_nodal, basis_1_edge, &space_1) != EVAL_SUCCESS ||
        system_1d_from_python(order_2, nodes_2, weights_2, basis_2_nodal, basis_2_edge, &space_2) != EVAL_SUCCESS)
    {
        return NULL;
    }

    matrix_full_t mass_out = {};
    PyArrayObject *mass_node = NULL, *mass_edge = NULL, *mass_surf = NULL;

    fem_space_2d_t *space;
    eval_result_t res = fem_space_2d_create(&space_1, &space_2, &quad, &space, &SYSTEM_ALLOCATOR);
    if (res != EVAL_SUCCESS)
        goto failed;

    if ((res = compute_mass_matrix_node(space, &mass_out, &SYSTEM_ALLOCATOR)) != EVAL_SUCCESS)
        goto failed;

    mass_node = matrix_full_to_array(&mass_out);
    deallocate(&SYSTEM_ALLOCATOR, mass_out.data);
    if (!mass_node)
        goto failed;

    if ((res = compute_mass_matrix_edge(space, &mass_out, &SYSTEM_ALLOCATOR)) != EVAL_SUCCESS)
        goto failed;

    mass_edge = matrix_full_to_array(&mass_out);
    deallocate(&SYSTEM_ALLOCATOR, mass_out.data);
    if (!mass_edge)
        goto failed;

    if ((res = compute_mass_matrix_surf(space, &mass_out, &SYSTEM_ALLOCATOR)) != EVAL_SUCCESS)
        goto failed;

    mass_surf = matrix_full_to_array(&mass_out);
    deallocate(&SYSTEM_ALLOCATOR, mass_out.data);
    if (!mass_surf)
        goto failed;

    deallocate(&SYSTEM_ALLOCATOR, space);

    return PyTuple_Pack(3, mass_node, mass_edge, mass_surf);

failed:
    Py_XDECREF(mass_node);
    Py_XDECREF(mass_edge);
    Py_XDECREF(mass_surf);
    return NULL;
}

MFV2D_INTERNAL eval_result_t system_1d_from_python(const unsigned order, PyObject *pts, PyObject *wts,
                                                   PyObject *node_val, PyObject *edge_val, fem_space_1d_t *p_out)
{
    fem_space_1d_t space = {.order = order};

    if (check_input_array((PyArrayObject *)pts, 1, (const npy_intp[1]){0}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "integration points") < 0)
    {
        return EVAL_UNSPECIFIED_ERROR;
    }
    space.n_pts = PyArray_SIZE((PyArrayObject *)pts);

    if (check_input_array((PyArrayObject *)wts, 1, (const npy_intp[1]){space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "integration weights") < 0 ||
        check_input_array((PyArrayObject *)node_val, 2, (const npy_intp[2]){order + 1, space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "nodal basis") < 0 ||
        check_input_array((PyArrayObject *)edge_val, 2, (const npy_intp[2]){order, space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "edge basis") < 0)
    {
        return EVAL_UNSPECIFIED_ERROR;
    }

    space.node = (double *)PyArray_DATA((PyArrayObject *)node_val);
    space.edge = (double *)PyArray_DATA((PyArrayObject *)edge_val);
    space.wgts = (double *)PyArray_DATA((PyArrayObject *)wts);
    space.pnts = (double *)PyArray_DATA((PyArrayObject *)pts);

    *p_out = space;
    return EVAL_SUCCESS;
}
