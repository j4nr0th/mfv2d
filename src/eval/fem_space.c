#include "fem_space.h"

static const char *mass_name[MASS_CNT] = {
    [MASS_0] = "MASS_0",     [MASS_1] = "MASS_1",     [MASS_2] = "MASS_2",
    [MASS_0_I] = "MASS_0_I", [MASS_1_I] = "MASS_1_I", [MASS_2_I] = "MASS_2_I",
};

MFV2D_INTERNAL
const char *mass_mtx_indices_str(mass_mtx_indices_t v)
{
    if (v < MASS_0 || v >= MASS_CNT)
        return "UNKNOWN";
    return mass_name[v];
}

MFV2D_INTERNAL mfv2d_result_t fem_space_2d_create(const fem_space_1d_t *space_h, const fem_space_1d_t *space_v,
                                                  const quad_info_t *const quad, fem_space_2d_t **p_out,
                                                  const allocator_callbacks *allocator)
{

    const unsigned rows = space_h->n_pts;
    const unsigned cols = space_v->n_pts;

    fem_space_2d_t *const out = allocate(allocator, sizeof *out + sizeof *out->jacobian * rows * cols);
    if (!out)
        return MFV2D_FAILED_ALLOC;
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
    return MFV2D_SUCCESS;
}

static unsigned fem_space_node_basis_cnt(const fem_space_2d_t *space)
{
    return (space->space_1.order + 1) * (space->space_2.order + 1);
}

static unsigned fem_space_edge_h_basis_cnt(const fem_space_2d_t *space)
{
    return space->space_1.order * (space->space_2.order + 1);
}

static unsigned fem_space_edge_v_basis_cnt(const fem_space_2d_t *space)
{
    return (space->space_1.order + 1) * space->space_2.order;
}

static unsigned fem_space_surf_basis_cnt(const fem_space_2d_t *space)
{
    return space->space_1.order * space->space_2.order;
}

static index_2d_t fem_space_integration_node_counts(const fem_space_2d_t *this)
{
    return (index_2d_t){this->space_1.n_pts, this->space_2.n_pts};
}

static index_2d_t nodal_basis_index(const fem_space_2d_t *space, const unsigned flat_index)
{
    const unsigned i = flat_index / (space->space_1.order + 1);
    const unsigned j = flat_index % (space->space_1.order + 1);
    return (index_2d_t){i, j};
}

static index_2d_t edge_h_basis_index(const fem_space_2d_t *space, const unsigned flat_index)
{
    const unsigned i = flat_index / space->space_1.order;
    const unsigned j = flat_index % space->space_1.order;
    return (index_2d_t){i, j};
}

static index_2d_t edge_v_basis_index(const fem_space_2d_t *space, const unsigned flat_index)
{
    const unsigned i = flat_index / (space->space_1.order + 1);
    const unsigned j = flat_index % (space->space_1.order + 1);
    return (index_2d_t){i, j};
}

static index_2d_t surf_basis_index(const fem_space_2d_t *space, const unsigned flat_index)
{
    const unsigned i = flat_index / space->space_1.order;
    const unsigned j = flat_index % space->space_1.order;
    return (index_2d_t){i, j};
}

static double node_basis_value_2d(const fem_space_2d_t *space, const unsigned i_basis, const unsigned j_basis,
                                  const unsigned i_point, const unsigned j_point)
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
        return 0;
    }

    const double *const basis_1 = space->space_1.node + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.node + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double node_basis_value(const fem_space_2d_t *space, const unsigned idx, const unsigned i_point,
                               const unsigned j_point)
{
    const index_2d_t index = nodal_basis_index(space, idx);
    return node_basis_value_2d(space, index.i, index.j, i_point, j_point);
}

static double edge_h_basis_value_2d(const fem_space_2d_t *space, const unsigned i_basis, const unsigned j_basis,
                                    const unsigned i_point, const unsigned j_point)
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
        return 0;
    }

    const double *const basis_1 = space->space_1.node + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.edge + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double edge_h_basis_value(const fem_space_2d_t *space, const unsigned idx, const unsigned i_point,
                                 const unsigned j_point)
{
    const index_2d_t index = edge_h_basis_index(space, idx);
    return edge_h_basis_value_2d(space, index.i, index.j, i_point, j_point);
}

static double edge_v_basis_value_2d(const fem_space_2d_t *space, const unsigned i_basis, const unsigned j_basis,
                                    const unsigned i_point, const unsigned j_point)
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
        return 0;
    }

    const double *const basis_1 = space->space_1.edge + i_basis * space->space_1.n_pts;
    const double *const basis_2 = space->space_2.node + j_basis * space->space_2.n_pts;

    return basis_1[i_point] * basis_2[j_point];
}

static double edge_v_basis_value(const fem_space_2d_t *space, const unsigned idx, const unsigned i_point,
                                 const unsigned j_point)
{
    const index_2d_t index = edge_v_basis_index(space, idx);
    return edge_v_basis_value_2d(space, index.i, index.j, i_point, j_point);
}

static double surf_basis_value_2d(const fem_space_2d_t *space, const unsigned i_basis, const unsigned j_basis,
                                  const unsigned i_point, const unsigned j_point)
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

static double surf_basis_value(const fem_space_2d_t *space, const unsigned idx, const unsigned i_point,
                               const unsigned j_point)
{
    const index_2d_t index = surf_basis_index(space, idx);
    return surf_basis_value_2d(space, index.i, index.j, i_point, j_point);
}

static double integration_weight_value(const fem_space_2d_t *space, const unsigned i_point, const unsigned j_point)
{
    if (ASSERT(i_point < space->space_1.n_pts, "Point is out of range for basis 1") ||
        ASSERT(j_point < space->space_2.n_pts, "Point is out of range for basis 2"))
        return 0;
    return space->space_1.wgts[j_point] * space->space_2.wgts[i_point];
}

MFV2D_INTERNAL mfv2d_result_t compute_mass_matrix_node(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                       const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = fem_space_node_basis_cnt(space);
    const unsigned cols = fem_space_node_basis_cnt(space);

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

    for (unsigned idx_weight = 0; idx_weight < rows; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < rows; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
            {
                for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                {
                    v += node_basis_value(space, idx_basis, i_point, j_point) *
                         node_basis_value(space, idx_weight, i_point, j_point) *
                         space->jacobian[j_point + i_point * space_h->n_pts].det *
                         integration_weight_value(space, i_point, j_point);
                }
            }
            // Exploit the symmetry of the matrix.
            out.data[idx_basis * rows + idx_weight] = out.data[idx_weight * cols + idx_basis] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL mfv2d_result_t compute_mass_matrix_edge(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                       const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned n_v_basis = fem_space_edge_v_basis_cnt(space); // (space_v->order + 1) * space_h->order;
    const unsigned n_h_basis = fem_space_edge_h_basis_cnt(space); // space_v->order * (space_h->order + 1);

    const unsigned rows = n_v_basis + n_h_basis;
    const unsigned cols = n_v_basis + n_h_basis;

    const size_t mem_size = sizeof(double) * rows * cols;
    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, mem_size)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

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

    // Block 11
    for (unsigned idx_weight = 0; idx_weight < n_v_basis; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_v_basis; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
            {
                for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                {
                    const double val_basis = edge_v_basis_value(space, idx_basis, i_point, j_point);
                    const double val_weight = edge_v_basis_value(space, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                    const double jac_term = (jac->j00 * jac->j00 + jac->j01 * jac->j01) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space, i_point, j_point);
                }
            }
            // Exploit the symmetry of the matrix.
            CHECK_MEMORY_BOUNDS(mem_size, (idx_basis + n_v_basis) * rows + (idx_weight + n_v_basis), sizeof(double));
            CHECK_MEMORY_BOUNDS(mem_size, (idx_weight + n_v_basis) * cols + (idx_basis + n_v_basis), sizeof(double));
            // out.data[(idx_basis + n_v_basis) * rows + (idx_weight + n_v_basis)] =
            out.data[(idx_weight + n_v_basis) * cols + (idx_basis + n_v_basis)] = v;
        }

    // Block 00
    for (unsigned idx_weight = 0; idx_weight < n_h_basis; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_h_basis; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
            {
                for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                {
                    const double val_basis = edge_h_basis_value(space, idx_basis, i_point, j_point);
                    const double val_weight = edge_h_basis_value(space, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                    const double jac_term = (jac->j11 * jac->j11 + jac->j10 * jac->j10) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space, i_point, j_point);
                }
            }
            // Exploit the symmetry of the matrix.
            CHECK_MEMORY_BOUNDS(mem_size, idx_basis * rows + idx_weight, sizeof(double));
            CHECK_MEMORY_BOUNDS(mem_size, idx_weight * cols + idx_basis, sizeof(double));
            // out.data[idx_basis * rows + idx_weight] =
            out.data[idx_weight * cols + idx_basis] = v;
        }

    // Block 10/01
    for (unsigned idx_weight = 0; idx_weight < n_h_basis; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_v_basis; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
            {
                for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                {
                    const double val_basis = edge_v_basis_value(space, idx_basis, i_point, j_point);
                    const double val_weight = edge_h_basis_value(space, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space->jacobian + (j_point + i_point * space_h->n_pts);
                    const double jac_term = (jac->j01 * jac->j11 + jac->j00 * jac->j10) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space, i_point, j_point);
                }
            }
            // Exploit the symmetry of the matrix.
            CHECK_MEMORY_BOUNDS(mem_size, idx_weight * rows + (idx_basis + n_v_basis), sizeof(double));
            CHECK_MEMORY_BOUNDS(mem_size, (idx_basis + n_v_basis) * cols + idx_weight, sizeof(double));
            out.data[idx_weight * rows + (idx_basis + n_v_basis)] =
                out.data[(idx_basis + n_v_basis) * cols + idx_weight] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL mfv2d_result_t compute_mass_matrix_surf(const fem_space_2d_t *space, matrix_full_t *p_out,
                                                       const allocator_callbacks *allocator)
{
    const fem_space_1d_t *const space_h = &space->space_1;
    const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = fem_space_surf_basis_cnt(space);
    const unsigned cols = fem_space_surf_basis_cnt(space);

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

    for (unsigned idx_weight = 0; idx_weight < rows; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < cols; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < space_v->n_pts; ++i_point)
            {
                for (unsigned j_point = 0; j_point < space_h->n_pts; ++j_point)
                {
                    v += surf_basis_value(space, idx_basis, i_point, j_point) *
                         surf_basis_value(space, idx_weight, i_point, j_point) /
                         space->jacobian[j_point + i_point * space_h->n_pts].det *
                         integration_weight_value(space, i_point, j_point);
                }
            }
            // Exploit the symmetry of the matrix.
            out.data[idx_basis * rows + idx_weight] = out.data[idx_weight * cols + idx_basis] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
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
    if (fem_space_1d_from_python(order_1, nodes_1, weights_1, basis_1_nodal, basis_1_edge, &space_1) != MFV2D_SUCCESS ||
        fem_space_1d_from_python(order_2, nodes_2, weights_2, basis_2_nodal, basis_2_edge, &space_2) != MFV2D_SUCCESS)
    {
        return NULL;
    }

    matrix_full_t mass_out = {};
    PyArrayObject *mass_node = NULL, *mass_edge = NULL, *mass_surf = NULL;

    fem_space_2d_t *space;
    mfv2d_result_t res = fem_space_2d_create(&space_1, &space_2, &quad, &space, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
        goto failed;

    if ((res = compute_mass_matrix_node(space, &mass_out, &SYSTEM_ALLOCATOR)) != MFV2D_SUCCESS)
        goto failed;

    mass_node = matrix_full_to_array(&mass_out);
    deallocate(&SYSTEM_ALLOCATOR, mass_out.data);
    if (!mass_node)
        goto failed;

    if ((res = compute_mass_matrix_edge(space, &mass_out, &SYSTEM_ALLOCATOR)) != MFV2D_SUCCESS)
        goto failed;

    mass_edge = matrix_full_to_array(&mass_out);
    deallocate(&SYSTEM_ALLOCATOR, mass_out.data);
    if (!mass_edge)
        goto failed;

    if ((res = compute_mass_matrix_surf(space, &mass_out, &SYSTEM_ALLOCATOR)) != MFV2D_SUCCESS)
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

MFV2D_INTERNAL mfv2d_result_t fem_space_1d_from_python(const unsigned order, PyObject *pts, PyObject *wts,
                                                       PyObject *node_val, PyObject *edge_val, fem_space_1d_t *p_out)
{
    fem_space_1d_t space = {.order = order};

    if (check_input_array((PyArrayObject *)pts, 1, (const npy_intp[1]){0}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "integration points") < 0)
    {
        return MFV2D_UNSPECIFIED_ERROR;
    }
    space.n_pts = PyArray_SIZE((PyArrayObject *)pts);

    if (check_input_array((PyArrayObject *)wts, 1, (const npy_intp[1]){space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "integration weights") < 0 ||
        check_input_array((PyArrayObject *)node_val, 2, (const npy_intp[2]){order + 1, space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "nodal basis") < 0 ||
        check_input_array((PyArrayObject *)edge_val, 2, (const npy_intp[2]){order, space.n_pts}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "edge basis") < 0)
    {
        return MFV2D_UNSPECIFIED_ERROR;
    }

    space.node = (double *)PyArray_DATA((PyArrayObject *)node_val);
    space.edge = (double *)PyArray_DATA((PyArrayObject *)edge_val);
    space.wgts = (double *)PyArray_DATA((PyArrayObject *)wts);
    space.pnts = (double *)PyArray_DATA((PyArrayObject *)pts);

    *p_out = space;
    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_node_edge(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field,
                                             const int transpose)
{
    const unsigned n_nodal = fem_space_node_basis_cnt(fem_space);
    const unsigned n_edge_h = fem_space_edge_h_basis_cnt(fem_space);
    const unsigned n_edge_v = fem_space_edge_v_basis_cnt(fem_space);

    unsigned rows, cols;
    if (transpose)
    {
        rows = n_edge_h + n_edge_v;
        cols = n_nodal;
    }
    else
    {
        rows = n_nodal;
        cols = n_edge_h + n_edge_v;
    }

    const matrix_full_t mat = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *mat.data * rows * cols)};
    if (!mat.data)
        return MFV2D_FAILED_ALLOC;

    const unsigned n_pts_2 = fem_space->space_2.n_pts;
    const unsigned n_pts_1 = fem_space->space_1.n_pts;

    // Mix 10
    //  Left half, which is involved with eta-basis
    for (unsigned i_weight = 0; i_weight < n_nodal; ++i_weight)
    {
        for (unsigned i_basis = 0; i_basis < n_edge_h; ++i_basis)
        {
            double val = 0.0;
            for (unsigned i = 0; i < n_pts_2; ++i)
            {
                for (unsigned j = 0; j < n_pts_1; ++j)
                {
                    const jacobian_t *const jac = fem_space->jacobian + (j + i * n_pts_1);
                    const double vector_comp = field[i * (2 * n_pts_1) + 2 * j + 0] * jac->j11 -
                                               field[i * (2 * n_pts_1) + 2 * j + 1] * jac->j10;
                    val += node_basis_value(fem_space, i_weight, i, j) * edge_h_basis_value(fem_space, i_basis, i, j) *
                           vector_comp * integration_weight_value(fem_space, i, j);
                }
            }
            if (transpose)
            {
                mat.data[i_basis * cols + i_weight] = val;
            }
            else
            {

                mat.data[i_weight * cols + i_basis] = val;
            }
        }
    }
    //  Right half, which is involved with xi-basis
    for (unsigned i_weight = 0; i_weight < n_nodal; ++i_weight)
    {
        for (unsigned i_basis = 0; i_basis < n_edge_v; ++i_basis)
        {
            double val = 0.0;
            for (unsigned i = 0; i < n_pts_2; ++i)
            {
                for (unsigned j = 0; j < n_pts_1; ++j)
                {
                    const jacobian_t *const jac = fem_space->jacobian + (j + i * n_pts_1);
                    const double vector_comp = -(field[i * (2 * n_pts_2) + 2 * j + 1] * jac->j00 -
                                                 field[i * (2 * n_pts_2) + 2 * j + 0] * jac->j01);
                    val += node_basis_value(fem_space, i_weight, i, j) * edge_v_basis_value(fem_space, i_basis, i, j) *
                           vector_comp * integration_weight_value(fem_space, i, j);
                }
            }
            if (transpose)
            {
                mat.data[(i_basis + n_edge_h) * cols + i_weight] = val;
            }
            else
            {

                mat.data[i_weight * cols + (i_basis + n_edge_h)] = val;
            }
        }
    }

    *p_out = mat;

    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge_edge(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field, const int dual)
{
    const unsigned n_h_basis = fem_space_edge_h_basis_cnt(fem_space);
    const unsigned n_v_basis = fem_space_edge_v_basis_cnt(fem_space);

    const unsigned rows = n_h_basis + n_v_basis;
    const unsigned cols = n_h_basis + n_v_basis;

    const matrix_full_t mat = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *mat.data * rows * cols)};
    if (!mat.data)
        return MFV2D_FAILED_ALLOC;

    const unsigned n_pts_2 = fem_space->space_2.n_pts;
    const unsigned n_pts_1 = fem_space->space_1.n_pts;

    if (!dual)
    {
        // Block 11
        for (unsigned idx_weight = 0; idx_weight < n_v_basis; ++idx_weight)
            for (unsigned idx_basis = 0; idx_basis < n_v_basis; ++idx_basis)
            {
                double v = 0;
                for (unsigned i_point = 0; i_point < n_pts_2; ++i_point)
                {
                    for (unsigned j_point = 0; j_point < n_pts_1; ++j_point)
                    {
                        const double val_basis = edge_v_basis_value(fem_space, idx_basis, i_point, j_point);
                        const double val_weight = edge_v_basis_value(fem_space, idx_weight, i_point, j_point);
                        const jacobian_t *jac = fem_space->jacobian + (j_point + i_point * n_pts_1);
                        const double jac_term = field[i_point * (2 * n_pts_1) + 2 * j_point + 0] *
                                                (jac->j00 * jac->j00 + jac->j01 * jac->j01) / jac->det;
                        v += val_basis * val_weight * jac_term * integration_weight_value(fem_space, i_point, j_point);
                    }
                }
                // out.data[(idx_basis + n_v_basis) * rows + (idx_weight + n_v_basis)] =
                mat.data[(idx_weight + n_v_basis) * cols + (idx_basis + n_v_basis)] = v;
            }

        // Block 00
        for (unsigned idx_weight = 0; idx_weight < n_h_basis; ++idx_weight)
            for (unsigned idx_basis = 0; idx_basis < n_h_basis; ++idx_basis)
            {
                double v = 0;
                for (unsigned i_point = 0; i_point < n_pts_2; ++i_point)
                {
                    for (unsigned j_point = 0; j_point < n_pts_1; ++j_point)
                    {
                        const double val_basis = edge_h_basis_value(fem_space, idx_basis, i_point, j_point);
                        const double val_weight = edge_h_basis_value(fem_space, idx_weight, i_point, j_point);
                        const jacobian_t *jac = fem_space->jacobian + (j_point + i_point * n_pts_1);
                        const double jac_term = field[i_point * (2 * n_pts_1) + 2 * j_point + 0] *
                                                (jac->j11 * jac->j11 + jac->j10 * jac->j10) / jac->det;
                        v += val_basis * val_weight * jac_term * integration_weight_value(fem_space, i_point, j_point);
                    }
                }
                // Exploit the symmetry of the matrix.
                // out.data[idx_basis * rows + idx_weight] =
                mat.data[idx_weight * cols + idx_basis] = v;
            }

        // Block 10/01
        for (unsigned idx_weight = 0; idx_weight < n_h_basis; ++idx_weight)
            for (unsigned idx_basis = 0; idx_basis < n_v_basis; ++idx_basis)
            {
                double v = 0;
                for (unsigned i_point = 0; i_point < n_pts_2; ++i_point)
                {
                    for (unsigned j_point = 0; j_point < n_pts_1; ++j_point)
                    {
                        const double val_basis = edge_v_basis_value(fem_space, idx_basis, i_point, j_point);
                        const double val_weight = edge_h_basis_value(fem_space, idx_weight, i_point, j_point);
                        const jacobian_t *jac = fem_space->jacobian + (j_point + i_point * n_pts_1);
                        const double jac_term = field[i_point * (2 * n_pts_1) + 2 * j_point + 0] *
                                                (jac->j01 * jac->j11 + jac->j00 * jac->j10) / jac->det;
                        v += val_basis * val_weight * jac_term * integration_weight_value(fem_space, i_point, j_point);
                    }
                }
                // Exploit the symmetry of the matrix.
                mat.data[idx_weight * rows + (idx_basis + n_v_basis)] =
                    mat.data[(idx_basis + n_v_basis) * cols + idx_weight] = v;
            }
    }
    else
    {
        memset(mat.data, 0, sizeof *mat.data * rows * cols);

        for (unsigned idx_weight = 0; idx_weight < n_h_basis; ++idx_weight)
            for (unsigned idx_basis = 0; idx_basis < n_v_basis; ++idx_basis)
            {
                double v = 0;
                for (unsigned i_point = 0; i_point < n_pts_2; ++i_point)
                {
                    for (unsigned j_point = 0; j_point < n_pts_1; ++j_point)
                    {
                        const double val_basis = edge_v_basis_value(fem_space, idx_basis, i_point, j_point);
                        const double val_weight = edge_h_basis_value(fem_space, idx_weight, i_point, j_point);
                        const double jac_term = field[i_point * (2 * n_pts_1) + 2 * j_point + 0];
                        v += val_basis * val_weight * jac_term * integration_weight_value(fem_space, i_point, j_point);
                    }
                }
                // Exploit the anit-symmetry of the matrix.
                mat.data[idx_weight * rows + (idx_basis + n_v_basis)] = +v;
                mat.data[(idx_basis + n_v_basis) * rows + idx_weight] = -v;
            }
    }

    *p_out = mat;

    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge_surf(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field,
                                             const int transpose)

{
    const unsigned n_edge_h = fem_space_edge_h_basis_cnt(fem_space);
    const unsigned n_edge_v = fem_space_edge_v_basis_cnt(fem_space);
    const unsigned n_surf = fem_space_surf_basis_cnt(fem_space);

    unsigned rows, cols;
    if (transpose)
    {
        rows = n_surf;
        cols = n_edge_h + n_edge_v;
    }
    else
    {
        rows = n_edge_h + n_edge_v;
        cols = n_surf;
    }

    const matrix_full_t mat = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *mat.data * rows * cols)};
    if (!mat.data)
        return MFV2D_FAILED_ALLOC;

    const unsigned n_pts_2 = fem_space->space_2.n_pts;
    const unsigned n_pts_1 = fem_space->space_1.n_pts;

    // Mix 21
    // Left half, which is involved with eta-basis
    for (unsigned i_weight = 0; i_weight < n_edge_h; ++i_weight)
    {
        for (unsigned i_basis = 0; i_basis < n_surf; ++i_basis)
        {
            double val = 0.0;
            for (unsigned i = 0; i < n_pts_2; ++i)
            {
                for (unsigned j = 0; j < n_pts_1; ++j)
                {
                    const jacobian_t *const jac = fem_space->jacobian + (j + i * n_pts_1);

                    const double vector_comp = -(field[i * (2 * n_pts_1) + 2 * j + 0] * jac->j10 +
                                                 field[i * (2 * n_pts_1) + 2 * j + 1] * jac->j11) /
                                               jac->det;
                    val += surf_basis_value(fem_space, i_basis, i, j) * edge_h_basis_value(fem_space, i_weight, i, j) *
                           vector_comp * integration_weight_value(fem_space, i, j);
                }
            }
            if (transpose)
            {
                mat.data[i_basis * cols + i_weight] = val;
            }
            else
            {

                mat.data[i_weight * cols + i_basis] = val;
            }
        }
    }
    //  Right half, which is involved with xi-basis
    for (unsigned i_weight = 0; i_weight < n_edge_v; ++i_weight)
    {
        for (unsigned i_basis = 0; i_basis < n_surf; ++i_basis)
        {
            double val = 0.0;
            for (unsigned i = 0; i < n_pts_2; ++i)
            {
                for (unsigned j = 0; j < n_pts_1; ++j)
                {
                    const jacobian_t *const jac = fem_space->jacobian + (j + i * n_pts_1);
                    const double vector_comp = -(field[i * (2 * n_pts_1) + 2 * j + 0] * jac->j00 +
                                                 field[i * (2 * n_pts_1) + 2 * j + 1] * jac->j01) /
                                               jac->det;
                    val += surf_basis_value(fem_space, i_basis, i, j) * edge_v_basis_value(fem_space, i_weight, i, j) *
                           vector_comp * integration_weight_value(fem_space, i, j);
                }
            }
            if (transpose)
            {
                mat.data[i_basis * cols + (i_weight + n_edge_h)] = val;
            }
            else
            {

                mat.data[(i_weight + n_edge_h) * cols + i_basis] = val;
            }
        }
    }

    *p_out = mat;

    return MFV2D_SUCCESS;
}
mfv2d_result_t compute_mass_matrix_node_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    const unsigned rows = fem_space_node_basis_cnt(space_out);
    const unsigned cols = fem_space_node_basis_cnt(space_in);
    const index_2d_t int_nodes = fem_space_integration_node_counts(space_in);
    {
        const index_2d_t int_nodes_2 = fem_space_integration_node_counts(space_out);
        if ASSERT (int_nodes.i == int_nodes_2.i && int_nodes.j == int_nodes_2.j, "Integration space of two must match")
        {
            return MFV2D_DIMS_MISMATCH;
        }
    }

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

    for (unsigned idx_weight = 0; idx_weight < rows; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < cols; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    v += node_basis_value(space_in, idx_basis, i_point, j_point) *
                         node_basis_value(space_out, idx_weight, i_point, j_point) *
                         space_in->jacobian[j_point + i_point * int_nodes.i].det *
                         integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[idx_weight * cols + idx_basis] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
}
mfv2d_result_t compute_mass_matrix_edge_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    const unsigned n_v_basis_in = fem_space_edge_v_basis_cnt(space_in);
    const unsigned n_h_basis_in = fem_space_edge_h_basis_cnt(space_in);
    const unsigned n_v_basis_out = fem_space_edge_v_basis_cnt(space_out);
    const unsigned n_h_basis_out = fem_space_edge_h_basis_cnt(space_out);

    const unsigned rows = n_v_basis_out + n_h_basis_out;
    const unsigned cols = n_v_basis_in + n_h_basis_in;
    const index_2d_t int_nodes = fem_space_integration_node_counts(space_in);
    {
        const index_2d_t int_nodes_2 = fem_space_integration_node_counts(space_out);
        if ASSERT (int_nodes.i == int_nodes_2.i && int_nodes.j == int_nodes_2.j, "Integration space of two must match")
        {
            return MFV2D_DIMS_MISMATCH;
        }
    }

    const size_t mem_size = sizeof(double) * rows * cols;
    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, mem_size)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

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

    // Block 11
    for (unsigned idx_weight = 0; idx_weight < n_v_basis_out; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_v_basis_in; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    const double val_basis = edge_v_basis_value(space_in, idx_basis, i_point, j_point);
                    const double val_weight = edge_v_basis_value(space_out, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space_in->jacobian + (j_point + i_point * int_nodes.i);
                    const double jac_term = (jac->j00 * jac->j00 + jac->j01 * jac->j01) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[(idx_weight + n_v_basis_out) * cols + (idx_basis + n_v_basis_in)] = v;
        }

    // Block 00
    for (unsigned idx_weight = 0; idx_weight < n_h_basis_out; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_h_basis_in; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    const double val_basis = edge_h_basis_value(space_in, idx_basis, i_point, j_point);
                    const double val_weight = edge_h_basis_value(space_out, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space_in->jacobian + (j_point + i_point * int_nodes.i);
                    const double jac_term = (jac->j11 * jac->j11 + jac->j10 * jac->j10) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[idx_weight * cols + idx_basis] = v;
        }

    // Block 01
    for (unsigned idx_weight = 0; idx_weight < n_h_basis_out; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_v_basis_in; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    const double val_basis = edge_v_basis_value(space_in, idx_basis, i_point, j_point);
                    const double val_weight = edge_h_basis_value(space_out, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space_in->jacobian + (j_point + i_point * int_nodes.i);
                    const double jac_term = (jac->j01 * jac->j11 + jac->j00 * jac->j10) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[idx_weight * cols + (idx_basis + n_v_basis_in)] = v;
        }
    // Block 10
    for (unsigned idx_weight = 0; idx_weight < n_v_basis_out; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < n_h_basis_in; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    const double val_basis = edge_h_basis_value(space_in, idx_basis, i_point, j_point);
                    const double val_weight = edge_v_basis_value(space_out, idx_weight, i_point, j_point);
                    const jacobian_t *jac = space_in->jacobian + (j_point + i_point * int_nodes.i);
                    const double jac_term = (jac->j01 * jac->j11 + jac->j00 * jac->j10) / jac->det;
                    v += val_basis * val_weight * jac_term * integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[(idx_weight + n_v_basis_out) * cols + idx_basis] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
}
mfv2d_result_t compute_mass_matrix_surf_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    // const fem_space_1d_t *const space_h = &space->space_1;
    // const fem_space_1d_t *const space_v = &space->space_2;

    const unsigned rows = fem_space_surf_basis_cnt(space_out);
    const unsigned cols = fem_space_surf_basis_cnt(space_in);

    const index_2d_t int_nodes = fem_space_integration_node_counts(space_in);
    {
        const index_2d_t int_nodes_2 = fem_space_integration_node_counts(space_out);
        if ASSERT (int_nodes.i == int_nodes_2.i && int_nodes.j == int_nodes_2.j, "Integration space of two must match")
        {
            return MFV2D_DIMS_MISMATCH;
        }
    }

    const matrix_full_t out = {.base = {.type = MATRIX_TYPE_FULL, .rows = rows, .cols = cols},
                               .data = allocate(allocator, sizeof *out.data * rows * cols)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;

    for (unsigned idx_weight = 0; idx_weight < rows; ++idx_weight)
        for (unsigned idx_basis = 0; idx_basis < cols; ++idx_basis)
        {
            double v = 0;
            for (unsigned i_point = 0; i_point < int_nodes.j; ++i_point)
            {
                for (unsigned j_point = 0; j_point < int_nodes.i; ++j_point)
                {
                    v += surf_basis_value(space_in, idx_basis, i_point, j_point) *
                         surf_basis_value(space_out, idx_weight, i_point, j_point) /
                         space_in->jacobian[j_point + i_point * int_nodes.i].det *
                         integration_weight_value(space_in, i_point, j_point);
                }
            }
            // There is no symmetry
            out.data[idx_weight * cols + idx_basis] = v;
        }

    *p_out = out;
    return MFV2D_SUCCESS;
}
