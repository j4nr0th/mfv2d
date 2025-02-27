//
// Created by jan on 20.2.2025.
//

#include "precomp.h"

static const char *mass_name[MASS_CNT] = {
    [MASS_0] = "MASS_0",     [MASS_1] = "MASS_1",     [MASS_2] = "MASS_2",
    [MASS_0_I] = "MASS_0_I", [MASS_1_I] = "MASS_1_I", [MASS_2_I] = "MASS_2_I",
};

INTERPLIB_INTERNAL
const char *mass_mtx_indices_str(mass_mtx_indices_t v)
{
    if (v < MASS_0 || v >= MASS_CNT)
        return "UNKNOWN";
    return mass_name[v];
}

INTERPLIB_INTERNAL
int basis_precomp_create(PyObject *serialized, basis_precomp_t *out)
{
    int order, n_int;
    PyObject *int_nodes, *node_precomp, *edge_00_precomp, *edge_01_precomp, *edge_11_precomp, *surface_precomp;
    if (!PyArg_ParseTuple(serialized, "iiOOOOOO", &order, &n_int, &int_nodes, &node_precomp, &edge_00_precomp,
                          &edge_01_precomp, &edge_11_precomp, &surface_precomp))
    {
        return 0;
    }

    PyArrayObject *const arr_int_nodes = (PyArrayObject *)PyArray_FromAny(
        int_nodes, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_node_precomp = (PyArrayObject *)PyArray_FromAny(
        node_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_edge_00_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_00_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_edge_01_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_01_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_edge_11_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_11_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_surface_precomp = (PyArrayObject *)PyArray_FromAny(
        surface_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);

    if (!arr_int_nodes || !arr_node_precomp || !arr_edge_00_precomp || !arr_edge_01_precomp || !arr_edge_11_precomp ||
        !arr_surface_precomp)
    {
        goto failed;
    }

    npy_intp sz;
    //  Check sizes for integration nodes match
    if ((sz = PyArray_DIM(arr_int_nodes, 0)) != n_int)
    {
        PyErr_Format(PyExc_ValueError, "Integration nodes don't have the size specified by n_int (%u vs %u).", sz,
                     n_int);
        goto failed;
    }
    // Check sizes for node precomp
    const npy_intp *dims = PyArray_DIMS(arr_node_precomp);
    if ((dims[0] != (order + 1) * (order + 1)) || (dims[1] != dims[0]) || (dims[2] != n_int) || dims[2] != dims[3])
    {
        PyErr_Format(PyExc_ValueError,
                     "Shape of the nodal pre-computed array is not as expected (expected"
                     " to get (%u, %u, %u, %u), but got (%u, %u, %u, %u)).",
                     (order + 1) * (order + 1), (order + 1) * (order + 1), n_int, n_int, dims[0], dims[1], dims[2],
                     dims[3]);
        goto failed;
    }
    // Check sizes for edge 00 precomp
    dims = PyArray_DIMS(arr_edge_00_precomp);
    if ((dims[0] != (order + 1) * (order)) || (dims[1] != dims[0]) || (dims[2] != n_int) || dims[2] != dims[3])
    {
        PyErr_Format(PyExc_ValueError,
                     "Shape of the edge 00 pre-computed array is not as expected (expected"
                     " to get (%u, %u, %u, %u), but got (%u, %u, %u, %u)).",
                     (order + 1) * (order), (order) * (order + 1), n_int, n_int, dims[0], dims[1], dims[2], dims[3]);
        goto failed;
    }
    // Check sizes for edge 01 precomp
    dims = PyArray_DIMS(arr_edge_01_precomp);
    if ((dims[0] != (order + 1) * (order)) || (dims[1] != dims[0]) || (dims[2] != n_int) || dims[2] != dims[3])
    {
        PyErr_Format(PyExc_ValueError,
                     "Shape of the edge 01 pre-computed array is not as expected (expected"
                     " to get (%u, %u, %u, %u), but got (%u, %u, %u, %u)).",
                     (order + 1) * (order), (order) * (order + 1), n_int, n_int, dims[0], dims[1], dims[2], dims[3]);
        goto failed;
    }
    // Check sizes for edge 11 precomp
    dims = PyArray_DIMS(arr_edge_11_precomp);
    if ((dims[0] != (order + 1) * (order)) || (dims[1] != dims[0]) || (dims[2] != n_int) || dims[2] != dims[3])
    {
        PyErr_Format(PyExc_ValueError,
                     "Shape of the edge 11 pre-computed array is not as expected (expected"
                     " to get (%u, %u, %u, %u), but got (%u, %u, %u, %u)).",
                     (order + 1) * (order), (order) * (order + 1), n_int, n_int, dims[0], dims[1], dims[2], dims[3]);
        goto failed;
    }

    // Check sizes for surface precomp
    dims = PyArray_DIMS(arr_surface_precomp);
    if ((dims[0] != (order) * (order)) || (dims[1] != dims[0]) || (dims[2] != n_int) || dims[2] != dims[3])
    {
        PyErr_Format(PyExc_ValueError,
                     "Shape of the nodal pre-computed array is not as expected (expected"
                     " to get (%u, %u, %u, %u), but got (%u, %u, %u, %u)).",
                     (order) * (order), (order) * (order), n_int, n_int, dims[0], dims[1], dims[2], dims[3]);
        goto failed;
    }

    *out = (basis_precomp_t){
        .order = order,
        .n_int = n_int,
        .nodes_int = PyArray_DATA(arr_int_nodes),
        .arr_int_nodes = arr_int_nodes,
        .mass_nodal = PyArray_DATA(arr_node_precomp),
        .arr_node = arr_node_precomp,
        .mass_edge_00 = PyArray_DATA(arr_edge_00_precomp),
        .arr_edge_00 = arr_edge_00_precomp,
        .mass_edge_01 = PyArray_DATA(arr_edge_01_precomp),
        .arr_edge_01 = arr_edge_01_precomp,
        .mass_edge_11 = PyArray_DATA(arr_edge_11_precomp),
        .arr_edge_11 = arr_edge_11_precomp,
        .mass_surf = PyArray_DATA(arr_surface_precomp),
        .arr_surf = arr_surface_precomp,
    };

    return 1;

failed:

    Py_XDECREF(arr_int_nodes);
    Py_XDECREF(arr_node_precomp);
    Py_XDECREF(arr_edge_00_precomp);
    Py_XDECREF(arr_edge_01_precomp);
    Py_XDECREF(arr_edge_11_precomp);
    Py_XDECREF(arr_surface_precomp);
    return 0;
}

/**
 * Invert the matrix by non-pivoted LU decomposition, where the diagonal of L is assumed to be 1.
 *
 * @param n Dimension of the matrix.
 * @param mat Matrix which to invert. Can be equal to ``out``.
 * @param buffer Buffer used for intermediate calculations. Receives the LU decomposition of the matrix.
 * @param out Where to write the resulting inverse matrix to. Can be equal to ``mat``.
 */
static void invert_matrix(const unsigned n, const double mat[static n * n], double buffer[restrict n * n],
                          double out[n * n])
{
    for (uint32_t i = 0; i < n; ++i)
    {
        //  Deal with a row of L
        for (uint_fast32_t j = 0; j < i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *li = buffer + n * i;
            //  Column of U
            const double *uj = buffer + j;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += li[k] * uj[k * n];
            }
            buffer[n * i + j] = (mat[n * i + j] - v) / uj[n * j];
        }

        //  Deal with a column of U
        for (uint_fast32_t j = 0; j <= i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *lj = buffer + n * j;
            //  Column of U
            const double *ui = buffer + i;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += lj[k] * ui[k * n];
            }
            buffer[n * j + i] = mat[n * j + i] - v;
        }
    }

    // Now use back and forward substitution to compute inverse of the matrix
    // based on the following expression:
    //
    //      L @ U @ (A^{-1}) = I
    //
    // First solve `L @ B = I` for `B`

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < n; ++j)
        {
            double v = j == i;
            for (unsigned k = 0; k < i; ++k)
            {
                v -= buffer[i * n + k] * out[k * n + j];
            }
            out[i * n + j] = v;
        }
    }
    // Now solve `U @ (A^{-1}) = B` for `A^{-1}`

    for (unsigned i = n; i > 0; --i)
    {
        for (unsigned j = 0; j < n; ++j)
        {
            double v = out[(i - 1) * n + j];
            for (unsigned k = i; k < n; ++k)
            {
                v -= buffer[(i - 1) * n + k] * out[k * n + j];
            }
            out[(i - 1) * n + j] = v / buffer[(i - 1) * n + (i - 1)];
        }
    }
}

INTERPLIB_INTERNAL
void basis_precomp_destroy(basis_precomp_t *this)
{
    Py_XDECREF(this->arr_int_nodes);
    Py_XDECREF(this->arr_node);
    Py_XDECREF(this->arr_edge_00);
    Py_XDECREF(this->arr_edge_01);
    Py_XDECREF(this->arr_edge_11);
    Py_XDECREF(this->arr_surf);
    *this = (basis_precomp_t){};
}

int precompute_create(const basis_precomp_t *basis, double x0, double x1, double x2, double x3, double y0, double y1,
                      double y2, double y3, precompute_t *out, allocator_callbacks *allocator)
{
    *out = (precompute_t){.basis = basis};

    const unsigned n_int = basis->n_int;

    jacobian_t *restrict p_jac;
    const size_t sz_jac = sizeof *p_jac * (n_int * n_int);
    p_jac = allocate(allocator, sz_jac);
    if (!p_jac)
    {
        return 0;
    }

    // Compute the Jacobian at the integration nodes.
    for (unsigned row = 0; row < n_int; ++row)
    {
        const double eta = basis->nodes_int[row];
        const double dxdxi = ((x1 - x0) * (1 - eta) + (x2 - x3) * (1 + eta)) / 4;
        const double dydxi = ((y1 - y0) * (1 - eta) + (y2 - y3) * (1 + eta)) / 4;
        for (unsigned col = 0; col < n_int; ++col)
        {
            const double xi = basis->nodes_int[col];
            const double dxdeta = ((x3 - x0) * (1 - xi) + (x2 - x1) * (1 + xi)) / 4;
            const double dydeta = ((y3 - y0) * (1 - xi) + (y2 - y1) * (1 + xi)) / 4;
            p_jac[row * n_int + col].j00 = dxdxi;
            p_jac[row * n_int + col].j01 = dxdeta;
            p_jac[row * n_int + col].j10 = dydxi;
            p_jac[row * n_int + col].j11 = dydeta;
            const double det = dxdxi * dydeta - dxdeta * dydxi;
            p_jac[row * n_int + col].det = det;
        }
    }
    out->jacobian = p_jac;

    return 1;
}

static int create_m0(precompute_t *this, const allocator_callbacks *allocator)
{
    const basis_precomp_t *basis = this->basis;
    // Allocate nodal mass matrix
    matrix_full_t mtx_nodal = {
        .base = {.type = MATRIX_TYPE_FULL,
                 .rows = (basis->order + 1) * (basis->order + 1),
                 .cols = (basis->order + 1) * (basis->order + 1)},
    };
    mtx_nodal.data = allocate(allocator, sizeof *mtx_nodal.data * mtx_nodal.base.rows * mtx_nodal.base.cols);
    if (!mtx_nodal.data)
    {
        return 0;
    }
    const unsigned n_int = basis->n_int;
    const jacobian_t *p_jac = this->jacobian;

    // Compute nodal mass matrix
    for (unsigned row = 0; row < mtx_nodal.base.rows; ++row)
    {
        for (unsigned col = 0; col <= row; ++col)
        {
            const unsigned offset = (row * mtx_nodal.base.cols + col) * n_int * n_int;
            double val = 0.0;
            for (unsigned i = 0; i < n_int; ++i)
            {
                for (unsigned j = 0; j < n_int; ++j)
                {
                    val += p_jac[i * n_int + j].det * basis->mass_nodal[offset + i * n_int + j];
                }
            }
            // Use symmetry
            mtx_nodal.data[row * mtx_nodal.base.cols + col] = val;
            mtx_nodal.data[col * mtx_nodal.base.cols + row] = val;
        }
    }
    this->mass_matrices[MASS_0] = mtx_nodal;
    return 1;
}

static int create_m1(precompute_t *this, const allocator_callbacks *allocator)
{
    const basis_precomp_t *basis = this->basis;

    // Allocate edge mass matrix
    matrix_full_t mtx_edge = {
        .base = {.type = MATRIX_TYPE_FULL,
                 .rows = 2 * basis->order * (basis->order + 1),
                 .cols = 2 * basis->order * (basis->order + 1)},
    };
    mtx_edge.data = allocate(allocator, sizeof *mtx_edge.data * mtx_edge.base.rows * mtx_edge.base.cols);
    if (!mtx_edge.data)
    {
        return 0;
    }
    const unsigned n_int = basis->n_int;
    const jacobian_t *p_jac = this->jacobian;

    // Compute edge mass matrix (00) part
    const unsigned n_edge = basis->order * (basis->order + 1);
    for (unsigned row = 0; row < n_edge; ++row)
    {
        for (unsigned col = 0; col <= row; ++col)
        {
            const unsigned offset = (row * n_edge + col) * n_int * n_int;
            double val = 0.0;
            for (unsigned i = 0; i < n_int; ++i)
            {
                for (unsigned j = 0; j < n_int; ++j)
                {
                    const double jac_term = p_jac[i * n_int + j].j11 * p_jac[i * n_int + j].j11 +
                                            p_jac[i * n_int + j].j01 * p_jac[i * n_int + j].j01;
                    val += basis->mass_edge_00[offset + i * n_int + j] * jac_term / p_jac[i * n_int + j].det;
                }
            }
            // Use symmetry
            mtx_edge.data[row * mtx_edge.base.cols + col] = val;
            mtx_edge.data[col * mtx_edge.base.cols + row] = val;
        }
    }

    // Compute edge mass matrix (01) part
    for (unsigned row = 0; row < n_edge; ++row)
    {
        for (unsigned col = 0; col < n_edge; ++col)
        {
            const unsigned offset = (row * n_edge + col) * n_int * n_int;
            double val = 0.0;
            for (unsigned i = 0; i < n_int; ++i)
            {
                for (unsigned j = 0; j < n_int; ++j)
                {
                    const double jac_term = p_jac[i * n_int + j].j10 * p_jac[i * n_int + j].j11 +
                                            p_jac[i * n_int + j].j00 * p_jac[i * n_int + j].j01;
                    val += basis->mass_edge_01[offset + i * n_int + j] * jac_term / p_jac[i * n_int + j].det;
                }
            }
            // Use symmetry
            mtx_edge.data[(n_edge + row) * mtx_edge.base.cols + col] = val;
            mtx_edge.data[col * mtx_edge.base.cols + (n_edge + row)] = val;
        }
    }

    // Compute edge mass matrix (11) part
    for (unsigned row = 0; row < n_edge; ++row)
    {
        for (unsigned col = 0; col <= row; ++col)
        {
            const unsigned offset = (row * n_edge + col) * n_int * n_int;
            double val = 0.0;
            for (unsigned i = 0; i < n_int; ++i)
            {
                for (unsigned j = 0; j < n_int; ++j)
                {
                    const double jac_term = p_jac[i * n_int + j].j10 * p_jac[i * n_int + j].j10 +
                                            p_jac[i * n_int + j].j00 * p_jac[i * n_int + j].j00;
                    val += basis->mass_edge_11[offset + i * n_int + j] * jac_term / p_jac[i * n_int + j].det;
                }
            }
            // Use symmetry
            mtx_edge.data[(n_edge + row) * mtx_edge.base.cols + (n_edge + col)] = val;
            mtx_edge.data[(n_edge + col) * mtx_edge.base.cols + (n_edge + row)] = val;
        }
    }

    this->mass_matrices[MASS_1] = mtx_edge;
    return 1;
}

static int create_m2(precompute_t *this, const allocator_callbacks *allocator)
{
    const basis_precomp_t *basis = this->basis;

    // Allocate surface mass matrix
    matrix_full_t mtx_surf = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = basis->order * basis->order, .cols = basis->order * basis->order},
    };
    mtx_surf.data = allocate(allocator, sizeof *mtx_surf.data * mtx_surf.base.rows * mtx_surf.base.cols);
    if (!mtx_surf.data)
    {
        return 0;
    }

    const unsigned n_int = basis->n_int;
    const jacobian_t *p_jac = this->jacobian;

    // Compute surface mass matrix
    for (unsigned row = 0; row < mtx_surf.base.rows; ++row)
    {
        for (unsigned col = 0; col <= row; ++col)
        {
            const unsigned offset = (row * mtx_surf.base.cols + col) * n_int * n_int;
            double val = 0.0;
            for (unsigned i = 0; i < n_int; ++i)
            {
                for (unsigned j = 0; j < n_int; ++j)
                {
                    val += basis->mass_surf[offset + i * n_int + j] / p_jac[i * n_int + j].det;
                }
            }

            // Use symmetry
            mtx_surf.data[row * mtx_surf.base.cols + col] = val;
            mtx_surf.data[col * mtx_surf.base.cols + row] = val;
        }
    }

    this->mass_matrices[MASS_2] = mtx_surf;
    return 1;
}

static int create_im0(precompute_t *this, const allocator_callbacks *allocator)
{
    if (this->mass_matrices[MASS_0].data == NULL && !create_m0(this, allocator))
    {
        return 0;
    }
    const basis_precomp_t *basis = this->basis;

    // Allocate inverse nodal mass matrix
    matrix_full_t mtx_inv_nodal = {
        .base = {.type = MATRIX_TYPE_FULL,
                 .rows = (basis->order + 1) * (basis->order + 1),
                 .cols = (basis->order + 1) * (basis->order + 1)},
    };
    mtx_inv_nodal.data =
        allocate(allocator, sizeof *mtx_inv_nodal.data * mtx_inv_nodal.base.rows * mtx_inv_nodal.base.cols);
    if (!mtx_inv_nodal.data)
    {
        return 0;
    }

    double *const restrict buffer =
        allocate(allocator, sizeof *mtx_inv_nodal.data * mtx_inv_nodal.base.rows * mtx_inv_nodal.base.cols);
    if (!buffer)
    {
        deallocate(allocator, mtx_inv_nodal.data);
        return 0;
    }

    invert_matrix(this->mass_matrices[MASS_0].base.rows, this->mass_matrices[MASS_0].data, buffer, mtx_inv_nodal.data);
    deallocate(allocator, buffer);

    this->mass_matrices[MASS_0_I] = mtx_inv_nodal;
    return 1;
}

static int create_im1(precompute_t *this, const allocator_callbacks *allocator)
{
    if (this->mass_matrices[MASS_1].data == NULL && !create_m1(this, allocator))
    {
        return 0;
    }
    const basis_precomp_t *basis = this->basis;

    // Allocate edge mass matrix
    matrix_full_t mtx_inv_edge = {
        .base = {.type = MATRIX_TYPE_FULL,
                 .rows = 2 * basis->order * (basis->order + 1),
                 .cols = 2 * basis->order * (basis->order + 1)},
    };
    mtx_inv_edge.data =
        allocate(allocator, sizeof *mtx_inv_edge.data * mtx_inv_edge.base.rows * mtx_inv_edge.base.cols);
    if (!mtx_inv_edge.data)
    {
        return 0;
    }

    double *const restrict buffer =
        allocate(allocator, sizeof *mtx_inv_edge.data * mtx_inv_edge.base.rows * mtx_inv_edge.base.cols);
    if (!buffer)
    {
        deallocate(allocator, mtx_inv_edge.data);
        return 0;
    }

    invert_matrix(this->mass_matrices[MASS_1].base.rows, this->mass_matrices[MASS_1].data, buffer, mtx_inv_edge.data);
    deallocate(allocator, buffer);

    this->mass_matrices[MASS_1_I] = mtx_inv_edge;
    return 1;
}

static int create_im2(precompute_t *this, const allocator_callbacks *allocator)
{
    if (this->mass_matrices[MASS_2].data == NULL && !create_m2(this, allocator))
    {
        return 0;
    }
    const basis_precomp_t *basis = this->basis;

    // Allocate surface mass matrix
    matrix_full_t mtx_inv_surf = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = basis->order * basis->order, .cols = basis->order * basis->order},
    };
    mtx_inv_surf.data =
        allocate(allocator, sizeof *mtx_inv_surf.data * mtx_inv_surf.base.rows * mtx_inv_surf.base.cols);
    if (!mtx_inv_surf.data)
    {
        return 0;
    }

    double *const restrict buffer =
        allocate(allocator, sizeof *mtx_inv_surf.data * mtx_inv_surf.base.rows * mtx_inv_surf.base.cols);
    if (!buffer)
    {
        deallocate(allocator, mtx_inv_surf.data);
        return 0;
    }

    invert_matrix(this->mass_matrices[MASS_2].base.rows, this->mass_matrices[MASS_2].data, buffer, mtx_inv_surf.data);
    deallocate(allocator, buffer);

    this->mass_matrices[MASS_2_I] = mtx_inv_surf;
    return 1;
}

const matrix_full_t *precompute_get_matrix(precompute_t *this, mass_mtx_indices_t idx,
                                           const allocator_callbacks *allocator)
{
    matrix_full_t *const p = this->mass_matrices + idx;
    if (p->data == NULL)
    {
        static const int (*const callbacks[MASS_CNT])(precompute_t *this, const allocator_callbacks *allocator) = {
            [MASS_0] = create_m0,    [MASS_1] = create_m1,    [MASS_2] = create_m2,
            [MASS_0_I] = create_im0, [MASS_1_I] = create_im1, [MASS_2_I] = create_im2,
        };
        if (!callbacks[idx](this, allocator))
            return NULL;
    }

    return p;
}
