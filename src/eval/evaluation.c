//
// Created by jan on 15.2.2025.
//

#include "evaluation.h"
#include "incidence.h"

#define MATRIX_OP_ENTRY(op) [op] = #op
static const char *matrix_op_strings[MATOP_COUNT] = {
    MATRIX_OP_ENTRY(MATOP_INVALID),   MATRIX_OP_ENTRY(MATOP_IDENTITY), MATRIX_OP_ENTRY(MATOP_MASS),
    MATRIX_OP_ENTRY(MATOP_INCIDENCE), MATRIX_OP_ENTRY(MATOP_PUSH),     MATRIX_OP_ENTRY(MATOP_SCALE),
    MATRIX_OP_ENTRY(MATOP_TRANSPOSE), MATRIX_OP_ENTRY(MATOP_SUM),
};
#undef MATRIX_OP_ENTRY

const char *matrix_op_str(const matrix_op_t op)
{
    if (op >= MATOP_COUNT)
        return "UNKNOWN";
    return matrix_op_strings[op];
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
int convert_bytecode(const unsigned n, bytecode_val_t bytecode[restrict n + 1], PyObject *items[static n],
                     unsigned *p_max_stack)
{
    bytecode[0].u32 = n;
    unsigned stack_load = 0, max_load = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const long val = PyLong_AsLong(items[i]);
        if (PyErr_Occurred())
        {
            return 0;
        }
        if (val <= MATOP_INVALID || val >= MATOP_COUNT)
        {
            PyErr_Format(PyExc_ValueError, "Invalid operation code %ld at position %zu.", val, i);
            return 0;
        }

        const matrix_op_t op = (matrix_op_t)val;
        bytecode[i + 1].op = op;

        int out_of_bounds = 0, bad_value = 0;
        switch (op)
        {
        case MATOP_IDENTITY:
            break;

        case MATOP_PUSH:
            stack_load += 1;
            if (stack_load > max_load)
            {
                max_load = stack_load;
            }
            break;

        case MATOP_TRANSPOSE:
            break;

        case MATOP_MATMUL:
            if (stack_load == 0)
            {
                PyErr_SetString(PyExc_ValueError, "Matmul instruction with nothing on the stack.");
                return 0;
            }
            stack_load -= 1;
            break;

        case MATOP_SCALE:
            if (n - i < 1)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].f64 = PyFloat_AsDouble(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        case MATOP_SUM:
            if (n - i < 1)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                const size_t n_sum = PyLong_AsUnsignedLong(items[i]);
                bytecode[i + 1].u32 = n_sum;
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                if (n_sum > stack_load)
                {
                    PyErr_Format(PyExc_ValueError, "Sum instruction for %zu matrices, but only %u are on stack", n_sum,
                                 stack_load);
                    return 0;
                }
                stack_load -= 1;
            }
            break;

        case MATOP_INCIDENCE:
            if (n - i < 2)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        case MATOP_MASS:
            if (n - i < 2)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        default:
            PyErr_Format(PyExc_ValueError, "Invalid error code %u.", (unsigned)op);
            return 0;
        }

        if (out_of_bounds)
        {
            PyErr_Format(PyExc_ValueError, "Out of bounds for the required item.");
            return 0;
        }

        if (bad_value)
        {
            return 0;
        }
    }
    *p_max_stack = max_load;
    return 1;
}

INTERPLIB_INTERNAL
int system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                           const allocator_callbacks *allocator)
{
    // Find number of forms
    {
        PyArrayObject *const order_array = (PyArrayObject *)PyArray_FromAny(
            orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
        if (!order_array)
            return 0;

        const unsigned n_forms = PyArray_DIM(order_array, 0);
        this->n_forms = n_forms;
        this->form_orders = allocate(allocator, sizeof(*this->form_orders) * n_forms);
        const unsigned *restrict p_o = PyArray_DATA(order_array);
        for (unsigned i = 0; i < n_forms; ++i)
        {
            const unsigned o = p_o[i];
            if (o > 2)
            {
                PyErr_Format(PyExc_ValueError, "Form can not be of order higher than 2 (it was %u)", o);
                Py_DECREF(order_array);
                return 0;
            }
            this->form_orders[i] = o + 1;
        }
        Py_DECREF(order_array);
    }

    // Now go though the rows
    ssize_t row_count = PySequence_Size(expr_matrix);
    if (row_count < 0)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }

    if (row_count != this->n_forms)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of forms deduced from order array (%u) does not match the number of expression rows (%u).",
                     this->n_forms, row_count);
        deallocate(allocator, this->form_orders);
        return 0;
    }

    this->bytecodes = allocate(allocator, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    if (!this->bytecodes)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }
    memset(this->bytecodes, 0, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    unsigned max_stack = 1;
    for (unsigned row = 0; row < this->n_forms; ++row)
    {
        PyObject *row_expr = PySequence_GetItem(expr_matrix, row);
        if (!row_expr)
        {
            goto failed_row;
        }
        row_count = PySequence_Size(row_expr);
        if (row_count < 0)
        {
            goto failed_row;
        }
        if (row_count != this->n_forms)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Number of forms deduced from order array (%u) does not match the number of expression in row %u (%u).",
                this->n_forms, row, row_count);
            goto failed_row;
        }

        for (unsigned col = 0; col < this->n_forms; ++col)
        {
            PyObject *expr = PySequence_GetItem(row_expr, col);
            if (!expr)
            {
                goto failed_row;
            }
            if (Py_IsNone(expr))
            {
                Py_DECREF(expr);
                continue;
            }
            PyObject *seq = PySequence_Fast(expr, "Bytecode must be a given as a sequence.");
            if (!seq)
            {
                Py_DECREF(expr);
                goto failed_row;
            }
            row_count = PySequence_Fast_GET_SIZE(seq);
            if (row_count < 0)
            {
                Py_DECREF(expr);
                goto failed_row;
            }

            bytecode_val_t *bc = allocate(allocator, sizeof(**this->bytecodes) * (row_count + 1));
            if (!bc)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            this->bytecodes[row * this->n_forms + col] = bc;
            unsigned stack;
            if (!convert_bytecode(row_count, bc, PySequence_Fast_ITEMS(seq), &stack))
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            if (stack > max_stack)
            {
                max_stack = stack;
            }
            Py_DECREF(seq);
            Py_DECREF(expr);
        }

        continue;

    failed_row: {
        Py_XDECREF(row_expr);
        for (unsigned i = row; i > 0; --i)
        {
            for (unsigned j = this->n_forms; j > 0; --j)
            {
                deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
            }
        }
        deallocate(allocator, this->bytecodes);
        deallocate(allocator, this->form_orders);
        Py_DECREF(row_expr);
        return 0;
    }
    }

    this->max_stack = max_stack;

    return 1;
}

void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator)
{
    deallocate(allocator, this->form_orders);
    for (unsigned i = this->n_forms; i > 0; --i)
    {
        for (unsigned j = this->n_forms; j > 0; --j)
        {
            deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
        }
    }
    *this = (system_template_t){};
}

int precompute_create(const basis_precomp_t *basis, double x0, double x1, double x2, double x3, double y0, double y1,
                      double y2, double y3, precompute_t *out, allocator_callbacks *allocator)
{
    *out = (precompute_t){};

    const unsigned n_int = basis->n_int;

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

    // Allocate edge mass matrix
    matrix_full_t mtx_edge = {
        .base = {.type = MATRIX_TYPE_FULL,
                 .rows = 2 * basis->order * (basis->order + 1),
                 .cols = 2 * basis->order * (basis->order + 1)},
    };
    mtx_edge.data = allocate(allocator, sizeof *mtx_edge.data * mtx_edge.base.rows * mtx_edge.base.cols);
    if (!mtx_edge.data)
    {
        deallocate(allocator, mtx_nodal.data);
        return 0;
    }

    // Allocate surface mass matrix
    matrix_full_t mtx_surf = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = basis->order * basis->order, .cols = basis->order * basis->order},
    };
    mtx_surf.data = allocate(allocator, sizeof *mtx_surf.data * mtx_surf.base.rows * mtx_surf.base.cols);
    if (!mtx_surf.data)
    {
        deallocate(allocator, mtx_edge.data);
        deallocate(allocator, mtx_nodal.data);
        return 0;
    }

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
        deallocate(allocator, mtx_surf.data);
        deallocate(allocator, mtx_edge.data);
        deallocate(allocator, mtx_nodal.data);
        return 0;
    }

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
        deallocate(allocator, mtx_inv_nodal.data);
        deallocate(allocator, mtx_surf.data);
        deallocate(allocator, mtx_edge.data);
        deallocate(allocator, mtx_nodal.data);
        return 0;
    }

    // Allocate surface mass matrix
    matrix_full_t mtx_inv_surf = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = basis->order * basis->order, .cols = basis->order * basis->order},
    };
    mtx_inv_surf.data =
        allocate(allocator, sizeof *mtx_inv_surf.data * mtx_inv_surf.base.rows * mtx_inv_surf.base.cols);
    if (!mtx_inv_surf.data)
    {
        deallocate(allocator, mtx_inv_edge.data);
        deallocate(allocator, mtx_inv_nodal.data);
        deallocate(allocator, mtx_surf.data);
        deallocate(allocator, mtx_edge.data);
        deallocate(allocator, mtx_nodal.data);
        return 0;
    }

    struct
    {
        double j00, j01, j10, j11, det;
    } *restrict p_jac;
    const size_t sz_jac = sizeof *p_jac * (n_int * n_int);
    p_jac = allocate(allocator, sz_jac);
    if (!p_jac)
    {
        deallocate(allocator, mtx_inv_surf.data);
        deallocate(allocator, mtx_inv_edge.data);
        deallocate(allocator, mtx_inv_nodal.data);
        deallocate(allocator, mtx_surf.data);
        deallocate(allocator, mtx_edge.data);
        deallocate(allocator, mtx_nodal.data);

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

    size_t sz_inv;
    if (mtx_nodal.base.rows > mtx_edge.base.rows)
    {
        if (mtx_nodal.base.rows > mtx_surf.base.rows)
        {
            sz_inv = mtx_nodal.base.rows;
        }
        else
        {
            sz_inv = mtx_surf.base.rows;
        }
    }
    else if (mtx_surf.base.rows > mtx_edge.base.rows)
    {
        sz_inv = mtx_surf.base.rows;
    }
    else
    {
        sz_inv = mtx_edge.base.rows;
    }
    sz_inv = sz_inv * sz_inv * sizeof(*mtx_nodal.data);
    if (sz_jac < sz_inv)
    {
        deallocate(allocator, p_jac);
        p_jac = allocate(allocator, sz_inv);
        if (!p_jac)
        {
            deallocate(allocator, mtx_inv_surf.data);
            deallocate(allocator, mtx_inv_edge.data);
            deallocate(allocator, mtx_inv_nodal.data);
            deallocate(allocator, mtx_surf.data);
            deallocate(allocator, mtx_edge.data);
            deallocate(allocator, mtx_nodal.data);
            return 0;
        }
    }

    invert_matrix(mtx_nodal.base.rows, mtx_nodal.data, (double *)p_jac, mtx_inv_nodal.data);
    invert_matrix(mtx_edge.base.rows, mtx_edge.data, (double *)p_jac, mtx_inv_edge.data);
    invert_matrix(mtx_surf.base.rows, mtx_surf.data, (double *)p_jac, mtx_inv_surf.data);

    out->mass_matrices[MASS_0] = mtx_nodal;
    out->mass_matrices[MASS_1] = mtx_edge;
    out->mass_matrices[MASS_2] = mtx_surf;
    out->mass_matrices[MASS_0_I] = mtx_inv_nodal;
    out->mass_matrices[MASS_1_I] = mtx_inv_edge;
    out->mass_matrices[MASS_2_I] = mtx_inv_surf;

    return 1;
}

PyArrayObject *matrix_full_to_array(const matrix_full_t *mat)
{
    const npy_intp dims[2] = {mat->base.rows, mat->base.cols};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;

    double *const restrict p_out = PyArray_DATA(out);
    memcpy(p_out, mat->data, sizeof(*p_out) * mat->base.rows * mat->base.cols);
    return out;
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

static eval_result_t incidence_to_full(const incidence_type_t type, const unsigned order, matrix_full_t *p_out,
                                       const allocator_callbacks *allocator)
{
    unsigned n_row, n_col;
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_1, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_0, order);
        break;
    case INCIDENCE_TYPE_10_T:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_0, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_1, order);
        break;
    case INCIDENCE_TYPE_21:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_2, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_1, order);
        break;
    case INCIDENCE_TYPE_21_T:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_1, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_2, order);
        break;
    default:
        return 0;
    }

    const matrix_full_t this = {.base = {.type = MATRIX_TYPE_FULL, .rows = n_row, .cols = n_col},
                                .data = allocate(allocator, n_row * n_col * sizeof(*this.data))};
    if (!this.data)
        return 0;
    memset(this.data, 0, sizeof(*this.data) * this.base.rows * this.base.cols);
    switch (type) // Only going to cover types we can encounter.
    {
    case INCIDENCE_TYPE_10:
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[(row * order + col) * this.base.cols + (order + 1) * row + col] = +1.0;
                this.data[(row * order + col) * this.base.cols + (order + 1) * row + col + 1] = -1.0;
            }
        }
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                this.data[((order + 1) * order + row * (order + 1) * col) * this.base.cols + (order + 1) * row + col] =
                    -1.0;
                this.data[((order + 1) * order + row * (order + 1) * col) * this.base.cols + (order + 1) * (row + 1) +
                          col] = +1.0;
            }
        }
        break;
    case INCIDENCE_TYPE_10_T:
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[((order + 1) * row + col) * this.base.cols + row * order + col] = +1.0;
                this.data[((order + 1) * row + col + 1) * this.base.cols + row * order + col] = -1.0;
            }
        }
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                this.data[((order + 1) * row + col) * this.base.cols + (order + 1) * order + row * (order + 1) * col] =
                    -1.0;
                this.data[((order + 1) * (row + 1) + col) * this.base.cols + (order + 1) * order +
                          row * (order + 1) * col] = +1.0;
            }
        }
        break;

    case INCIDENCE_TYPE_21:
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[(row * order + col) * this.base.cols + (row * order + col)] = +1.0;
                this.data[(row * order + col) * this.base.cols + ((row + 1) * order + col)] = -1.0;
                this.data[(row * order + col) * this.base.cols + order * (order + 1) + (row * (order + 1) + col)] =
                    +1.0;
                this.data[(row * order + col) * this.base.cols + order * (order + 1) + (row * (order + 1) + col + 1)] =
                    -1.0;
            }
        }
        break;

    case INCIDENCE_TYPE_21_T:
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[((row * order + col)) * this.base.cols + row * order + col] = +1.0;
                this.data[(((row + 1) * order + col)) * this.base.cols + row * order + col] = -1.0;
                this.data[(order * (order + 1) + (row * (order + 1) + col)) * this.base.cols + row * order + col] =
                    +1.0;
                this.data[(order * (order + 1) + (row * (order + 1) + col + 1)) * this.base.cols + row * order + col] =
                    -1.0;
            }
        }
        break;
    }

    *p_out = this;
    return 1;
}

static eval_result_t apply_incidence_to_full_left(const incidence_type_t type, const unsigned order,
                                                  const matrix_full_t *in, matrix_full_t *p_out,
                                                  const allocator_callbacks *allocator)
{
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        return apply_e10_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_10_T:
        return apply_e10t_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21:
        return apply_e21_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21_T:
        return apply_e21t_left(order, in, p_out, allocator);
    default:
        return EVAL_BAD_ENUM;
    }
}

static eval_result_t apply_incidence_to_full_right(const incidence_type_t type, const unsigned order,
                                                   const matrix_full_t *in, matrix_full_t *p_out,
                                                   const allocator_callbacks *allocator)
{
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        return apply_e10_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_10_T:
        return apply_e10t_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21:
        return apply_e21_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21_T:
        return apply_e21t_right(order, in, p_out, allocator);
    default:
        return EVAL_BAD_ENUM;
    }
}

static eval_result_t matrix_full_copy(const matrix_full_t *this, matrix_full_t *out,
                                      const allocator_callbacks *allocator)
{
    double *const restrict ptr = allocate(allocator, sizeof(*ptr) * this->base.rows * this->base.cols);
    if (!ptr)
        return EVAL_FAILED_ALLOC;
    memcpy(ptr, this->data, sizeof(*ptr) * this->base.rows * this->base.cols);
    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = this->base.rows, .cols = this->base.cols},
                           .data = ptr};
    return EVAL_SUCCESS;
}

static eval_result_t matrix_full_multiply(const matrix_full_t *left, const matrix_full_t *right, matrix_full_t *out,
                                          const allocator_callbacks *allocator)
{
    if (left->base.cols != right->base.rows)
        return EVAL_DIMS_MISMATCH;

    const unsigned n = right->base.rows;

    const unsigned n_rows = left->base.rows;
    const unsigned n_cols = right->base.cols;
    double *const restrict ptr = allocate(allocator, sizeof(*ptr) * n_rows * n_cols);
    const matrix_full_t this = {.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};

    for (unsigned row = 0; row < n_rows; ++row)
    {
        for (unsigned col = 0; col < n_cols; ++col)
        {
            double v = 0.0;
            for (unsigned k = 0; k < n; ++k)
            {
                v += left->data[row * n + k] * right->data[k * n_cols + col];
            }
            ptr[row * n_cols + col] = v;
        }
    }

    *out = this;
    return EVAL_SUCCESS;
}

static eval_result_t matrix_full_add_inplace(const matrix_full_t *in, matrix_full_t *out)
{
    if (in->base.cols != out->base.cols || in->base.rows != out->base.rows)
        return EVAL_DIMS_MISMATCH;

    for (unsigned row = 0; row < in->base.rows; ++row)
    {
        for (unsigned col = 0; col < in->base.cols; ++col)
        {
            out->data[row * out->base.cols + col] += out->data[row * out->base.cols + col];
        }
    }

    return EVAL_SUCCESS;
}

static eval_result_t matrix_multiply(const unsigned order, const matrix_t *right, const matrix_t *left, matrix_t *out,
                                     const allocator_callbacks *allocator)
{
    double k_right = right->coefficient, k_left = left->coefficient;
    eval_result_t res = EVAL_SUCCESS;
    switch (right->type)
    {
    case MATRIX_TYPE_IDENTITY:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY};
            break;

        case MATRIX_TYPE_INCIDENCE:
            *out = *left;
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_copy((matrix_full_t *)left, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INVALID:
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY};
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_INCIDENCE:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = *right;
            break;

        case MATRIX_TYPE_INCIDENCE: {
            matrix_full_t tmp;
            if ((res = incidence_to_full(right->incidence.incidence, order, &tmp, allocator)) != EVAL_SUCCESS)
                return res;
            res = apply_incidence_to_full_left(left->incidence.incidence, order, &tmp, (matrix_full_t *)out, allocator);
            deallocate(allocator, tmp.data);
        }
        break;

        case MATRIX_TYPE_FULL:
            res = apply_incidence_to_full_right(right->incidence.incidence, order, &left->full, (matrix_full_t *)out,
                                                allocator);
            break;

        case MATRIX_TYPE_INVALID:
            *out = *right;
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_FULL:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            res = matrix_full_copy(&right->full, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INCIDENCE:
            res = apply_incidence_to_full_left(left->incidence.incidence, order, &right->full, (matrix_full_t *)out,
                                               allocator);
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_multiply(&left->full, &right->full, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INVALID:
            res = matrix_full_copy(&right->full, (matrix_full_t *)out, allocator);
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_INVALID:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = *left;
            k_right = 1.0;
            break;

        case MATRIX_TYPE_INCIDENCE:
            *out = *left;
            k_right = 1.0;
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_copy((matrix_full_t *)left, (matrix_full_t *)out, allocator);
            k_right = 1.0;
            break;

        case MATRIX_TYPE_INVALID:
            return EVAL_BAD_ENUM;
        }
        break;
    }
    out->coefficient = k_right * k_left;
    return res;
}

static void matrix_multiply_inplace(const matrix_full_t *this, const double k)
{
    if (k == 1.0)
        return;
    for (unsigned i = 0; i < this->base.rows * this->base.cols; ++i)
    {
        this->data[i] *= k;
    }
}

static void matrix_add_diagonal_inplace(const matrix_full_t *this, const double k)
{
    if (k == 0)
        return;
    const unsigned n = this->base.cols > this->base.rows ? this->base.rows : this->base.cols;
    for (unsigned i = 0; i < n; ++i)
    {
        this->data[i * this->base.cols + i] += k;
    }
}

INTERPLIB_INTERNAL
void matrix_cleanup(matrix_t *this, const allocator_callbacks *allocator)
{
    if (this->type == MATRIX_TYPE_FULL)
    {
        deallocate(allocator, this->full.data);
    }
    *this = (matrix_t){};
}

static eval_result_t matrix_add(const unsigned order, matrix_t *right, matrix_t *left, matrix_t *out,
                                const allocator_callbacks *allocator)
{
    eval_result_t res = EVAL_SUCCESS;
    if (left->type == MATRIX_TYPE_INVALID || right->type == MATRIX_TYPE_INVALID)
        return EVAL_BAD_ENUM;

    int free_left = 1, free_right = 1;
    if (left->type == MATRIX_TYPE_FULL)
    {
        free_left = 0;
        matrix_multiply_inplace(&left->full, left->coefficient);
        left->coefficient = 1.0;
    }

    if (right->type == MATRIX_TYPE_FULL)
    {
        free_right = 0;
        matrix_multiply_inplace(&right->full, right->coefficient);
        right->coefficient = 1.0;
    }

    if (left->type == right->type)
    {
        if (left->type == MATRIX_TYPE_IDENTITY)
        {
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY, .coefficient = left->coefficient + right->coefficient};
            return EVAL_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_INCIDENCE && left->incidence.incidence == right->incidence.incidence)
        {
            *out = (matrix_t){.type = MATRIX_TYPE_INCIDENCE,
                              .incidence.incidence = left->incidence.incidence,
                              .coefficient = left->coefficient + right->coefficient};
            return EVAL_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_FULL)
        {
            return matrix_full_multiply(&left->full, &right->full, &out->full, allocator);
        }
    }

    if (!free_left && !free_right)
    {
        return EVAL_WRONG_MAT_TYPES;
    }

    matrix_full_t full_old;
    matrix_t *to_convert;
    if (free_left)
    {
        to_convert = left;
        full_old = right->full;
    }
    else
    {
        to_convert = right;
        full_old = left->full;
    }

    if (to_convert->type == MATRIX_TYPE_IDENTITY)
    {
        if (full_old.base.rows != full_old.base.cols)
            return EVAL_DIMS_MISMATCH;

        res = matrix_full_copy(&full_old, &out->full, allocator);
        if (res != EVAL_SUCCESS)
            return res;
        matrix_add_diagonal_inplace(&out->full, to_convert->coefficient);
    }

    // to_convert->type == MATRIX_TYPE_INCIDENCE
    res = incidence_to_full(to_convert->incidence.incidence, order, &out->full, allocator);
    if (res != EVAL_SUCCESS)
        return res;
    res = matrix_full_add_inplace(&full_old, &out->full);
    out->coefficient = 1.0;

    return res;
}

static void clean_stack(matrix_t *stack, unsigned cnt, const allocator_callbacks *allocator)
{
    for (unsigned i = cnt; i > 0; ++i)
    {
        matrix_cleanup(stack + i - 1, allocator);
    }
}

INTERPLIB_INTERNAL
int evaluate_element_term(form_order_t form, unsigned order, const bytecode_val_t *code, precompute_t *precomp,
                          unsigned n_stack, matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                          matrix_full_t *p_out)
{
    const unsigned n_ops = code[0].u32;
    code += 1;

    unsigned stack_pos = 0;
    matrix_t current = {
        .type = MATRIX_TYPE_INVALID,
        .coefficient = 0.0,
    };

    for (unsigned i = 0; i < n_ops; ++i)
    {
        const matrix_op_t op = code[i].op;
        unsigned remaining = n_ops - i - 1;
        switch (op)
        {
        case MATOP_IDENTITY:
            if (current.type == MATRIX_TYPE_INVALID)
            {
                current.type = MATRIX_TYPE_IDENTITY;
                current.coefficient = 1.0;
            }
            break;

        case MATOP_SCALE:
            if (remaining < 1)
            {
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            i += 1;
            if (current.type == MATRIX_TYPE_INVALID)
            {
                current.type = MATRIX_TYPE_IDENTITY;
                current.coefficient = code[i].f64;
            }
            else
            {
                current.coefficient *= code[i].f64;
            }
            break;

        case MATOP_TRANSPOSE: {
            const unsigned tmp = current.base.rows;
            current.base.rows = current.base.cols;
            current.base.cols = tmp;
        }
            switch (current.type)
            {
            case MATRIX_TYPE_INCIDENCE: {
                incidence_type_t t = current.incidence.incidence;
                if (t <= INCIDENCE_TYPE_INVALID || t >= INCIDENCE_TYPE_CNT)
                {
                    return 0;
                }
                // Shift E10 0 and
                t = ((t - 1) ^ 1) + 1;
                current.incidence.incidence = t;
            }
            break;

            case MATRIX_TYPE_FULL: {
                /*TODO: Transpose the full matrix*/
            }
            break;

            case MATRIX_TYPE_IDENTITY:
                /*NO-OP*/
                break;

            default:
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_BAD_ENUM;
            }
            break;

        case MATOP_PUSH:
            if (stack_pos == n_stack)
            {
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_STACK_OVERFLOW;
            }
            stack[stack_pos] = current;
            stack_pos += 1;
            current = (matrix_t){
                .type = MATRIX_TYPE_INVALID,
                .coefficient = 0.0,
            };
            break;

        case MATOP_INCIDENCE:
            if (remaining < 2)
            {
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                incidence_type_t t = code[i].u32;
                i += 1;
                const unsigned dual = code[i].u32;
                if (t <= INCIDENCE_TYPE_INVALID || t >= INCIDENCE_TYPE_CNT)
                {
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }
                if (dual)
                {
                    t = ((t - 1) ^ 1) + 1;
                }

                switch (current.type)
                {
                case MATRIX_TYPE_INVALID:
                    current =
                        (matrix_t){.type = MATRIX_TYPE_INCIDENCE, .incidence = {.incidence = t}, .coefficient = 1.0};
                    break;
                case MATRIX_TYPE_IDENTITY:
                    current = (matrix_t){.type = MATRIX_TYPE_INCIDENCE,
                                         .incidence = {.incidence = t},
                                         .coefficient = current.coefficient};
                    break;

                    matrix_t new_mat;
                    eval_result_t res;

                case MATRIX_TYPE_INCIDENCE:
                    res = incidence_to_full(current.type, order, &current.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    // FALLTHROUGH

                case MATRIX_TYPE_FULL:
                    res = apply_incidence_to_full_left(t, order, &current.full, &new_mat.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    new_mat.coefficient = current.coefficient;
                    matrix_cleanup(&current, allocator);
                    current = new_mat;
                    break;
                }
                if (dual)
                {
                    current.coefficient *= -1;
                }
                // matrix_t this = { .incidence={.base = {.type = MATRIX_TYPE_INCIDENCE}, .incidence = t}, .coefficient
                // = 1.0}; matrix_t new_mat; const eval_result_t res = matrix_multiply(order, &current, &this, &new_mat,
                // allocator); matrix_cleanup(&current, allocator); if (res != EVAL_SUCCESS)
                // {
                //     return res;
                // }
                // current = new_mat;
            }
            break;

        case MATOP_MASS:
            if (remaining < 2)
            {
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                mass_mtx_indices_t t = code[i].u32;
                i += 1;
                const unsigned inverse = code[i].u32;
                if (t < MASS_0 || t & 1 || t >= MASS_CNT)
                {
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }
                if (inverse)
                {
                    t |= 1;
                }
                matrix_t this = {.type = MATRIX_TYPE_FULL, .coefficient = 1.0};
                this.full = precomp->mass_matrices[t];
                eval_result_t res;
                switch (current.type)
                {
                case MATRIX_TYPE_INVALID:
                    current.coefficient = 1.0;
                    // FALLTHROUGH
                case MATRIX_TYPE_IDENTITY:
                    res = matrix_full_copy(&this.full, &current.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    break;
                case MATRIX_TYPE_INCIDENCE:
                    res = apply_incidence_to_full_right(current.incidence.incidence, order, &this.full, &current.full,
                                                        allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    break;
                    matrix_t new_mat;
                case MATRIX_TYPE_FULL:
                    res = matrix_full_multiply(&this.full, &current.full, &new_mat.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    matrix_cleanup(&current, allocator);
                    current = new_mat;
                    current.coefficient = 1.0;
                    break;
                }
                // const eval_result_t res = matrix_multiply(order, &current, &this, &new_mat, allocator);
                // matrix_cleanup(&current, allocator);
                // if (res != EVAL_SUCCESS)
                // {
                //     return res;
                // }
                // current = new_mat;
            }
            break;

        case MATOP_MATMUL:
            if (stack_pos == 0)
            {
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_STACK_UNDERFLOW;
            }
            stack_pos -= 1;
            {
                matrix_t right = stack[stack_pos];
                matrix_t new_mat;
                const eval_result_t res = matrix_multiply(order, &right, &current, &new_mat, allocator);
                matrix_cleanup(stack + stack_pos, allocator);
                if (res != EVAL_SUCCESS)
                {
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return res;
                }
            }
            break;

        case MATOP_SUM:
            if (remaining < 1)
            {
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                const unsigned count = code[i].u32;
                if (count > stack_pos)
                {
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_STACK_UNDERFLOW;
                }
                for (unsigned j = 0; j < count; ++j)
                {
                    matrix_t new;
                    stack_pos -= 1;
                    matrix_t *left = stack + stack_pos;
                    const eval_result_t res = matrix_add(order, &current, left, &new, allocator);
                    matrix_cleanup(&current, allocator);
                    matrix_cleanup(left, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    current = new;
                }
            }
            break;

        default:
            clean_stack(stack, stack_pos, allocator);
            matrix_cleanup(&current, allocator);
            return EVAL_BAD_ENUM;
        }
    }
    clean_stack(stack, stack_pos, allocator);
    if (stack_pos != 0)
    {
        matrix_cleanup(&current, allocator);
        return EVAL_STACK_NOT_EMPTY;
    }

    switch (current.type)
    {
    case MATRIX_TYPE_IDENTITY: {
        const unsigned n = form_degrees_of_freedom_count(form, order);
        double *const restrict ptr = allocate(allocator, sizeof *ptr * n * n);
        if (!ptr)
        {
            return EVAL_FAILED_ALLOC;
        }
        memset(ptr, 0, sizeof *ptr * n * n);
        for (unsigned i = 0; i < n; ++i)
        {
            ptr[i * n + i] = current.coefficient;
        }
        *p_out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n, .cols = n}, .data = ptr};
    }
    break;

    case MATRIX_TYPE_FULL:
        matrix_multiply_inplace(&current.full, current.coefficient);
        *p_out = current.full;
        break;

    case MATRIX_TYPE_INCIDENCE:
        const eval_result_t res = incidence_to_full(current.incidence.incidence, order, p_out, allocator);
        if (res != EVAL_SUCCESS)
            return res;
        matrix_multiply_inplace(p_out, current.coefficient);
        break;

    default:
        return EVAL_BAD_ENUM;
    }

    return EVAL_SUCCESS;
}
