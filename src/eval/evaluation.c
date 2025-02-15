//
// Created by jan on 15.2.2025.
//

#include "evaluation.h"

#define MATRIX_OP_ENTRY(op) [op] = #op
static const char *matrix_op_strings[MATOP_COUNT] = {
    MATRIX_OP_ENTRY(MATOP_INVALID),   MATRIX_OP_ENTRY(MATOP_IDENTITY), MATRIX_OP_ENTRY(MATOP_MASS),
    MATRIX_OP_ENTRY(MATOP_INCIDENCE), MATRIX_OP_ENTRY(MATOP_PUSH),     MATRIX_OP_ENTRY(MATOP_SCALE),
    MATRIX_OP_ENTRY(MATOP_TRANSPOSE), MATRIX_OP_ENTRY(MATOP_SUM),
};
#undef MATRIX_OP_ENTRY

const char *matrx_op_str(const matrix_op_t op)
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
