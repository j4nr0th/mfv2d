#include "sparse_system.h"

mfv2d_result_t sparse_system_apply_diagonal(const system_t *system, const dense_vector_t *vec,
                                            const dense_vector_t *out)
{
    if (system != vec->parent || system != out->parent)
    {
        return MFV2D_BAD_VECTOR_PARENT;
    }

    for (unsigned i = 0; i < system->n_blocks; ++i)
    {
        const system_block_t *const block = system->blocks + i;
        const double *const mat = block->diagonal_block;

        unsigned n_in;
        double *p_in;
        element_dense_vector(vec, i, &n_in, &p_in);

        unsigned n_out;
        double *p_out;
        element_dense_vector(out, i, &n_out, &p_out);

        ASSERT(n_in == n_out, "Inconsistent number of elements in input and output vectors");
        ASSERT(n_in == block->n, "Inconsistent number of rows in matrix and vector");
        ASSERT(n_in == block->n, "Inconsistent number of cols in matrix and vector");

        for (unsigned row = 0; row < n_out; ++row)
        {
            double v = 0.0;
            for (unsigned col = 0; col < n_in; ++col)
            {
                v += p_in[col] * mat[row * block->n + col];
            }
            p_out[row] = v;
        }
    }

    return MFV2D_SUCCESS;
}

mfv2d_result_t sparse_system_apply_diagonal_inverse(const system_t *system, const dense_vector_t *vec,
                                                    const dense_vector_t *out)
{
    if (system != vec->parent || system != out->parent)
    {
        return MFV2D_BAD_VECTOR_PARENT;
    }

    for (unsigned i = 0; i < system->n_blocks; ++i)
    {
        const system_block_t *const block = system->blocks + i;

        unsigned n_in;
        double *p_in;
        element_dense_vector(vec, i, &n_in, &p_in);

        unsigned n_out;
        double *p_out;
        element_dense_vector(out, i, &n_out, &p_out);

        ASSERT(n_in == n_out, "Inconsistent number of elements in input and output vectors");
        ASSERT(n_in == block->n, "Inconsistent number of rows in matrix and vector");

        // Manually pivot the vector
        for (unsigned row = 0; row < n_out; ++row)
        {
            // TODO: check this is not the other way around
            p_out[block->pivots[row]] = p_in[row];
        }

        // Now solve using the decomposition (we can use the same input and output buffer, since we un-pivoted).
        solve_lu(block->n, block->n, block->diagonal_lu, p_out, p_out);
    }

    return MFV2D_SUCCESS;
}

mfv2d_result_t sparse_system_apply_trace(const system_t *system, const dense_vector_t *vec, const trace_vector_t *out)
{
    if (system != vec->parent || system != out->parent)
    {
        return MFV2D_BAD_VECTOR_PARENT;
    }

    // Zero out the output vector
    for (unsigned i = 0; i < system->n_blocks; ++i)
    {
        const svector_t *const svec = out->values + i;
        for (unsigned j = 0; j < svec->count; ++j)
        {
            svec->entries[j].value = 0.0;
        }
    }

    for (unsigned i = 0; i < system->n_blocks; ++i)
    {
        const system_block_t *const block = system->blocks + i;
        const jmtxd_matrix_crs *const constraints = block->constraints;

        unsigned n_in;
        double *p_in;
        element_dense_vector(vec, i, &n_in, &p_in);

        ASSERT(n_in == block->n, "Inconsistent number of rows in matrix and vector");
        ASSERT(n_in == constraints->base.cols, "Inconsistent number of vector and constraint matrix columns");

        // Compute the sparse output
        for (unsigned l_index = 0; l_index < constraints->base.rows; ++l_index)
        {
            const double value = jmtxd_matrix_crs_vector_multiply_row(constraints, p_in, l_index);
            if (value == 0.0)
                continue;

            // Check which constraints goes where
            const unsigned elem_off = system->trace_offsets[l_index];
            const unsigned elem_cnt = system->trace_offsets[l_index + 1] - elem_off;
            const unsigned *const elements = system->trace_values + elem_off;
            for (unsigned j = 0; j < elem_cnt; ++j)
            {
                const svector_t out_v = out->values[elements[j]];

                unsigned k;
                for (k = 0; k < out_v.count; ++k)
                {
                    if (out_v.entries[k].index == l_index)
                    {
                        out_v.entries[k].value += value;
                        break;
                    }
                }
                ASSERT(k < out_v.count, "Trace vector of block %u did not contain value with index %u", elements[j],
                       l_index);
            }
        }
    }

    return MFV2D_SUCCESS;
}

mfv2d_result_t sparse_system_apply_trace_transpose(const system_t *system, const trace_vector_t *vec,
                                                   const dense_vector_t *out)
{
    if (system != vec->parent || system != out->parent)
    {
        return MFV2D_BAD_VECTOR_PARENT;
    }

    for (unsigned i = 0; i < system->n_blocks; ++i)
    {
        const system_block_t *const block = system->blocks + i;
        const jmtxd_matrix_crs *const constraints = block->constraints;
        const svector_t *const svec = vec->values + i;

        unsigned n_out;
        double *p_out;
        element_dense_vector(out, i, &n_out, &p_out);
        ASSERT(n_out == constraints->base.cols, "Inconsistent number of cols in matrix and vector");
        ASSERT(n_out == block->n, "Inconsistent number of rows in matrix and vector");

        // Zero the output
        for (unsigned j = 0; j < n_out; ++j)
        {
            p_out[j] = 0.0;
        }

        for (unsigned j = 0; j < svec->count; ++j)
        {
            uint32_t *cols;
            double *values;
            const uint32_t n = jmtxd_matrix_crs_get_row(constraints, svec->entries[j].index, &cols, &values);
            for (unsigned k = 0; k < n; ++k)
            {
                p_out[cols[k]] += values[k] * svec->entries[j].value;
            }
        }
    }

    return MFV2D_SUCCESS;
}
