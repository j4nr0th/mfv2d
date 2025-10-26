#ifndef MFV2D_SPARSE_SYSTEM_H
#define MFV2D_SPARSE_SYSTEM_H
#include "matrix.h"
#include "svector.h"
#include <jmtx/jmtxd.h>

typedef struct
{
    // Dimension of the block
    unsigned n;
    // Diagonal n x n block
    double *diagonal_block;
    // Sparse constraint matrix m x n
    jmtxd_matrix_crs *constraints;
    // Inverse of the diagonal part
    double *diagonal_lu;
    // Pivot array for the diagonal inverse
    unsigned *pivots;
} system_block_t;

typedef struct
{
    // Number of blocks in the system
    unsigned n_blocks;
    // Array of n_blocks system blocks
    system_block_t *blocks;
    // Number of trace variables
    unsigned n_trace;
    // Offset array for each trace value's block indices
    unsigned *trace_offsets;
    // Packed array, which specifies what blocks each trace value gets assigned to
    unsigned *trace_values;
} system_t;

typedef struct
{
    // System the vector belongs to
    const system_t *parent;
    // Offset array for each block's vector values
    unsigned *offsets;
    // Packed array of vector values
    double *values;
} dense_vector_t;

typedef struct
{
    // System the vector belongs to
    const system_t *parent;
    // Array of trace values belonging to each element
    svector_t *values;
} trace_vector_t;

/**
 * Applies the diagonal blocks of a sparse system to an input vector and stores the results
 * in an output vector. Each block's diagonal matrix is multiplied with the corresponding
 * segment of the input vector, and the result is written to the corresponding segment of the
 * output vector.
 *
 * @param system Pointer to the sparse system containing the diagonal blocks to apply.
 * @param vec Pointer to the input dense vector. Must be associated with the same system as the `system` parameter.
 * @param out Pointer to the output dense vector. Must be associated with the same system as the `system` parameter.
 *
 * @return Returns a mfv2d_result_t indicating the result of the operation:
 *         - MFV2D_SUCCESS: The operation completed successfully.
 *         - MFV2D_BAD_VECTOR_PARENT: The input or output vector does not belong to the provided system.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_system_apply_diagonal(const system_t *system, const dense_vector_t *vec,
                                            const dense_vector_t *out);

/**
 * Applies the inverse of the diagonal blocks of a sparse system to an input vector and
 * stores the results in an output vector. Each block's diagonal matrix is inverted and
 * multiplied with the corresponding segment of the input vector, and the result is written
 * to the corresponding segment of the output vector.
 *
 * @param system Pointer to the sparse system containing the diagonal blocks to invert and apply.
 * @param vec Pointer to the input dense vector. Must be associated with the same system as the `system` parameter.
 * @param out Pointer to the output dense vector. Must be associated with the same system as the `system` parameter.
 *
 * @return Returns a mfv2d_result_t indicating the result of the operation:
 *         - MFV2D_SUCCESS: The operation completed successfully.
 *         - MFV2D_BAD_VECTOR_PARENT: The input or output vector does not belong to the provided system.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_system_apply_diagonal_inverse(const system_t *system, const dense_vector_t *vec,
                                                    const dense_vector_t *out);

/**
 * Applies the trace constraints of a sparse system to an input vector and updates the output
 * trace vector accordingly. For each block in the system, the sparse matrix of constraints is
 * applied to the corresponding segment of the input vector. The results are accumulated in the
 * trace vector.
 *
 * @param system Pointer to the sparse system containing the blocks and trace constraints to apply.
 * @param vec Pointer to the input dense vector. Must be associated with the same system as the `system` parameter.
 * @param out Pointer to the trace vector where the results will be accumulated. Must be associated with the same system
 * as the `system` parameter.
 *
 * @return Returns a mfv2d_result_t indicating the result of the operation:
 *         - MFV2D_SUCCESS: The operation completed successfully.
 *         - MFV2D_BAD_VECTOR_PARENT: The input or output vector does not belong to the provided system.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_system_apply_trace(const system_t *system, const dense_vector_t *vec, const trace_vector_t *out);

/**
 * Applies the transpose of the trace constraints of a sparse system to a trace vector and
 * accumulates the results into a dense vector. This operation computes the effect of each
 * block's trace constraints on the trace vector and writes the accumulated results to the
 * corresponding segments of the dense output vector.
 *
 * @param system Pointer to the sparse system containing the trace constraints to apply.
 * @param vec Pointer to the input trace vector. Must belong to the same system as the `system` parameter.
 * @param out Pointer to the output dense vector. Must belong to the same system as the `system` parameter.
 *
 * @return Returns a mfv2d_result_t status code indicating the result of the computation:
 *         - MFV2D_SUCCESS: The operation completed successfully.
 *         - MFV2D_BAD_VECTOR_PARENT: The input or output vector does not belong to the provided system.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_system_apply_trace_transpose(const system_t *system, const trace_vector_t *vec,
                                                   const dense_vector_t *out);

static inline void trace_element_indices(const system_t *system, const unsigned it, unsigned *p_count,
                                         unsigned **pp_indices)
{
    ASSERT(it < system->n_trace, "Trace index %u out of bounds for system with %u trace variables", it,
           system->n_trace);
    *p_count = system->trace_offsets[it + 1] - system->trace_offsets[it];
    *pp_indices = system->trace_values + system->trace_offsets[it];
}

static inline void element_dense_vector(const dense_vector_t *vector, const unsigned ie, unsigned *p_count,
                                        double **pp_values)
{
    ASSERT(ie < vector->parent->n_blocks, "Element index %u out of bounds for system with %u blocks", ie,
           vector->parent->n_blocks);
    const unsigned begin = vector->offsets[ie];
    const unsigned end = vector->offsets[ie + 1];
    *p_count = end - begin;
    *pp_values = vector->values + begin;
}

#endif // MFV2D_SPARSE_SYSTEM_H
