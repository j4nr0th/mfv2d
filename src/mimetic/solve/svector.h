//
// Created by jan on 19.3.2025.
//

#ifndef SVECTOR_H
#define SVECTOR_H

#include "../../common.h"

typedef double scalar_t;
typedef struct
{
    uint64_t index;
    scalar_t value;
} entry_t;

typedef struct
{
    uint64_t n, size, capacity;
    entry_t *restrict entries;
} svector_t;

typedef struct
{
    uint64_t n;    // Order of the matrix
    uint64_t k, l; // Rotation indices
    scalar_t c, s; // Rotation values
} givens_rotation_t;

typedef struct
{
    PyObject_HEAD;
    uint64_t n, count, capacity;
    entry_t entries[];
} svec_object_t;

INTERPLIB_INTERNAL
extern PyTypeObject svec_type_object;

/**
 * Create a new sparse vector with no entries and desired capacity.
 *
 * @param this Memory where the resulting vector is initialized.
 * @param n Dimension of the vector.
 * @param capacity Desired capacity of the vector.
 * @param allocator Allocator used to get the memory for the vector.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int sparse_vector_new(svector_t *this, uint64_t n, uint64_t capacity, const allocator_callbacks *allocator);

/**
 * Clean up the memory used by the vector and clear the memory where it was stored.
 *
 * @param this Memory where the vector is stored.
 * @param allocator Allocator to release the memory with.
 */
INTERPLIB_INTERNAL
void sparse_vec_del(svector_t *this, const allocator_callbacks *allocator);

/**
 * Increase the size of the vector if too small.
 *
 * @param this Memory where the vector is stored.
 * @param capacity New required capacity.
 * @param allocator Allocator that can be used to reallocate the buffers as needed.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int sparse_vec_resize(svector_t *this, uint64_t capacity, const allocator_callbacks *allocator);

/**
 * Apply Givens rotations to two affected rows.
 *
 * @param c Cosine of the Given's rotation.
 * @param s Sine of the Given's rotation
 * @param row_i Top row for the rotation.
 * @param row_j Bottom row for the rotation.
 * @param out_i Sparse vector which receives the resulting top row.
 * @param out_j Sparse vector which receives the resulting bottom row.
 * @param allocator Allocator used for needed memory.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int apply_givens_rotation(scalar_t c, scalar_t s, const svector_t *row_i, const svector_t *row_j,
                          svector_t *restrict out_i, svector_t *restrict out_j, const allocator_callbacks *allocator);

#endif // SVECTOR_H
