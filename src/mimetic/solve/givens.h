//
// Created by jan on 19.3.2025.
//

#ifndef GIVENS_H
#define GIVENS_H
#include "svector.h"

typedef struct
{
    uint64_t n;    // Order of the matrix
    uint64_t k, l; // Rotation indices
    scalar_t c, s; // Rotation values
} givens_rotation_t;

typedef struct
{
    PyObject_HEAD;
    givens_rotation_t data;
} givens_object_t;

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

extern PyTypeObject givens_rotation_type_object;

#endif // GIVENS_H
