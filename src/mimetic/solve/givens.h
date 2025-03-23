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

typedef struct
{
    PyObject_VAR_HEAD;
    uint64_t n;
    givens_rotation_t data[];
} givens_series_t;

/**
 * Apply Givens rotations to two affected rows.
 *
 * @param c Cosine of the Given's rotation.
 * @param s Sine of the Given's rotation
 * @param row_i Top row for the rotation.
 * @param row_j Bottom row for the rotation.
 * @param out_i Sparse vector which receives the resulting top row.
 * @param out_j Sparse vector which receives the resulting bottom row.
 * @param cut_j How many elements should be skipped for the out_j row.
 * This is useful when it is known that the first entry will be eliminated.
 * @param allocator Allocator used for needed memory.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int apply_givens_rotation(scalar_t c, scalar_t s, const svector_t *row_i, const svector_t *row_j,
                          svector_t *restrict out_i, svector_t *restrict out_j, unsigned cut_j,
                          const allocator_callbacks *allocator);

/**
 * Convert a C-based Givens rotation object into a Python version of it.
 *
 * @param g Givens rotation object to convert.
 * @return Pointer to the newly created object, or NULL on failure.
 */
INTERPLIB_INTERNAL
givens_object_t *givens_to_python(const givens_rotation_t *g);

INTERPLIB_INTERNAL
extern PyTypeObject givens_rotation_type_object;

INTERPLIB_INTERNAL
extern PyTypeObject givens_series_type_object;

/**
 * Apply Givens rotation to a sparse column vector.
 *
 * @param g Givens rotation that is to be applied.
 * @param vec Vector to be rotated.
 * @param allocator Allocator used to resize/allocate memory for output if needed.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int givens_rotate_sparse_vector_inplace(const givens_rotation_t *g, svector_t *vec,
                                        const allocator_callbacks *allocator);

/**
 * Convert a C-based Givens rotations object into a Python version GivensSeries.
 *
 * @param n Number of Givens rotations to join together.
 * @param g Givens rotation object to convert.
 * @return Pointer to the newly created object, or NULL on failure.
 */
INTERPLIB_INTERNAL
givens_series_t *givens_series_to_python(uint64_t n, const givens_rotation_t g[static n]);

#endif // GIVENS_H
