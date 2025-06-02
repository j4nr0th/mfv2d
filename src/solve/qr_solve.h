//
// Created by jan on 18.3.2025.
//

#ifndef CCS_MATRIX_H
#define CCS_MATRIX_H

#include "../common.h"
#include "givens.h"
#include "lil_matrix.h"

typedef struct
{
    uint64_t rows, cols;
    uint64_t capacity, size;
    // Contains this.size values, but has the size of this.capacity
    scalar_t *restrict values;
    // Contains this.rows + 1 entries of beginning of row offsets
    uint64_t *restrict row_offsets;
} crs_matrix_t;

typedef struct
{
    PyObject_HEAD;
    crs_matrix_t mtx;
} crs_object_t;

extern PyTypeObject crs_type_object;

/**
 * Perform QR decomposition by applying Givens rotations to the split matrix.
 *
 * @param n_max Maximum number of operations allowed.
 * @param mat Matrix that will be decomposed.
 * @param p_ng Pointer that will receive the number of Given's rotations required.
 * @param p_givens Pointer that will receive an array (pointer) of p_ng Given's rotations that were performed
 * @param allocator Allocator callbacks that will be used.
 * @return Zero if successful.
 */
MFV2D_INTERNAL
int decompose_qr(int64_t n_max, const lil_matrix_t *mat, uint64_t *p_ng, givens_rotation_t **const p_givens,
                 const allocator_callbacks *allocator);

#endif // CCS_MATRIX_H
