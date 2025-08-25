//
// Created by jan on 20.2.2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include "../common/common.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

#include "../common/error.h"

typedef enum
{
    MATRIX_TYPE_INVALID = 0,
    MATRIX_TYPE_IDENTITY = 1,
    MATRIX_TYPE_INCIDENCE = 2,
    MATRIX_TYPE_FULL,
} matrix_type_t;

typedef struct
{
    matrix_type_t type;
    unsigned rows, cols;
} matrix_base_t;

typedef struct
{
    matrix_base_t base;
} matrix_identity_t;

typedef enum
{
    INCIDENCE_TYPE_10 = 0,
    INCIDENCE_TYPE_21 = 1,
    INCIDENCE_TYPE_10_T = 2,
    INCIDENCE_TYPE_21_T = 3,
    INCIDENCE_TYPE_CNT,
} incidence_type_t;

MFV2D_INTERNAL
const char *incidence_type_str(incidence_type_t type);

typedef struct
{
    matrix_base_t base;
    incidence_type_t incidence;
} matrix_incidence_t;

typedef struct
{
    matrix_base_t base;
    double *data;
} matrix_full_t;

typedef struct
{
    union {
        matrix_type_t type;
        matrix_base_t base;
        matrix_identity_t identity;
        matrix_incidence_t incidence;
        matrix_full_t full;
    };
    double coefficient;
} matrix_t;

typedef struct
{
    double j00, j01, j10, j11, det;
} jacobian_t;
MFV2D_INTERNAL
void matrix_print(const matrix_t *mtx);

/**
 * Create a new PyArray with contents of the full array.
 *
 * @param mat Matrix to turn into a PyArrayObject.
 * @return Pointer to the new array on success, NULL with Python error set on failure.
 */
MFV2D_INTERNAL
PyArrayObject *matrix_full_to_array(const matrix_full_t *mat);

MFV2D_INTERNAL
void matrix_cleanup(matrix_t *this, const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t matrix_full_copy(const matrix_full_t *this, matrix_full_t *out, const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t matrix_full_multiply(const matrix_full_t *left, const matrix_full_t *right, matrix_full_t *out,
                                    const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t matrix_add(const unsigned order, matrix_t *right, matrix_t *left, matrix_t *out,
                          const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t matrix_full_add_inplace(const matrix_full_t *in, matrix_full_t *out);

MFV2D_INTERNAL
mfv2d_result_t matrix_multiply(error_stack_t *error_stack, const unsigned order, const matrix_t *right,
                               const matrix_t *left, matrix_t *out, const allocator_callbacks *allocator);

MFV2D_INTERNAL
void matrix_multiply_inplace(const matrix_full_t *this, const double k);

MFV2D_INTERNAL
void matrix_add_diagonal_inplace(const matrix_full_t *this, const double k);

MFV2D_INTERNAL
void invert_matrix(size_t n, const double mat[static n * n], double buffer[restrict n * n], double out[n * n]);

/**
 * Computes the inverse of a square matrix and stores the result in a provided output matrix.
 *
 * @param this Pointer to the input matrix (must be a square matrix).
 * @param p_out Pointer to the output matrix where the inverted matrix will be stored.
 * @param allocator Pointer to the allocator callbacks used for memory allocation.
 * @return MFV2D_SUCCESS on successful inversion, or an appropriate error code on failure (EVAL_DIMS_MISMATCH if
 * the input matrix is not square, or EVAL_FAILED_ALLOC if memory allocation fails).
 */
MFV2D_INTERNAL
mfv2d_result_t matrix_full_invert(const matrix_full_t *this, matrix_full_t *p_out,
                                  const allocator_callbacks *allocator);

MFV2D_INTERNAL
PyObject *python_compute_matrix_inverse(PyObject *self, PyObject *arg);

MFV2D_INTERNAL
extern const char compute_matrix_inverse_docstr[];

#endif // MATRIX_H
