//
// Created by jan on 20.2.2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include "../common.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

#include "error.h"

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

INTERPLIB_INTERNAL
void matrix_print(const matrix_t *mtx);

/**
 * Create a new PyArray with contents of the full array.
 *
 * @param mat Matrix to turn into a PyArrayObject.
 * @return Pointer to the new array on success, NULL with Python error set on failure.
 */
INTERPLIB_INTERNAL
PyArrayObject *matrix_full_to_array(const matrix_full_t *mat);

INTERPLIB_INTERNAL
void matrix_cleanup(matrix_t *this, const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
eval_result_t matrix_full_copy(const matrix_full_t *this, matrix_full_t *out, const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
eval_result_t matrix_full_multiply(const matrix_full_t *left, const matrix_full_t *right, matrix_full_t *out,
                                   const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
eval_result_t matrix_add(const unsigned order, matrix_t *right, matrix_t *left, matrix_t *out,
                         const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
eval_result_t matrix_full_add_inplace(const matrix_full_t *in, matrix_full_t *out);

INTERPLIB_INTERNAL
eval_result_t matrix_multiply(error_stack_t *error_stack, const unsigned order, const matrix_t *right,
                              const matrix_t *left, matrix_t *out, const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
void matrix_multiply_inplace(const matrix_full_t *this, const double k);

INTERPLIB_INTERNAL
void matrix_add_diagonal_inplace(const matrix_full_t *this, const double k);

#endif // MATRIX_H
