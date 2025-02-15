//
// Created by jan on 15.2.2025.
//

#ifndef EVALUATION_H
#define EVALUATION_H

#include "../common.h"
#include "../common_defines.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

typedef enum
{
    MATOP_INVALID = 0,
    MATOP_IDENTITY = 1,
    MATOP_MASS = 2,
    MATOP_INCIDENCE = 3,
    MATOP_PUSH = 4,
    MATOP_MATMUL = 5,
    MATOP_SCALE = 6,
    MATOP_TRANSPOSE = 7,
    MATOP_SUM = 8,
    MATOP_COUNT,
} matrix_op_t;

const char *matrx_op_str(matrix_op_t op);

typedef union {
    matrix_op_t op;
    double f64;
    unsigned u32;
} bytecode_t;

typedef enum
{
    MASS_0 = 0,
    MASS_1 = 1,
    MASS_2 = 2,
    MASS_0_I = 3,
    MASS_1_I = 4,
    MASS_2_I = 5,
    MASS_CNT = 6,
} mass_mtx_indices_t;

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
    INCIDENCE_TYPE_INVALID = 0,
    INCIDENCE_TYPE_10 = 1,
    INCIDENCE_TYPE_10_T = 2,
    INCIDENCE_TYPE_21 = 3,
    INCIDENCE_TYPE_21_T = 3,
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

typedef union {
    matrix_type_t type;
    matrix_base_t base;
    matrix_identity_t identity;
    matrix_incidence_t incidence;
    matrix_full_t full;
} matrix_t;

typedef struct
{
    unsigned order;
    unsigned n_int;
    const double *nodes_int;
    const double *mass_nodal;
    const double *mass_edge_00;
    const double *mass_edge_01;
    const double *mass_edge_11;
    const double *mass_surf;
} basis_precomp_t;

typedef struct
{
    matrix_full_t mass_matrices[MASS_CNT];
} precompute_t;

/**
 * Create a `precompute_t` object with all matrices. This function expects Python `BasisCache` data as input, so it is
 * equivalent to the `Element.mass_matrix_node` and similar methods.
 *
 * Another notable point about this function is that it actually allocates and deallocates the memory it needs in a
 * FIFO order, with both output and internal memory. This means, a stack-based allocator can be used to make memory
 * allocation overheads trivially small.
 *
 * @param basis Pre-computed basis double products on the reference element. These are filled with contents of Python
 * `BasisCache` object.
 * @param x0 Bottom left corner's x coordinate.
 * @param x1 Bottom right corner's x coordinate.
 * @param x2 Top right corner's x coordinate.
 * @param x3 Top left corner's x coordinate.
 * @param y0 Bottom left corner's y coordinate.
 * @param y1 Bottom right corner's y coordinate.
 * @param y2 Top right corner's y coordinate.
 * @param y3 Top left corner's y coordinate.
 * @param out Pointer which receives the computed mass matrices.
 * @param allocator Allocator to be used in this function for output and intermediate buffers. Can be stack-based
 * @return Non-zero on success.
 */
INTERPLIB_INTERNAL
int precompute_create(const basis_precomp_t *basis, double x0, double x1, double x2, double x3, double y0, double y1,
                      double y2, double y3, precompute_t *out, allocator_callbacks *allocator);

INTERPLIB_INTERNAL
PyArrayObject *matrix_full_to_array(const matrix_full_t *mat);

#endif // EVALUATION_H
