//
// Created by jan on 15.2.2025.
//

#ifndef EVALUATION_H
#define EVALUATION_H

#include "../common.h"
#include "../common_defines.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

#include "error.h"

typedef enum
{
    // Error, this is not a valid operation
    MATOP_INVALID = 0,
    // Identity operation, do nothing.
    MATOP_IDENTITY = 1,
    // Mass matrix, next two values in bytecode are the order and if it should be inverted
    MATOP_MASS = 2,
    // Incidence matrix, next two values in bytecode are the order and if it should be dual
    MATOP_INCIDENCE = 3,
    // Push matrix on stack in order to prepare for multiplication or summation
    MATOP_PUSH = 4,
    // Multiply with matrix currently on stack
    MATOP_MATMUL = 5,
    // Scale by constant, which is the next bytecode value
    MATOP_SCALE = 6,
    // Transpose the current matrix
    MATOP_TRANSPOSE = 7,
    // Sum matrices with those on stack, the next bytecode value is says how many are to be popped from the stack.
    MATOP_SUM = 8,
    // Not an instruction, used to count how many instructions there are.
    MATOP_COUNT,
} matrix_op_t;

INTERPLIB_INTERNAL
const char *matrix_op_str(matrix_op_t op);

typedef union {
    matrix_op_t op;
    double f64;
    unsigned u32;
} bytecode_val_t;

typedef enum
{
    MASS_0 = 0,
    MASS_1 = 1,
    MASS_2 = 2,
    MASS_0_I = 3,
    MASS_1_I = 4,
    MASS_2_I = 5,
    MASS_CNT,
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
    PyArrayObject *arr_int_nodes;
    PyArrayObject *arr_node;
    PyArrayObject *arr_edge_00;
    PyArrayObject *arr_edge_01;
    PyArrayObject *arr_edge_11;
    PyArrayObject *arr_surf;
} basis_precomp_t;

typedef struct
{
    matrix_full_t mass_matrices[MASS_CNT];
} precompute_t;

typedef enum
{
    FORM_ORDER_UNKNOWN = 0,
    FORM_ORDER_0 = 1,
    FORM_ORDER_1 = 2,
    FORM_ORDER_2 = 3,
} form_order_t;

typedef struct
{
    unsigned max_stack;
    unsigned n_forms;
    form_order_t *form_orders;
    bytecode_val_t **bytecodes;
} system_template_t;

/**
 * Convert a Python sequence of MatOpCode, int, and float objects into the C-bytecode.
 *
 * @param n Number of elements in the sequence to convert.
 * @param bytecode Buffer to fill with bytecode.
 * @param items Python objects which are to be converted to instructions.
 * @param p_max_stack Pointer which receives the maximum number of matrices on the argument stack.
 * @return Non-zero on success.
 */
INTERPLIB_INTERNAL
int convert_bytecode(const unsigned n, bytecode_val_t bytecode[restrict n + 1], PyObject *items[static n],
                     unsigned *p_max_stack);

INTERPLIB_INTERNAL
int system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                           const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator);

static unsigned form_degrees_of_freedom_count(const form_order_t form, const unsigned order)
{
    switch (form)
    {
    case FORM_ORDER_0:
        return (order + 1) * (order + 1);
    case FORM_ORDER_1:
        return 2 * order * (order + 1);
    case FORM_ORDER_2:
        return order * order;
    default:
        return 0;
    }
}

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

/**
 * Turn Python serialized data into C-friendly form.
 *
 * @param serialized Serialized BasisCache tuple obtained by calling `BasisCache.c_serializaton`.
 * @param out Pointer to the struct which is filled out with arrays.
 * @return Non-zero on success.
 */
INTERPLIB_INTERNAL
int basis_precomp_create(PyObject *serialized, basis_precomp_t *out);

/**
 * Release the memory associated with the C-friendly precomputed data.
 *
 * @param this Basis precomputation to release.
 */
INTERPLIB_INTERNAL
void basis_precomp_destroy(basis_precomp_t *this);

/**
 * Create a new PyArray with contents of the full array.
 *
 * @param mat Matrix to turn into a PyArrayObject.
 * @return Pointer to the new array on success, NULL with Python error set on failure.
 */
INTERPLIB_INTERNAL
PyArrayObject *matrix_full_to_array(const matrix_full_t *mat);

INTERPLIB_INTERNAL
int evaluate_element_term(error_stack_t *error_stack, form_order_t form, unsigned order, const bytecode_val_t *code,
                          precompute_t *precomp, unsigned n_stack, matrix_t stack[restrict n_stack],
                          const allocator_callbacks *allocator, matrix_full_t *p_out);

INTERPLIB_INTERNAL
eval_result_t apply_incidence_to_full_left(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                           matrix_full_t *p_out, const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
eval_result_t apply_incidence_to_full_right(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                            matrix_full_t *p_out, const allocator_callbacks *allocator);

#endif // EVALUATION_H
