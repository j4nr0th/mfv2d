//
// Created by jan on 20.2.2025.
//

#ifndef BYTECODE_H
#define BYTECODE_H

#include "../common/common.h"
#include "forms.h"
#include "integrating_fields.h"

typedef enum
{
    // Error, this is not a valid operation
    MATOP_INVALID = 0,
    // Identity operation - do nothing.
    MATOP_IDENTITY = 1,
    // Mass matrix, the next two values in bytecode are the order, and if it should be inverted
    MATOP_MASS = 2,
    // Incidence matrix, the next two values in bytecode are the order, and if it should be dual
    MATOP_INCIDENCE = 3,
    // Push matrix on the stack to prepare for multiplication or summation
    MATOP_PUSH = 4,
    // Multiply with matrix currently on stack
    MATOP_MATMUL = 5,
    // Scale by constant, which is the next bytecode value
    MATOP_SCALE = 6,
    // Sum matrices with those on stack, the next bytecode value is says how many are to be popped from the stack.
    MATOP_SUM = 7,
    // Interior product with vector field.
    MATOP_INTERPROD = 8,
    // Not an instruction, used to count how many instructions there are.
    MATOP_COUNT,
} matrix_op_type_t;

MFV2D_INTERNAL
const char *matrix_op_type_str(matrix_op_type_t op);

typedef struct
{
    matrix_op_type_t op;
} matrix_op_identity_t;

typedef struct
{
    matrix_op_type_t op;
    form_order_t order;
    unsigned invert;
} matrix_op_mass_t;

typedef struct
{
    matrix_op_type_t op;
    form_order_t order;
    unsigned transpose;
} matrix_op_incidence_t;

typedef struct
{
    matrix_op_type_t op;
} matrix_op_push_t;

typedef struct
{
    matrix_op_type_t op;
} matrix_op_matmul_t;

typedef struct
{
    matrix_op_type_t op;
    double k;
} matrix_op_scale_t;

typedef struct
{
    matrix_op_type_t op;
    unsigned n;
} matrix_op_sum_t;

typedef struct
{
    matrix_op_type_t op;
    form_order_t order;
    unsigned field_index;
    unsigned dual;
    unsigned adjoint;
} matrix_op_interprod_t;

typedef union {
    matrix_op_type_t type;
    matrix_op_identity_t identity;
    matrix_op_mass_t mass;
    matrix_op_incidence_t incidence;
    matrix_op_push_t push;
    matrix_op_matmul_t matmul;
    matrix_op_scale_t scale;
    matrix_op_sum_t sum;
    matrix_op_interprod_t interprod;
} matrix_op_t;

typedef struct
{
    unsigned count;
    matrix_op_t ops[];
} bytecode_t;

/**
 * Convert a Python sequence of MatOpCode, int, and float objects into the C-bytecode.
 *
 * @param n Number of elements in the sequence to convert.
 * @param ops Array which is to be filled with translated instructions.
 * @param items Python objects which are to be converted to instructions.
 * @param p_max_stack Pointer which receives the maximum number of matrices on the argument stack.
 * @param p_field_cnt Pointer that receives the upper bound of fields present in the system.
 * @param p_form_orders Orders of fields that were identified.
 * @return MFV2D_SUCCESS if successful.
 */
MFV2D_INTERNAL
mfv2d_result_t convert_bytecode(unsigned n, matrix_op_t ops[restrict const n], PyObject *const items[static n],
                                unsigned *p_max_stack, unsigned *p_field_cnt, unsigned max_fields,
                                field_spec_t field_specs[restrict const max_fields],
                                const element_form_spec_t *form_specs);

MFV2D_INTERNAL
int matrix_op_type_from_object(PyObject *o, matrix_op_type_t *out);

MFV2D_INTERNAL
PyObject *check_bytecode(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char check_bytecode_docstr[];

#endif // BYTECODE_H
