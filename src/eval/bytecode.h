//
// Created by jan on 20.2.2025.
//

#ifndef BYTECODE_H
#define BYTECODE_H

#include "../common.h"

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
    // Interior product with vector field.
    MATOP_INTERPROD = 9,
    // Not an instruction, used to count how many instructions there are.
    MATOP_COUNT,
} matrix_op_t;

MFV2D_INTERNAL
const char *matrix_op_str(matrix_op_t op);

typedef union {
    matrix_op_t op;
    double f64;
    unsigned u32;
} bytecode_t;

/**
 * Convert a Python sequence of MatOpCode, int, and float objects into the C-bytecode.
 *
 * @param n Number of elements in the sequence to convert.
 * @param bytecode Buffer to fill with bytecode.
 * @param items Python objects which are to be converted to instructions.
 * @param p_max_stack Pointer which receives the maximum number of matrices on the argument stack.
 * @return Non-zero on success.
 */
MFV2D_INTERNAL
int convert_bytecode(unsigned n, bytecode_t bytecode[restrict n + 1], PyObject *items[static n], unsigned *p_max_stack);

#endif // BYTECODE_H
