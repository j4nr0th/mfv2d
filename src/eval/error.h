//
// Created by jan on 16.2.2025.
//

#ifndef ERROR_H
#define ERROR_H

typedef enum
{
    EVAL_SUCCESS = 0,             // Success!
    EVAL_FAILED_ALLOC = 1,        // Memory allocation failed (got NULL)
    EVAL_BAD_ENUM = 2,            // Enum value which was unknown or should not be encountered.
    EVAL_DIMS_MISMATCH = 3,       // Matrix dimensions don't match (probably my fault).
    EVAL_DOUBLE_INCIDENCE = 4,    // Two incidence matrices in a row.
    EVAL_OUT_OF_INSTRUCTIONS = 5, // No more instructions when the instruction specifies additional args
    EVAL_STACK_OVERFLOW = 6,      // No more space on the matrix stack
    EVAL_STACK_UNDERFLOW = 7,     // No more arguments on the matrix stack
    EVAL_WRONG_MAT_TYPES = 8,     // Bad matrix type combination
    EVAL_STACK_NOT_EMPTY = 9,     // Stack was not empty.
} eval_result_t;

#endif // ERROR_H
