//
// Created by jan on 16.2.2025.
//

#ifndef ERROR_H
#define ERROR_H

#include "../common.h"

typedef enum
{
    EVAL_SUCCESS = 0,             // Success!
    EVAL_FAILED_ALLOC = 1,        // Memory allocation failed (got NULL).
    EVAL_BAD_ENUM = 2,            // Enum value which was unknown or should not be encountered.
    EVAL_DIMS_MISMATCH = 3,       // Matrix dimensions don't match (probably my fault).
    EVAL_DOUBLE_INCIDENCE = 4,    // Two incidence matrices in a row.
    EVAL_OUT_OF_INSTRUCTIONS = 5, // No more instructions when the instruction specifies additional args.
    EVAL_STACK_OVERFLOW = 6,      // No more space on the matrix stack.
    EVAL_STACK_UNDERFLOW = 7,     // No more arguments on the matrix stack.
    EVAL_WRONG_MAT_TYPES = 8,     // Bad matrix type combination.
    EVAL_STACK_NOT_EMPTY = 9,     // Stack was not empty.
    EVAL_NOT_SQUARE = 10,         // Surface was not topologically square.
    EVAL_NOT_IN_SURFACE = 11,     // Line was not in the surface
    EVAL_ORDER_MISMATCH = 12,     // Element orders don't match
    EVAL_UNSPECIFIED_ERROR = 13,  // Unspecified
    EVAL_COUNT,                   // Used to check if out of range.
} eval_result_t;

MFV2D_INTERNAL
const char *eval_result_str(eval_result_t e);

typedef struct
{
    eval_result_t code;
    char *message;
    int line;
    const char *file;
    const char *function;
} error_message_t;

typedef struct
{
    const allocator_callbacks *allocator;
    unsigned capacity;
    unsigned position;
    error_message_t messages[];
} error_stack_t;

MFV2D_INTERNAL
error_stack_t *error_stack_create(unsigned capacity, const allocator_callbacks *allocator);

#ifdef __GNUC__
__attribute__((format(printf, 6, 7)))
#endif
MFV2D_INTERNAL void
error_message_submit(error_stack_t *stack, const char *file, int line, const char *func, eval_result_t err,
                     const char *msg, ...);

#ifndef EVAL_ERROR
#define EVAL_ERROR(stack, err, msg, ...)                                                                               \
    error_message_submit((stack), __FILE__, __LINE__, __func__, (err), (msg)__VA_OPT__(, ) __VA_ARGS__)
#endif // EVAL_ERROR

#endif // ERROR_H
