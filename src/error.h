//
// Created by jan on 16.2.2025.
//

#ifndef ERROR_H
#define ERROR_H

#include "common.h"

typedef enum
{
    MFV2D_SUCCESS = 0,             // Success!
    MFV2D_FAILED_ALLOC = 1,        // Memory allocation failed (got NULL).
    MFV2D_BAD_ENUM = 2,            // Enum value which was unknown or should not be encountered.
    MFV2D_DIMS_MISMATCH = 3,       // Matrix dimensions don't match (probably my fault).
    MFV2D_DOUBLE_INCIDENCE = 4,    // Two incidence matrices in a row.
    MFV2D_OUT_OF_INSTRUCTIONS = 5, // No more instructions when the instruction specifies additional args.
    MFV2D_STACK_OVERFLOW = 6,      // No more space on the matrix stack.
    MFV2D_STACK_UNDERFLOW = 7,     // No more arguments on the matrix stack.
    MFV2D_WRONG_MAT_TYPES = 8,     // Bad matrix type combination.
    MFV2D_STACK_NOT_EMPTY = 9,     // Stack was not empty.
    MFV2D_NOT_SQUARE = 10,         // Surface was not topologically square.
    MFV2D_NOT_IN_SURFACE = 11,     // Line was not in the surface
    MFV2D_ORDER_MISMATCH = 12,     // Element orders don't match
    MFV2D_NOT_CONVERGED = 13,      // Iterations did not converge
    MFV2D_INDEX_OUT_OF_RANGE,      // Index out of range
    MFV2D_NOT_A_LEAF,              // Element is not a leaf
    MFV2D_FAILED_CALLBACK,         // An external callback failed.
    MFV2D_UNSPECIFIED_ERROR,       // Unspecified
    EVAL_COUNT,                    // Used to check if out of range.
} mfv2d_result_t;

MFV2D_INTERNAL
const char *mfv2d_result_str(mfv2d_result_t e);

typedef struct
{
    mfv2d_result_t code;
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
error_message_submit(error_stack_t *stack, const char *file, int line, const char *func, mfv2d_result_t err,
                     const char *msg, ...);

#ifndef MFV2D_ERROR
#define MFV2D_ERROR(stack, err, msg, ...)                                                                              \
    error_message_submit((stack), __FILE__, __LINE__, __func__, (err), (msg)__VA_OPT__(, ) __VA_ARGS__)
#endif // MFV2D_ERROR

#endif // ERROR_H
