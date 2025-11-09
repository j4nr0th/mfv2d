//
// Created by jan on 16.2.2025.
//

#ifndef ERROR_H
#define ERROR_H

#include "common.h"

typedef enum
{
    MFV2D_SUCCESS = 0,         // Success!
    MFV2D_FAILED_ALLOC,        // Memory allocation failed (got NULL).
    MFV2D_BAD_ENUM,            // Enum value which was unknown or should not be encountered.
    MFV2D_DIMS_MISMATCH,       // Matrix dimensions don't match (probably my fault).
    MFV2D_DOUBLE_INCIDENCE,    // Two incidence matrices in a row.
    MFV2D_OUT_OF_INSTRUCTIONS, // No more instructions when the instruction specifies additional args.
    MFV2D_STACK_OVERFLOW,      // No more space on the matrix stack.
    MFV2D_STACK_UNDERFLOW,     // No more arguments on the matrix stack.
    MFV2D_WRONG_MAT_TYPES,     // Bad matrix type combination.
    MFV2D_STACK_NOT_EMPTY,     // Stack was not empty.
    MFV2D_NOT_SQUARE,          // Surface was not topologically square.
    MFV2D_NOT_IN_SURFACE,      // Line was not in the surface
    MFV2D_ORDER_MISMATCH,      // Element orders don't match
    MFV2D_NOT_CONVERGED,       // Iterations did not converge
    MFV2D_INDEX_OUT_OF_RANGE,  // Index out of range
    MFV2D_NOT_A_LEAF,          // Element is not a leaf
    MFV2D_FAILED_CALLBACK,     // An external callback failed.
    MFV2D_UNSPECIFIED_ERROR,   // Unspecified
    MFV2D_BAD_ARGUMENT,        // Bad user specified argument (do not use for internal functions!)
    MFV2D_PYTHON_EXCEPTION,    // Python call raised an exception
    MFV2D_PIVOT_FAILED,        // Could not pivot the matrix
    MFV2D_BAD_VECTOR_PARENT,   // Vector does not belong to the system that should operate on it
    MFV2D_MISMATCHED_TRACE,    // Trace vector does not have required entries
    MFV2D_RESULT_COUNT,        // Used to check if out of range.
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
