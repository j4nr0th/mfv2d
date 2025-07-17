//
// Created by jan on 17.2.2025.
//
#include "error.h"

#define MFV2D_RESULT_STR_ENTRY(v) [(v)] = #v
static const char *eval_result_strings[EVAL_COUNT] = {
    MFV2D_RESULT_STR_ENTRY(MFV2D_SUCCESS),          MFV2D_RESULT_STR_ENTRY(MFV2D_FAILED_ALLOC),
    MFV2D_RESULT_STR_ENTRY(MFV2D_BAD_ENUM),         MFV2D_RESULT_STR_ENTRY(MFV2D_DIMS_MISMATCH),
    MFV2D_RESULT_STR_ENTRY(MFV2D_DOUBLE_INCIDENCE), MFV2D_RESULT_STR_ENTRY(MFV2D_OUT_OF_INSTRUCTIONS),
    MFV2D_RESULT_STR_ENTRY(MFV2D_STACK_OVERFLOW),   MFV2D_RESULT_STR_ENTRY(MFV2D_STACK_UNDERFLOW),
    MFV2D_RESULT_STR_ENTRY(MFV2D_WRONG_MAT_TYPES),  MFV2D_RESULT_STR_ENTRY(MFV2D_STACK_NOT_EMPTY),
    MFV2D_RESULT_STR_ENTRY(MFV2D_NOT_SQUARE),       MFV2D_RESULT_STR_ENTRY(MFV2D_NOT_IN_SURFACE),
    MFV2D_RESULT_STR_ENTRY(MFV2D_ORDER_MISMATCH),   MFV2D_RESULT_STR_ENTRY(MFV2D_UNSPECIFIED_ERROR),
    MFV2D_RESULT_STR_ENTRY(MFV2D_NOT_CONVERGED),    MFV2D_RESULT_STR_ENTRY(MFV2D_INDEX_OUT_OF_RANGE),
    MFV2D_RESULT_STR_ENTRY(MFV2D_NOT_A_LEAF),
};
#undef MFV2D_RESULT_STR_ENTRY

const char *mfv2d_result_str(mfv2d_result_t e)
{
    if (e < MFV2D_SUCCESS || e >= EVAL_COUNT)
        return "UNKNOWN";
    return eval_result_strings[e];
}

error_stack_t *error_stack_create(unsigned capacity, const allocator_callbacks *allocator)
{
    error_stack_t *const this = allocate(allocator, sizeof(*this) * capacity * sizeof(error_message_t));
    if (!this)
        return this;
    this->capacity = capacity;
    this->position = 0;
    this->allocator = allocator;

    return this;
}

void error_message_submit(error_stack_t *stack, const char *file, int line, const char *func, mfv2d_result_t err,
                          const char *msg, ...)
{
    if (stack->position == stack->capacity)
        return;

    va_list args, cpy;
    va_start(args, msg);
    va_copy(cpy, args);
    const int len = vsnprintf(NULL, 0, msg, cpy);
    va_end(cpy);
    if (len < 0)
        return;
    char *const buffer = allocate(stack->allocator, sizeof *buffer * (len + 1));
    if (!buffer)
        return;
    vsnprintf(buffer, len, msg, args);
    va_end(args);
    stack->messages[stack->position] =
        (error_message_t){.code = err, .message = buffer, .line = line, .file = file, .function = func};
    stack->position += 1;
}
