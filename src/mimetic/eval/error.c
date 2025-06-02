//
// Created by jan on 17.2.2025.
//
#include "error.h"

#define EVAL_RESULT_STR_ENTRY(v) [(v)] = #v
static const char *eval_result_strings[EVAL_COUNT] = {
    EVAL_RESULT_STR_ENTRY(EVAL_SUCCESS),          EVAL_RESULT_STR_ENTRY(EVAL_FAILED_ALLOC),
    EVAL_RESULT_STR_ENTRY(EVAL_BAD_ENUM),         EVAL_RESULT_STR_ENTRY(EVAL_DIMS_MISMATCH),
    EVAL_RESULT_STR_ENTRY(EVAL_DOUBLE_INCIDENCE), EVAL_RESULT_STR_ENTRY(EVAL_OUT_OF_INSTRUCTIONS),
    EVAL_RESULT_STR_ENTRY(EVAL_STACK_OVERFLOW),   EVAL_RESULT_STR_ENTRY(EVAL_STACK_UNDERFLOW),
    EVAL_RESULT_STR_ENTRY(EVAL_WRONG_MAT_TYPES),  EVAL_RESULT_STR_ENTRY(EVAL_STACK_NOT_EMPTY),
    EVAL_RESULT_STR_ENTRY(EVAL_NOT_SQUARE),       EVAL_RESULT_STR_ENTRY(EVAL_NOT_IN_SURFACE),
    EVAL_RESULT_STR_ENTRY(EVAL_ORDER_MISMATCH),
};
#undef EVAL_RESULT_STR_ENTRY

const char *eval_result_str(eval_result_t e)
{
    if (e < EVAL_SUCCESS || e >= EVAL_COUNT)
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

void error_message_submit(error_stack_t *stack, const char *file, int line, const char *func, eval_result_t err,
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
