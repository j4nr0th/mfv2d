//
// Created by jan on 15.2.2025.
//

#ifndef EVALUATION_H
#define EVALUATION_H

#include "../../common.h"
#include "../../common_defines.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

#include "bytecode.h"
#include "error.h"
#include "precomp.h"

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
    bytecode_t **bytecodes;
} system_template_t;

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

INTERPLIB_INTERNAL
eval_result_t evaluate_element_term(error_stack_t *error_stack, form_order_t form, unsigned order,
                                    const bytecode_t *code, precompute_t *precomp, unsigned n_stack,
                                    matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                    matrix_full_t *p_out);

INTERPLIB_INTERNAL
eval_result_t evaluate_element_term_sibling(error_stack_t *error_stack, form_order_t form, unsigned order,
                                            const bytecode_t *code, precompute_t *precomp,
                                            const field_information_t *vector_fields, unsigned n_stack,
                                            matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                            matrix_full_t *p_out);

#endif // EVALUATION_H
