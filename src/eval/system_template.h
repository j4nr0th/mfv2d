#ifndef SYSTEM_TEMPLATE_H
#define SYSTEM_TEMPLATE_H

#include "bytecode.h"

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

MFV2D_INTERNAL
int system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                           const field_information_t *fields, const allocator_callbacks *allocator);

MFV2D_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator);

static unsigned form_degrees_of_freedom_count(const form_order_t form, const unsigned order_1, const unsigned order_2)
{
    switch (form)
    {
    case FORM_ORDER_0:
        return (order_1 + 1) * (order_2 + 1);
    case FORM_ORDER_1:
        return order_1 * (order_2 + 1) + (order_1 + 1) * order_2;
    case FORM_ORDER_2:
        return order_1 * order_2;
    default:
        return 0;
    }
}

#endif // SYSTEM_TEMPLATE_H
