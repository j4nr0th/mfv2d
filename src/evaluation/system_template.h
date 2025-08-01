#ifndef SYSTEM_TEMPLATE_H
#define SYSTEM_TEMPLATE_H

#include "bytecode.h"
#include "forms.h"

typedef struct
{
    unsigned max_stack;
    unsigned n_forms;
    form_order_t *form_orders;
    form_order_t field_orders[INTEGRATING_FIELDS_MAX_COUNT];
    bytecode_t **bytecodes;
} system_template_t;

MFV2D_INTERNAL
mfv2d_result_t convert_system_forms(PyObject *orders, unsigned *p_n_forms, form_order_t **pp_form_orders,
                                    const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                                      unsigned n_fields, const allocator_callbacks *allocator);

MFV2D_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator);

#endif // SYSTEM_TEMPLATE_H
