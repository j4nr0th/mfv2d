#ifndef SYSTEM_TEMPLATE_H
#define SYSTEM_TEMPLATE_H

#include "../fem_space/element_fem_space.h"
#include "bytecode.h"
#include "forms.h"
#include "integrating_fields.h"

typedef struct
{
    unsigned max_stack;
    unsigned n_forms;
    field_values_t fields;
    bytecode_t **bytecodes;
} system_template_t;

MFV2D_INTERNAL
mfv2d_result_t convert_system_forms(PyObject *orders, unsigned *p_n_forms, form_order_t **pp_form_orders,
                                    const allocator_callbacks *allocator);

MFV2D_INTERNAL
mfv2d_result_t system_template_create(system_template_t *this, const element_form_spec_t *form_specs,
                                      const element_fem_space_2d_t *fem_space, PyObject *expr_matrix,
                                      const allocator_callbacks *allocator, const double degrees_of_freedom[restrict]);

MFV2D_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator);

#endif // SYSTEM_TEMPLATE_H
