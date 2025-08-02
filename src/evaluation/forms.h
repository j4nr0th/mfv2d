#ifndef FORMS_H
#define FORMS_H

#include "../common/common.h"
#include <Python.h>

typedef enum
{
    FORM_ORDER_UNKNOWN = 0,
    FORM_ORDER_0 = 1,
    FORM_ORDER_1 = 2,
    FORM_ORDER_2 = 3,
} form_order_t;

MFV2D_INTERNAL
const char *form_order_str(form_order_t order);

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

MFV2D_INTERNAL
form_order_t form_order_from_object(PyObject *object);

enum
{
    MAXIMUM_FORM_NAME_LENGTH = 31
};

typedef struct
{
    char name[MAXIMUM_FORM_NAME_LENGTH + 1];
    form_order_t order;
} form_spec_t;

typedef struct
{
    PyObject_VAR_HEAD;
    form_spec_t forms[];
} element_form_spec_t;

MFV2D_INTERNAL
extern PyTypeObject element_form_spec_type;

MFV2D_INTERNAL
extern PyTypeObject element_form_spec_iter_type;

MFV2D_INTERNAL
unsigned element_form_offset(const element_form_spec_t *const spec, unsigned index, unsigned order_1, unsigned order_2);

MFV2D_INTERNAL
unsigned element_form_specs_total_count(const element_form_spec_t *const spec, unsigned order_1, unsigned order_2);

#endif // FORMS_H
