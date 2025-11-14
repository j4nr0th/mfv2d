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

MFV2D_INTERNAL
unsigned form_degrees_of_freedom_count(form_order_t form, unsigned order_1, unsigned order_2);

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
// extern PyTypeObject element_form_spec_type;
extern PyType_Spec element_form_spec_type_spec;

MFV2D_INTERNAL
// extern PyTypeObject element_form_spec_iter_type;
extern PyType_Spec element_form_spec_iter_type_spec;

MFV2D_INTERNAL
unsigned element_form_offset(const element_form_spec_t *spec, unsigned index, unsigned order_1, unsigned order_2);

MFV2D_INTERNAL
unsigned element_form_specs_total_count(const element_form_spec_t *spec, unsigned order_1, unsigned order_2);

#endif // FORMS_H
