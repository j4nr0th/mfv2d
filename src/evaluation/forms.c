//
// Created by jan on 7/25/25.
//

#include "forms.h"

static const char *form_order_str_table[] = {
    [FORM_ORDER_UNKNOWN] = "FORM_ORDER_UNKNOWN",
    [FORM_ORDER_0] = "FORM_ORDER_0",
    [FORM_ORDER_1] = "FORM_ORDER_1",
    [FORM_ORDER_2] = "FORM_ORDER_2",
};

const char *form_order_str(form_order_t order)
{
    // ALWAYS RANGE CHECK!
    if (order < FORM_ORDER_UNKNOWN || order > FORM_ORDER_2)
        return "FORM_ORDER_INVALID";
    return form_order_str_table[order];
}
form_order_t form_order_from_object(PyObject *object)
{
    const long val = PyLong_AsLong(object);
    if (PyErr_Occurred() || val < FORM_ORDER_0 || val > FORM_ORDER_2)
    {
        raise_exception_from_current(PyExc_ValueError, "Invalid form order: %ld", val);
        return FORM_ORDER_UNKNOWN;
    }
    return (form_order_t)val;
}
