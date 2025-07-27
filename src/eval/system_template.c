#include "system_template.h"

#include "matrix.h"

#include "integrating_fields.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

MFV2D_INTERNAL
mfv2d_result_t convert_system_forms(PyObject *orders, unsigned *p_n_forms, form_order_t **pp_form_orders,
                                    const allocator_callbacks *allocator)
{
    PyArrayObject *const order_array = (PyArrayObject *)PyArray_FromAny(
        orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!order_array)
        return MFV2D_FAILED_ALLOC;

    const unsigned n_forms = PyArray_DIM(order_array, 0);
    form_order_t *const form_orders = allocate(allocator, sizeof(*form_orders) * n_forms);
    const unsigned *restrict p_orders = PyArray_DATA(order_array);
    for (unsigned i = 0; i < n_forms; ++i)
    {
        const unsigned order = p_orders[i];
        if (order > 3 || order == 0)
        {
            PyErr_Format(PyExc_ValueError, "Form can not be of order higher than 2 (it was %u)", order);
            Py_DECREF(order_array);
            return MFV2D_BAD_ARGUMENT;
        }
        form_orders[i] = order;
    }
    Py_DECREF(order_array);
    *p_n_forms = n_forms;
    *pp_form_orders = form_orders;
    return MFV2D_SUCCESS;
}
MFV2D_INTERNAL
mfv2d_result_t system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                                      const unsigned n_fields, const allocator_callbacks *allocator)
{
    mfv2d_result_t result;
    // Find the number of forms
    if ((result = convert_system_forms(orders, &this->n_forms, &this->form_orders, allocator)) != MFV2D_SUCCESS)
        return result;

    // Now go through the rows
    const ssize_t row_count = PySequence_Size(expr_matrix);
    if (row_count < 0)
    {
        deallocate(allocator, this->form_orders);
        return MFV2D_FAILED_ALLOC;
    }

    if (row_count != this->n_forms)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of forms deduced from order array (%u) does not match the number of expression rows (%u).",
                     this->n_forms, row_count);
        deallocate(allocator, this->form_orders);
        return MFV2D_BAD_ARGUMENT;
    }

    this->bytecodes = allocate(allocator, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    if (!this->bytecodes)
    {
        deallocate(allocator, this->form_orders);
        return MFV2D_FAILED_ALLOC;
    }
    memset(this->bytecodes, 0, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);

    memset(this->field_orders, FORM_ORDER_UNKNOWN, sizeof(*this->field_orders) * INTEGRATING_FIELDS_MAX_COUNT);

    unsigned max_stack = 1;
    for (unsigned row = 0; row < this->n_forms; ++row)
    {
        PyObject *row_expr = PySequence_GetItem(expr_matrix, row);
        if (!row_expr)
        {
            goto failed_row;
        }
        const ssize_t column_count = PySequence_Size(row_expr);
        if (column_count < 0)
        {
            goto failed_row;
        }
        if (column_count != this->n_forms)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Number of forms deduced from order array (%u) does not match the number of expression in row %u (%u).",
                this->n_forms, row, column_count);
            goto failed_row;
        }

        for (unsigned col = 0; col < this->n_forms; ++col)
        {
            PyObject *expr = PySequence_GetItem(row_expr, col);
            if (!expr)
            {
                result = MFV2D_UNSPECIFIED_ERROR;
                goto failed_row;
            }
            if (Py_IsNone(expr))
            {
                Py_DECREF(expr);
                continue;
            }
            PyObject *seq = PySequence_Fast(expr, "Bytecode must be a given as a sequence.");
            if (!seq)
            {
                Py_DECREF(expr);
                result = MFV2D_UNSPECIFIED_ERROR;
                goto failed_row;
            }
            const ssize_t expr_count = PySequence_Fast_GET_SIZE(seq);
            if (expr_count < 0)
            {
                Py_DECREF(expr);
                result = MFV2D_UNSPECIFIED_ERROR;
                goto failed_row;
            }

            bytecode_t *const bc = allocate(allocator, sizeof(*bc) + sizeof(*bc->ops) * expr_count);
            if (!bc)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                result = MFV2D_FAILED_ALLOC;
                goto failed_row;
            }
            bc->count = expr_count;
            this->bytecodes[row * this->n_forms + col] = bc;
            unsigned stack;
            if ((result = convert_bytecode(expr_count, bc->ops, PySequence_Fast_ITEMS(seq), &stack, n_fields,
                                           this->field_orders)) != MFV2D_SUCCESS)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            if (stack > max_stack)
            {
                max_stack = stack;
            }
            Py_DECREF(seq);
            Py_DECREF(expr);
        }

        continue;

    failed_row: {
        Py_XDECREF(row_expr);
        for (unsigned i = row; i > 0; --i)
        {
            for (unsigned j = this->n_forms; j > 0; --j)
            {
                deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
            }
        }
        deallocate(allocator, this->bytecodes);
        deallocate(allocator, this->form_orders);
        Py_DECREF(row_expr);
        return result;
    }
    }

    this->max_stack = max_stack;

    // for (unsigned i_field = 0; i_field < *p_field_cnt; ++i_field)
    // {
    //     if (field_orders[i_field] == FORM_ORDER_UNKNOWN)
    //     {
    //         PyErr_Format(PyExc_ValueError, "Field %u out of %u has no order specified.", i_field, *p_field_cnt);
    //         deallocate(allocator, this->form_orders);
    //         deallocate(allocator, this->bytecodes);
    //         return MFV2D_BAD_ARGUMENT;
    //     }
    // }

    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator)
{
    deallocate(allocator, this->form_orders);
    for (unsigned i = this->n_forms; this->bytecodes && i > 0; --i)
    {
        for (unsigned j = this->n_forms; j > 0; --j)
        {
            deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
        }
    }
    deallocate(allocator, this->bytecodes);
    *this = (system_template_t){};
}
