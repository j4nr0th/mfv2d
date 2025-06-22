#include "system_template.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

MFV2D_INTERNAL
int system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix, unsigned n_vec_fields,
                           const allocator_callbacks *allocator)
{
    // Find the number of forms
    {
        PyArrayObject *const order_array = (PyArrayObject *)PyArray_FromAny(
            orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
        if (!order_array)
            return 0;

        const unsigned n_forms = PyArray_DIM(order_array, 0);
        this->n_forms = n_forms;
        this->form_orders = allocate(allocator, sizeof(*this->form_orders) * n_forms);
        const unsigned *restrict p_o = PyArray_DATA(order_array);
        for (unsigned i = 0; i < n_forms; ++i)
        {
            const unsigned o = p_o[i];
            if (o > 3 || o == 0)
            {
                PyErr_Format(PyExc_ValueError, "Form can not be of order higher than 2 (it was %u)", o);
                Py_DECREF(order_array);
                return 0;
            }
            this->form_orders[i] = o;
        }
        Py_DECREF(order_array);
    }

    // Now go through the rows
    const ssize_t row_count = PySequence_Size(expr_matrix);
    if (row_count < 0)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }

    if (row_count != this->n_forms)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of forms deduced from order array (%u) does not match the number of expression rows (%u).",
                     this->n_forms, row_count);
        deallocate(allocator, this->form_orders);
        return 0;
    }

    this->bytecodes = allocate(allocator, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    if (!this->bytecodes)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }
    memset(this->bytecodes, 0, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
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
                goto failed_row;
            }
            const ssize_t expr_count = PySequence_Fast_GET_SIZE(seq);
            if (expr_count < 0)
            {
                Py_DECREF(expr);
                goto failed_row;
            }

            bytecode_t *bc = allocate(allocator, sizeof(**this->bytecodes) * (expr_count + 1));
            if (!bc)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            this->bytecodes[row * this->n_forms + col] = bc;
            unsigned stack;
            if (!convert_bytecode(expr_count, bc, PySequence_Fast_ITEMS(seq), &stack, n_vec_fields))
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
        return 0;
    }
    }

    this->max_stack = max_stack;

    return 1;
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
