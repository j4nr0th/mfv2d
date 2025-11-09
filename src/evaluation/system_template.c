#include "system_template.h"

#include "../algebra/matrix.h"

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
mfv2d_result_t system_template_create(system_template_t *this, const element_form_spec_t *const form_specs,
                                      const element_fem_space_2d_t *fem_space, PyObject *expr_matrix,
                                      const allocator_callbacks *allocator, const double degrees_of_freedom[restrict])
{
    mfv2d_result_t result;

    // Now go through the rows
    const ssize_t row_count = PySequence_Size(expr_matrix);
    if (row_count < 0)
    {
        return MFV2D_FAILED_ALLOC;
    }

    const unsigned n_forms = Py_SIZE(form_specs);
    if (row_count != n_forms)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of forms deduced from order array (%u) does not match the number of expression rows (%u).",
                     n_forms, row_count);
        return MFV2D_BAD_ARGUMENT;
    }
    this->n_forms = n_forms;

    this->bytecodes = allocate(allocator, sizeof(*this->bytecodes) * n_forms * n_forms);
    if (!this->bytecodes)
    {
        return MFV2D_FAILED_ALLOC;
    }
    memset(this->bytecodes, 0, sizeof(*this->bytecodes) * n_forms * n_forms);

    field_spec_t field_specs[INTEGRATING_FIELDS_MAX_COUNT] = {};

    unsigned max_stack = 1;
    // Translate bytecode and grab field specifications
    for (unsigned row = 0; row < n_forms; ++row)
    {
        PyObject *row_expr = PySequence_GetItem(expr_matrix, row);
        if (!row_expr)
        {
            system_template_destroy(this, allocator);
            return MFV2D_PYTHON_EXCEPTION;
        }
        const ssize_t column_count = PySequence_Size(row_expr);
        if (column_count < 0)
        {
            system_template_destroy(this, allocator);
            return MFV2D_PYTHON_EXCEPTION;
        }
        if (column_count != n_forms)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Number of forms deduced from order array (%u) does not match the number of expression in row %u (%u).",
                n_forms, row, column_count);

            system_template_destroy(this, allocator);
            return MFV2D_BAD_ARGUMENT;
        }

        for (unsigned col = 0; col < n_forms; ++col)
        {
            PyObject *expr = PySequence_GetItem(row_expr, col);
            if (!expr)
            {
                system_template_destroy(this, allocator);
                return MFV2D_PYTHON_EXCEPTION;
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
                system_template_destroy(this, allocator);
                return MFV2D_PYTHON_EXCEPTION;
            }
            // No need to check, this does not fail
            const Py_ssize_t expr_count = PySequence_Fast_GET_SIZE(seq);

            bytecode_t *const bc = allocate(allocator, sizeof(*bc) + sizeof(*bc->ops) * expr_count);
            if (!bc)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                system_template_destroy(this, allocator);
                return MFV2D_FAILED_ALLOC;
            }
            bc->count = expr_count;
            this->bytecodes[row * n_forms + col] = bc;
            unsigned stack;
            if ((result =
                     convert_bytecode(expr_count, bc->ops, PySequence_Fast_ITEMS(seq), &stack, &this->fields.n_fields,
                                      INTEGRATING_FIELDS_MAX_COUNT, field_specs, form_specs)) != MFV2D_SUCCESS)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                system_template_destroy(this, allocator);
                return result;
            }
            if (stack > max_stack)
            {
                max_stack = stack;
            }
            Py_DECREF(seq);
            Py_DECREF(expr);
        }
    }

    // Now evaluate the field specifications

    const mfv2d_result_t result_fields =
        compute_fields(fem_space->fem_space, &fem_space->corners, &this->fields, this->fields.n_fields, field_specs,
                       allocator, form_specs, degrees_of_freedom);

    if (result_fields != MFV2D_SUCCESS)
    {
        system_template_destroy(this, allocator);
        return result_fields;
    }

    this->max_stack = max_stack;

    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator)
{
    for (unsigned i = this->n_forms; this->bytecodes && i > 0; --i)
    {
        for (unsigned j = this->n_forms; j > 0; --j)
        {
            deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
        }
    }
    deallocate(allocator, this->bytecodes);
    deallocate(allocator, this->fields.buffer);
    *this = (system_template_t){};
}
