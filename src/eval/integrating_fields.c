//
// Created by jan on 7/24/25.
//

#include "integrating_fields.h"

#include "allocator.h"
#include "element_fem_space.h"
#include "fem_space.h"
#include "system_template.h"

typedef enum
{
    FIELD_SPEC_UNKNOWN = 1,
    FIELD_SPEC_CALLABLE = 2,
    FIELD_SPEC_UNUSED = -1,
} field_spec_type_t;

typedef struct
{
    field_spec_type_t type;
    unsigned index;
    // form_order_t form_order;
} field_spec_unknown_t;

typedef struct
{
    field_spec_type_t type;
    PyObject *callable;
} field_spec_callable_t;

typedef union {
    field_spec_type_t type;
    field_spec_unknown_t unknown;
    field_spec_callable_t callable;
} field_spec_t;

MFV2D_INTERNAL
mfv2d_result_t compute_fields(const fem_space_2d_t *fem_space, const quad_info_t *quad, field_information_t *fields,
                              const unsigned n_fields, const form_order_t field_orders[static n_fields],
                              const allocator_callbacks *allocator, const unsigned n_unknowns,
                              const form_order_t unknown_orders[static n_unknowns],
                              const double degrees_of_freedom[restrict], PyObject *field_data)
{
    // Check if there's any need for this.
    if (Py_IsNone(field_data))
        return MFV2D_SUCCESS;

    // Check type
    if (!PyTuple_Check(field_data))
    {
        PyErr_Format(PyExc_TypeError, "Expected a tuple of field specifications, got %s", Py_TYPE(field_data)->tp_name);
        return MFV2D_BAD_ARGUMENT;
    }

    // Get count
    if (PyTuple_GET_SIZE(field_data) < n_fields)
    {
        PyErr_Format(PyExc_ValueError, "Expected at least %d field specifications, got %d", n_fields,
                     PyTuple_GET_SIZE(field_data));
        return MFV2D_BAD_ARGUMENT;
    }

    if (n_fields == 0)
        // No work to do.
        return MFV2D_SUCCESS;

    // Too many. Maximum is hardcoded for the sake of simplicity. Also, who would need more than 16 fields?
    // Maybe hyper-sonics people with their 30 chemical species, but for that there will be bigger problems
    // with this code.
    if (n_fields > INTEGRATING_FIELDS_MAX_COUNT)
    {
        // Should not really happen, but let's have this here for the sake of safety.
        PyErr_Format(PyExc_ValueError, "Too many fields specified, maximum is %d", INTEGRATING_FIELDS_MAX_COUNT);
        return MFV2D_BAD_ARGUMENT;
    }

    field_spec_t field_specs[INTEGRATING_FIELDS_MAX_COUNT];
    unsigned n_unknown = 0;
    unsigned n_callable = 0;

    for (unsigned i = 0; i < n_fields; ++i)
    {
        if (field_orders[i] == FORM_ORDER_UNKNOWN)
        {
            // This is field is not present in the bytecode.
            field_specs[i].type = FIELD_SPEC_UNUSED;
            continue;
        }
        if (field_orders[i] > FORM_ORDER_2 || field_orders[i] < FORM_ORDER_0)
        {
            PyErr_Format(PyExc_ValueError, "Field %u has invalid order %s", i, form_order_str(field_orders[i]));
            return MFV2D_BAD_ARGUMENT;
        }
        PyObject *item = PyTuple_GET_ITEM(field_data, i);
        if (PyCallable_Check(item))
        {
            n_callable += 1;
            field_spec_callable_t *const this = &field_specs[i].callable;
            this->type = FIELD_SPEC_CALLABLE;
            this->callable = item;
        }
        else
        {
            n_unknown += 1;
            field_spec_unknown_t *const this = &field_specs[i].unknown;
            this->type = FIELD_SPEC_UNKNOWN;
            this->index = PyLong_AsLong(item);
            if (PyErr_Occurred())
                return MFV2D_BAD_ARGUMENT;
            if (this->index >= n_unknowns)
            {
                PyErr_Format(PyExc_ValueError,
                             "Field %u specifies to be based on unknown with index %u, but only %u "
                             "unknowns are in the system.",
                             i, this->index, n_unknowns);
                return MFV2D_BAD_ARGUMENT;
            }
            if (unknown_orders[this->index] != field_orders[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Field %u specifies to be based on unknown form with order %s, but the form has order %s "
                             "in the system.",
                             i, form_order_str(field_orders[i]), form_order_str(unknown_orders[this->index]));
            }
            // this->form_order = (form_order_t);
            // if (this->form_order <= FORM_ORDER_UNKNOWN || this->form_order > FORM_ORDER_2)
            // {
            //     PyErr_Format(PyExc_ValueError, "Unknown form order %d", this->form_order);
            //     return MFV2D_BAD_ARGUMENT;
            // }
        }
    }

    unsigned n_field_components = 0;
    // Count of field components before and including the current field.
    unsigned field_cumulative_components[INTEGRATING_FIELDS_MAX_COUNT];

    for (unsigned i = 0; i < n_fields; ++i)
    {
        const form_order_t order = field_orders[i];
        if (order == FORM_ORDER_1)
        {
            n_field_components += 2;
        }
        else
        {
            n_field_components += 1;
        }
        field_cumulative_components[i] = n_field_components;
    }

    const unsigned n_field_points = fem_space->space_1.n_pts * fem_space->space_2.n_pts;
    double *const field_buffer =
        (double *)allocate(allocator, sizeof(double) * n_field_points * field_cumulative_components[n_fields - 1]);
    if (field_buffer == NULL)
    {
        return MFV2D_FAILED_ALLOC;
    }

    // Compute fields from callables

    if (n_callable > 0)
    {
        // Prepare coordinate arrays x and y
        const npy_intp dims[2] = {fem_space->space_2.n_pts, fem_space->space_1.n_pts};
        PyArrayObject *const coordinates_x = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject *const coordinates_y = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

        if (!coordinates_x || !coordinates_y)
        {
            deallocate(allocator, field_buffer);
            Py_XDECREF(coordinates_x);
            Py_XDECREF(coordinates_y);
            return MFV2D_FAILED_ALLOC;
        }
        double *const x = (double *)PyArray_DATA(coordinates_x);
        double *const y = (double *)PyArray_DATA(coordinates_y);

        for (unsigned i = 0; i < fem_space->space_2.n_pts; ++i)
        {
            const double eta = fem_space->space_2.pnts[i];
            const double psi_eta_0 = (1 - eta) / 2.0;
            const double psi_eta_1 = (1 + eta) / 2.0;
            for (unsigned j = 0; j < fem_space->space_1.n_pts; ++j)
            {
                const double xi = fem_space->space_1.pnts[j];
                const double psi_xi_0 = (1 - xi) / 2.0;
                const double psi_xi_1 = (1 + xi) / 2.0;

                x[i * fem_space->space_1.n_pts + j] = psi_xi_0 * psi_eta_0 * quad->x0 +
                                                      psi_xi_1 * psi_eta_0 * quad->x1 +
                                                      psi_xi_1 * psi_eta_1 * quad->x2 + psi_xi_0 * psi_eta_1 * quad->x3;

                y[i * fem_space->space_1.n_pts + j] = psi_xi_0 * psi_eta_0 * quad->y0 +
                                                      psi_xi_1 * psi_eta_0 * quad->y1 +
                                                      psi_xi_1 * psi_eta_1 * quad->y2 + psi_xi_0 * psi_eta_1 * quad->y3;
            }
        }

        // For every callable field specification call it, convert to NumPy array, then copy it to the buffer.
        for (unsigned i = 0, j = 0; i < n_fields && j < n_callable; ++i)
        {
            const field_spec_callable_t *const this = &field_specs[i].callable;
            if (this->type != FIELD_SPEC_CALLABLE)
                continue;
            j += 1;

            PyObject *const result = PyObject_CallFunctionObjArgs(this->callable, coordinates_x, coordinates_y, NULL);
            if (!result)
            {

                PyObject *repr = PyObject_Repr(this->callable);
                if (repr)
                {
                    const char *name = PyUnicode_AsUTF8(repr);
                    raise_exception_from_current(PyExc_RuntimeError, "Failed to call field callback %s", name);
                    Py_DECREF(repr);
                }
                raise_exception_from_current(PyExc_RuntimeError, "Failed to call field callback UNKNOWN.");
                Py_DECREF(coordinates_x);
                Py_DECREF(coordinates_y);
                deallocate(allocator, field_buffer);
                return MFV2D_FAILED_CALLBACK;
            }

            const form_order_t order = field_orders[i];
            PyArrayObject *const array =
                (PyArrayObject *)PyArray_FROMANY(result, NPY_DOUBLE, 2, 3, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
            Py_DECREF(result);

            if (!array)
            {
                Py_DECREF(coordinates_x);
                Py_DECREF(coordinates_y);
                deallocate(allocator, field_buffer);
                return MFV2D_FAILED_ALLOC;
            }
            const npy_intp res_dims[3] = {fem_space->space_2.n_pts, fem_space->space_1.n_pts, 2};

            if ((order == FORM_ORDER_1 &&
                 check_input_array(array, 3, res_dims, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                   "field_data") < 0) ||
                (order != FORM_ORDER_1 &&
                 check_input_array(array, 2, res_dims, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                   "field_data") < 0))
            {

                Py_DECREF(coordinates_x);
                Py_DECREF(coordinates_y);
                Py_DECREF(array);
                raise_exception_from_current(PyExc_RuntimeError, "Invalid field data for field %d", i);
                deallocate(allocator, field_buffer);
                return MFV2D_BAD_ARGUMENT;
            }

            const double *const res = (double *)PyArray_DATA(array);
            double *const field_ptr = field_buffer + (i == 0 ? 0 : field_cumulative_components[i - 1] * n_field_points);
            memcpy(field_ptr, res, sizeof(double) * n_field_points * (order == FORM_ORDER_1 ? 2 : 1));

            fields->fields[i] = field_ptr;
            Py_DECREF(array);
        }

        Py_DECREF(coordinates_x);
        Py_DECREF(coordinates_y);
    }

    if (n_unknown > 0)
    {
        for (unsigned i = 0, j = 0; i < n_fields && j < n_unknown; ++i)
        {
            const field_spec_unknown_t *const this = &field_specs[i].unknown;
            if (this->type != FIELD_SPEC_UNKNOWN)
                continue;
            j += 1;
            const unsigned index = this->index;
            const form_order_t order = unknown_orders[index];
            unsigned dof_offset = 0;
            for (unsigned k = 0; k < index; ++k)
            {
                dof_offset += form_degrees_of_freedom_count(unknown_orders[k], fem_space->space_1.order,
                                                            fem_space->space_2.order);
            }

            double *const field_ptr = field_buffer + (i == 0 ? 0 : field_cumulative_components[i - 1] * n_field_points);
            switch (order)
            {
            case FORM_ORDER_0:
                reconstruct_field_0_form(fem_space, degrees_of_freedom + dof_offset, field_ptr);
                break;

            case FORM_ORDER_1:
                reconstruct_field_1_form(fem_space, degrees_of_freedom + dof_offset, field_ptr);
                break;

            case FORM_ORDER_2:
                reconstruct_field_2_form(fem_space, degrees_of_freedom + dof_offset, field_ptr);
                break;

            default:
                ASSERT(0, "Should never reach this.");
                deallocate(allocator, field_buffer);
                return MFV2D_BAD_ARGUMENT;
            }
            fields->fields[i] = field_ptr;
        }
    }

    fields->n_fields = n_fields;
    fields->buffer = field_buffer;
    return MFV2D_SUCCESS;
}

void reconstruct_field_0_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict])
{
    // Zero out the output buffer
    memset(output, 0, sizeof(*output) * space->space_1.n_pts * space->space_2.n_pts);
    const unsigned basis_count = fem_space_node_basis_cnt(space);

    // Loop over the basis and their degrees of freedom
    for (unsigned i_basis = 0; i_basis < basis_count; ++i_basis)
    {
        const double coefficient = degrees_of_freedom[i_basis];
        for (unsigned i = 0; i < space->space_2.n_pts; ++i)
        {
            for (unsigned j = 0; j < space->space_1.n_pts; ++j)
            {
                output[j + i * space->space_1.n_pts] += coefficient * node_basis_value(space, i_basis, i, j);
            }
        }
    }

    // Scale by the Jacobian's determinant
    for (unsigned i = 0; i < space->space_2.n_pts; ++i)
    {
        for (unsigned j = 0; j < space->space_1.n_pts; ++j)
        {
            output[j + i * space->space_1.n_pts] *= space->jacobian[j + i * space->space_1.n_pts].det;
        }
    }
}

void reconstruct_field_1_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict])
{
    // Zero out the output buffer
    const unsigned n_pts = space->space_1.n_pts * space->space_2.n_pts;
    memset(output, 0, sizeof(*output) * n_pts * 2);
    const unsigned basis_count_h = fem_space_edge_h_basis_cnt(space);
    const unsigned basis_count_v = fem_space_edge_v_basis_cnt(space);

    // First, compute the components as xi and eta

    // This here is Eta
    for (unsigned i_basis = 0; i_basis < basis_count_h; ++i_basis)
    {
        const double coefficient = degrees_of_freedom[i_basis];
        for (unsigned i = 0; i < space->space_2.n_pts; ++i)
        {
            for (unsigned j = 0; j < space->space_1.n_pts; ++j)
            {
                output[2 * (j + i * space->space_1.n_pts) + 0] +=
                    coefficient * edge_h_basis_value(space, i_basis, i, j);
            }
        }
    }

    // This here is Xi
    for (unsigned i_basis = 0; i_basis < basis_count_v; ++i_basis)
    {
        const double coefficient = degrees_of_freedom[i_basis + basis_count_h];
        for (unsigned i = 0; i < space->space_2.n_pts; ++i)
        {
            for (unsigned j = 0; j < space->space_1.n_pts; ++j)
            {
                output[2 * (j + i * space->space_1.n_pts) + 1] +=
                    coefficient * edge_v_basis_value(space, i_basis, i, j);
            }
        }
    }

    // Transfer from (eta, xi) to (x, y)
    for (unsigned i = 0; i < space->space_2.n_pts; ++i)
    {
        for (unsigned j = 0; j < space->space_1.n_pts; ++j)
        {
            const double v_eta = output[2 * (j + i * space->space_1.n_pts) + 0];
            const double v_xi = output[2 * (j + i * space->space_1.n_pts) + 1];
            const jacobian_t *const jac = space->jacobian + j + i * space->space_1.n_pts;
            const double v_x = jac->j00 * v_xi + jac->j10 * v_eta;
            const double v_y = jac->j01 * v_xi + jac->j11 * v_eta;
            // Scale by the Jacobian's determinant
            output[2 * (j + i * space->space_1.n_pts) + 0] = v_x / jac->det;
            output[2 * (j + i * space->space_1.n_pts) + 1] = v_y / jac->det;
        }
    }
}

void reconstruct_field_2_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict])
{
    // Zero out the output buffer
    memset(output, 0, sizeof(*output) * space->space_1.n_pts * space->space_2.n_pts);
    const unsigned basis_count = fem_space_surf_basis_cnt(space);

    // Loop over the basis and their degrees of freedom
    for (unsigned i_basis = 0; i_basis < basis_count; ++i_basis)
    {
        const double coefficient = degrees_of_freedom[i_basis];
        for (unsigned i = 0; i < space->space_2.n_pts; ++i)
        {
            for (unsigned j = 0; j < space->space_1.n_pts; ++j)
            {
                output[j + i * space->space_1.n_pts] += coefficient * surf_basis_value(space, i_basis, i, j);
            }
        }
    }

    // Scale by the Jacobian's determinant
    for (unsigned i = 0; i < space->space_2.n_pts; ++i)
    {
        for (unsigned j = 0; j < space->space_1.n_pts; ++j)
        {
            output[j + i * space->space_1.n_pts] /= space->jacobian[j + i * space->space_1.n_pts].det;
        }
    }
}

MFV2D_INTERNAL
PyObject *compute_integrating_fields(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    element_fem_space_2d_t *element_space;
    PyObject *py_form_orders;
    PyObject *py_field_information;
    PyObject *py_field_orders;
    PyArrayObject *degrees_of_freedom;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O!O!O!O!",
            (char *[6]){"fem_space", "form_orders", "field_information", "field_orders", "degrees_of_freedom", NULL},
            &element_fem_space_2d_type, &element_space, &PyTuple_Type, &py_form_orders, &PyTuple_Type,
            &py_field_information, &PyTuple_Type, &py_field_orders, &PyArray_Type, &degrees_of_freedom))
    {
        return NULL;
    }
    const unsigned n_fields = PyTuple_GET_SIZE(py_field_information);
    if (PyTuple_GET_SIZE(py_field_orders) != n_fields)
    {
        PyErr_Format(PyExc_ValueError,
                     "Field orders and field information must have the same length (one was %u, other was %u).",
                     (unsigned)PyTuple_GET_SIZE(py_field_orders), (unsigned)n_fields);
        return NULL;
    }
    if (n_fields >= INTEGRATING_FIELDS_MAX_COUNT)
    {
        PyErr_Format(PyExc_ValueError, "Too many fields specified (maximum is %u, but %u were given).",
                     INTEGRATING_FIELDS_MAX_COUNT, n_fields);
        return NULL;
    }

    mfv2d_result_t result;

    unsigned n_forms = 0;
    form_order_t *form_orders;
    if ((result = convert_system_forms(py_form_orders, &n_forms, &form_orders, &SYSTEM_ALLOCATOR)) != MFV2D_SUCCESS)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Failed to convert form orders, reason: %s",
                                     mfv2d_result_str(result));
        return NULL;
    }

    unsigned total_dof_count = 0;
    for (unsigned i = 0; i < n_forms; ++i)
    {
        total_dof_count += form_degrees_of_freedom_count(form_orders[i], element_space->basis_xi->order,
                                                         element_space->basis_eta->order);
    }
    if (check_input_array(degrees_of_freedom, 1, (const npy_intp[1]){total_dof_count}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "degrees_of_freedom"))
    {
        deallocate(&SYSTEM_ALLOCATOR, form_orders);
        return NULL;
    }

    form_order_t field_orders[INTEGRATING_FIELDS_MAX_COUNT];
    for (unsigned i = 0; i < n_fields; ++i)
    {
        const form_order_t order = PyLong_AsLong(PyTuple_GET_ITEM(py_field_orders, i));
        if (PyErr_Occurred())
        {
            deallocate(&SYSTEM_ALLOCATOR, form_orders);
            return NULL;
        }
        field_orders[i] = order;
    }

    field_information_t fields;
    result =
        compute_fields(element_space->fem_space, &element_space->corners, &fields, n_fields, field_orders,
                       &SYSTEM_ALLOCATOR, n_forms, form_orders, PyArray_DATA(degrees_of_freedom), py_field_information);
    deallocate(&SYSTEM_ALLOCATOR, form_orders);

    if (result != MFV2D_SUCCESS)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Failed to compute fields, reason: %s",
                                     mfv2d_result_str(result));
        return NULL;
    }

    PyObject *py_fields = PyTuple_New(n_fields);
    const unsigned n_field_points = element_space->fem_space->space_1.n_pts * element_space->fem_space->space_2.n_pts;
    for (unsigned i = 0; i < n_fields; ++i)
    {
        const npy_intp dims[3] = {element_space->fem_space->space_2.n_pts, element_space->fem_space->space_1.n_pts, 2};
        const form_order_t order = field_orders[i];
        const unsigned n_components = order == FORM_ORDER_1 ? 2 : 1;
        PyArrayObject *const py_field = (PyArrayObject *)PyArray_SimpleNew(n_components + 1, dims, NPY_DOUBLE);
        if (!py_field)
        {
            deallocate(&SYSTEM_ALLOCATOR, fields.buffer);
            Py_DECREF(py_fields);
            return NULL;
        }
        memcpy(PyArray_DATA(py_field), fields.fields[i], sizeof(double) * n_field_points * n_components);

        PyTuple_SET_ITEM(py_fields, i, py_field);
    }

    deallocate(&SYSTEM_ALLOCATOR, fields.buffer);
    return py_fields;
}

MFV2D_INTERNAL
const char compute_integrating_fields_docstring[] =
    "Compute fields at integration points.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fem_space : ElementFemSpace2D\n"
    "    Element FEM space to use for basis and integration rules.\n"
    "\n"
    "form_orders : tuple of UnknownFormOrder\n"
    "    Orders of differential forms in the system.\n"
    "\n"
    "field_information : tuple of int or Function2D\n"
    "    Information of how to compute the field - an integer indicates to use degrees of\n"
    "    freedom of that form, while a function indicates it should be called and\n"
    "    evaluated.\n"
    "\n"
    "degrees_of_freedom : array\n"
    "    Array with degrees of freedom from which the fields may be computed.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "tuple of arrays\n"
    "    Fields reconstructed at the integration points.\n";
