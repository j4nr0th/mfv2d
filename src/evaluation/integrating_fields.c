//
// Created by jan on 7/24/25.
//

#include "integrating_fields.h"

#include "../common/allocator.h"
#include "../fem_space/element_fem_space.h"
#include "../fem_space/fem_space.h"
#include "system_template.h"

MFV2D_INTERNAL
mfv2d_result_t compute_fields(const fem_space_2d_t *fem_space, const quad_info_t *quad, field_values_t *fields,
                              const unsigned n_fields, const field_spec_t field_specs[const static n_fields],
                              const allocator_callbacks *allocator, const element_form_spec_t *form_spec,
                              const double degrees_of_freedom[restrict])
{
    if (n_fields == 0)
        // No work to do.
        return MFV2D_SUCCESS;

    // Too many. Maximum is hardcoded for the sake of simplicity. Also, who would need more than 16 fields?
    // Maybe hyper-sonics people with their 30 chemical species, but for that there will be bigger problems
    // with this code.
    if (n_fields > INTEGRATING_FIELDS_MAX_COUNT)
    {
        // Should not really happen, but let's have this here for the sake of safety.
        PyErr_Format(PyExc_ValueError, "Too many fields specified, got %u but the maximum is %d", n_fields,
                     INTEGRATING_FIELDS_MAX_COUNT);
        return MFV2D_BAD_ARGUMENT;
    }

    unsigned n_unknown = 0;
    unsigned n_callable = 0;
    unsigned n_field_components = 0;

    for (unsigned i = 0; i < n_fields; ++i)
    {
        const field_spec_t *const spec = field_specs + i;
        if (spec->type == FIELD_SPEC_UNKNOWN)
        {
            n_unknown += 1;
        }
        else if (spec->type == FIELD_SPEC_CALLABLE)
        {
            n_callable += 1;
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "Invalid field specification type %d", spec->type);
            return MFV2D_BAD_ARGUMENT;
        }

        // Count of field components before and including the current field.
        if (spec->form_order == FORM_ORDER_1)
        {
            n_field_components += 2;
        }
        else
        {
            n_field_components += 1;
        }
    }

    const unsigned n_field_points = fem_space->space_1.n_pts * fem_space->space_2.n_pts;
    double *const field_buffer = (double *)allocate(allocator, sizeof(double) * n_field_points * n_field_components);
    if (field_buffer == NULL)
    {
        return MFV2D_FAILED_ALLOC;
    }

    // Compute fields from callables

    if (n_callable > 0)
    {
        unsigned components_processed = 0;
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

            const unsigned field_components = this->form_order == FORM_ORDER_1 ? 2 : 1;
            if (this->type != FIELD_SPEC_CALLABLE)
            {
                components_processed += field_components;
                continue;
            }
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

            if ((this->form_order == FORM_ORDER_1 &&
                 check_input_array(array, 3, res_dims, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                   "field_data") < 0) ||
                (this->form_order != FORM_ORDER_1 &&
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
            double *const field_ptr = field_buffer + n_field_points * components_processed;
            memcpy(field_ptr, res, sizeof(double) * n_field_points * field_components);

            fields->fields[i] = field_ptr;
            Py_DECREF(array);
            components_processed += field_components;
        }

        Py_DECREF(coordinates_x);
        Py_DECREF(coordinates_y);
    }

    if (n_unknown > 0)
    {
        unsigned components_processed = 0;
        for (unsigned i = 0, j = 0; i < n_fields && j < n_unknown; ++i)
        {
            const field_spec_unknown_t *const this = &field_specs[i].unknown;
            const unsigned field_components = this->form_order == FORM_ORDER_1 ? 2 : 1;
            if (this->type != FIELD_SPEC_UNKNOWN)
            {
                components_processed += field_components;
                continue;
            }
            j += 1;
            const unsigned index = this->index;
            const form_order_t order = this->form_order;
            unsigned dof_offset = 0;
            for (unsigned k = 0; k < index; ++k)
            {
                dof_offset += form_degrees_of_freedom_count(form_spec->forms[k].order, fem_space->space_1.order,
                                                            fem_space->space_2.order);
            }

            double *const field_ptr = field_buffer + n_field_points * components_processed;
            components_processed += field_components;
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
PyObject *compute_integrating_fields(PyObject *mod, PyObject *args, PyObject *kwds)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    element_fem_space_2d_t *element_space;
    element_form_spec_t *form_specs;
    PyObject *py_field_orders;
    PyObject *field_information;
    PyArrayObject *degrees_of_freedom;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O!O!O!O!",
            (char *[6]){"fem_space", "form_specs", "field_orders", "field_information", "degrees_of_freedom", NULL},
            state->type_fem_space, &element_space, &element_form_spec_type, &form_specs, &PyTuple_Type,
            &py_field_orders, &PyTuple_Type, &field_information, &PyArray_Type, &degrees_of_freedom))
    {
        return NULL;
    }
    const unsigned n_fields = PyTuple_GET_SIZE(field_information);
    if (n_fields != PyTuple_GET_SIZE(py_field_orders))
    {
        PyErr_Format(PyExc_ValueError, "Field orders and field information must have the same length.");
        return NULL;
    }
    if (n_fields >= INTEGRATING_FIELDS_MAX_COUNT)
    {
        PyErr_Format(PyExc_ValueError, "Too many fields specified (maximum is %u, but %u were given).",
                     INTEGRATING_FIELDS_MAX_COUNT, n_fields);
        return NULL;
    }

    const unsigned total_dof_count = element_form_specs_total_count(form_specs, element_space->fem_space->space_1.order,
                                                                    element_space->fem_space->space_2.order);
    if (check_input_array(degrees_of_freedom, 1, (const npy_intp[1]){total_dof_count}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "degrees_of_freedom"))
    {
        return NULL;
    }

    unsigned n_forms = 0;
    form_order_t *required_orders = NULL;
    mfv2d_result_t result = convert_system_forms(py_field_orders, &n_forms, &required_orders, &SYSTEM_ALLOCATOR);
    if (result != MFV2D_SUCCESS)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Failed to convert form orders, reason: %s",
                                     mfv2d_result_str(result));
        return NULL;
    }

    field_spec_t field_specs[INTEGRATING_FIELDS_MAX_COUNT];
    for (unsigned i = 0; i < n_fields; ++i)
    {
        field_spec_t *const this = &field_specs[i];
        PyObject *const field_information_item = PyTuple_GET_ITEM(field_information, i);

        if (PyCallable_Check(field_information_item))
        {
            this->callable = (field_spec_callable_t){
                .type = FIELD_SPEC_CALLABLE, .form_order = required_orders[i], .callable = field_information_item};
        }
        else
        {
            Py_ssize_t name_length;
            const char *unknown_name = PyUnicode_AsUTF8AndSize(field_information_item, &name_length);
            if (!unknown_name)
            {
                raise_exception_from_current(
                    PyExc_RuntimeError,
                    "Field entry %u is not a callable, but could also not be converted into unknown name.", i);
                deallocate(&SYSTEM_ALLOCATOR, required_orders);
                return NULL;
            }
            if (name_length == 0 || name_length > MAXIMUM_FORM_NAME_LENGTH)
            {
                PyErr_Format(PyExc_ValueError, "Field name must be a non-empty string of length <= %u, but got '%s'.",
                             MAXIMUM_FORM_NAME_LENGTH, unknown_name);
                deallocate(&SYSTEM_ALLOCATOR, required_orders);
                return NULL;
            }

            unsigned i_unknown;
            for (i_unknown = 0; i_unknown < Py_SIZE(form_specs); ++i_unknown)
            {
                if (strcmp(unknown_name, form_specs->forms[i_unknown].name) == 0)
                {
                    if (required_orders[i] != form_specs->forms[i_unknown].order)
                    {
                        PyErr_Format(PyExc_ValueError, "Field \"%s\" is of order %s, but order %s was requested.",
                                     unknown_name, form_order_str(form_specs->forms[i_unknown].order),
                                     form_order_str(required_orders[i]));
                        deallocate(&SYSTEM_ALLOCATOR, required_orders);
                        return NULL;
                    }

                    break;
                }
            }
            if (i_unknown == Py_SIZE(form_specs))
            {
                PyErr_Format(PyExc_ValueError, "Unknown field '%s'.", unknown_name);
                deallocate(&SYSTEM_ALLOCATOR, required_orders);
                return NULL;
            }
            this->unknown = (field_spec_unknown_t){
                .type = FIELD_SPEC_UNKNOWN, .form_order = required_orders[i], .index = i_unknown};
        }
    }

    field_values_t fields;
    result = compute_fields(element_space->fem_space, &element_space->corners, &fields, n_fields, field_specs,
                            &SYSTEM_ALLOCATOR, form_specs, PyArray_DATA(degrees_of_freedom));

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
        const form_order_t order = field_specs[i].form_order;
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
