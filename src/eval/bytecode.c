//
// Created by jan on 20.2.2025.
//

#include "bytecode.h"
#include "integrating_fields.h"

#include "matrix.h"
#include "system_template.h"

#define MATRIX_OP_ENTRY(op) [op] = #op
static const char *matrix_op_strings[MATOP_COUNT] = {
    MATRIX_OP_ENTRY(MATOP_INVALID), MATRIX_OP_ENTRY(MATOP_IDENTITY),  MATRIX_OP_ENTRY(MATOP_MASS),
    MATRIX_OP_ENTRY(MATOP_MATMUL),  MATRIX_OP_ENTRY(MATOP_INCIDENCE), MATRIX_OP_ENTRY(MATOP_PUSH),
    MATRIX_OP_ENTRY(MATOP_SCALE),   MATRIX_OP_ENTRY(MATOP_SUM),
};
#undef MATRIX_OP_ENTRY

const char *matrix_op_type_str(const matrix_op_type_t op)
{
    if (op >= MATOP_COUNT)
        return "UNKNOWN";
    return matrix_op_strings[op];
}

static int check_instruction_count(const unsigned available, const unsigned required, const matrix_op_type_t type)
{
    if (available < required)
    {
        PyErr_Format(PyExc_ValueError,
                     "Not enough instructions available for the required instruction type %s, which requires %u, but "
                     "only %u are left.",
                     matrix_op_type_str(type), required, available);
        return 0;
    }
    return 1;
}

/**
 * Converts a Python object to a value of type `form_order_t`.
 *
 * @param o A pointer to the Python object to be converted.
 * @param p_order A pointer to a `form_order_t` variable where the converted value will be stored.
 * @return Returns 1 if the conversion was successful, otherwise returns 0.
 */
static int converter_form_order(PyObject *const o, form_order_t *p_order)
{
    const long val = PyLong_AsLong(o);
    if (PyErr_Occurred())
    {
        return 0;
    }
    if (val < FORM_ORDER_0 || val > FORM_ORDER_2)
    {
        PyErr_Format(PyExc_ValueError, "Invalid order %ld (allowed range is %u to %u).", val, FORM_ORDER_0,
                     FORM_ORDER_2);
        return 0;
    }
    *p_order = (form_order_t)val;
    return 1;
}

static mfv2d_result_t convert_scale(PyObject *const o, matrix_op_scale_t *const out)
{
    matrix_op_type_t type;
    double k;
    if (!PyArg_ParseTuple(o, "O&d", matrix_op_type_from_object, &type, &k))
    {
        return MFV2D_BAD_ARGUMENT;
    }
    if (type != MATOP_SCALE)
    {
        PyErr_Format(PyExc_ValueError, "Expected scale operation, but got %s.", matrix_op_type_str(type));
        return MFV2D_BAD_ARGUMENT;
    }
    *out = (matrix_op_scale_t){.k = k, .op = MATOP_SCALE};
    return MFV2D_SUCCESS;
}

static mfv2d_result_t convert_sum(PyObject *const o, matrix_op_sum_t *const out, unsigned *const p_stack_load)
{
    matrix_op_type_t type;
    unsigned n;
    if (!PyArg_ParseTuple(o, "O&I", matrix_op_type_from_object, &type, &n))
    {
        return MFV2D_BAD_ARGUMENT;
    }
    if (type != MATOP_SUM)
    {
        PyErr_Format(PyExc_ValueError, "Expected sum operation, but got %s.", matrix_op_type_str(type));
        return MFV2D_BAD_ARGUMENT;
    }
    if (n == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Sum operation with zero matrices.");
        return MFV2D_BAD_ARGUMENT;
    }
    if (n > *p_stack_load)
    {
        PyErr_Format(PyExc_ValueError, "Sum instruction for %u matrices, but only %u are on stack", n, *p_stack_load);
        return MFV2D_BAD_ARGUMENT;
    }
    *p_stack_load -= n;
    *out = (matrix_op_sum_t){.n = n, .op = MATOP_SUM};
    return MFV2D_SUCCESS;
}

static mfv2d_result_t convert_incidence(PyObject *const o, matrix_op_incidence_t *const out)
{
    matrix_op_type_t type;
    form_order_t order;
    int transpose;
    if (!PyArg_ParseTuple(o, "O&O&p", matrix_op_type_from_object, &type, converter_form_order, &order, &transpose))
    {
        return MFV2D_BAD_ARGUMENT;
    }
    if (type != MATOP_INCIDENCE)
    {
        PyErr_Format(PyExc_ValueError, "Expected incidence operation, but got %s.", matrix_op_type_str(type));
        return MFV2D_BAD_ARGUMENT;
    }
    *out = (matrix_op_incidence_t){.order = order, .transpose = transpose, .op = MATOP_INCIDENCE};
    return MFV2D_SUCCESS;
}

static mfv2d_result_t convert_mass(PyObject *const o, matrix_op_mass_t *const out)
{
    matrix_op_type_t type;
    form_order_t order;
    int invert;
    if (!PyArg_ParseTuple(o, "O&O&p", matrix_op_type_from_object, &type, converter_form_order, &order, &invert))
    {
        return MFV2D_BAD_ARGUMENT;
    }
    if (type != MATOP_MASS)
    {
        PyErr_Format(PyExc_ValueError, "Expected mass operation, but got %s.", matrix_op_type_str(type));
        return MFV2D_BAD_ARGUMENT;
    }
    *out = (matrix_op_mass_t){.order = order, .invert = invert, .op = MATOP_MASS};
    return MFV2D_SUCCESS;
}

static mfv2d_result_t convert_interprod(PyObject *const o, matrix_op_interprod_t *const out, unsigned field_cnt,
                                        form_order_t p_form_orders[INTEGRATING_FIELDS_MAX_COUNT])
{
    matrix_op_type_t type;
    form_order_t start_order;
    unsigned field_index;
    int dual;
    int adjoint;
    if (!PyArg_ParseTuple(o, "O&O&Ipp", matrix_op_type_from_object, &type, converter_form_order, &start_order,
                          &field_index, &dual, &adjoint))
    {
        return MFV2D_BAD_ARGUMENT;
    }
    if (field_index >= INTEGRATING_FIELDS_MAX_COUNT)
    {
        PyErr_Format(PyExc_ValueError, "Field index %u is out of bounds, maximum %u can be specified.", field_index,
                     INTEGRATING_FIELDS_MAX_COUNT);
        return MFV2D_BAD_ARGUMENT;
    }

    if (field_cnt <= field_index)
    {
        PyErr_Format(PyExc_ValueError,
                     "Field index %u is out of bounds, as specifications for only %u fields were given.", field_index,
                     field_cnt);
        return MFV2D_BAD_ARGUMENT;
    }
    form_order_t required_order;
    // Determine the field's order based on what operation it is.
    if (adjoint)
    {
        if (dual)
        {
            if (start_order == FORM_ORDER_1)
            {
                required_order = FORM_ORDER_1;
            }
            else // start_order == FORM_ORDER_2
            {
                required_order = FORM_ORDER_0;
            }
        }
        else
        {
            if (start_order == FORM_ORDER_1)
            {
                required_order = FORM_ORDER_1;
            }
            else // start_order == FORM_ORDER_2
            {
                required_order = FORM_ORDER_2;
            }
        }
    }
    else
    {
        required_order = FORM_ORDER_1;
    }
    if (p_form_orders[field_index] != required_order)
    {
        if (p_form_orders[field_index] == FORM_ORDER_UNKNOWN)
        {
            p_form_orders[field_index] = required_order;
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "Field %u is of order %u, but the required order is %u.", field_index,
                         required_order, p_form_orders[field_index]);
            return MFV2D_BAD_ARGUMENT;
        }
    }
    *out = (matrix_op_interprod_t){
        .order = start_order, .field_index = field_index, .dual = dual, .adjoint = adjoint, .op = MATOP_INTERPROD};
    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
mfv2d_result_t convert_bytecode(const unsigned n, matrix_op_t ops[restrict const n], PyObject *const items[static n],
                                unsigned *const p_max_stack, unsigned field_cnt,
                                form_order_t p_form_orders[INTEGRATING_FIELDS_MAX_COUNT])
{
    unsigned stack_load = 0, max_load = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (!PyTuple_Check(items[i]))
        {
            PyErr_Format(PyExc_ValueError, "Expected tuple for instruction %u, but got %s.", (unsigned)i,
                         Py_TYPE(items[i])->tp_name);
            return MFV2D_BAD_ARGUMENT;
        }

        if (PyTuple_Size(items[i]) == 0)
        {
            PyErr_Format(PyExc_ValueError, "Empty tuple for instruction %u.", (unsigned)i);
            return MFV2D_BAD_ARGUMENT;
        }

        matrix_op_type_t op_type;
        if (!matrix_op_type_from_object(PyTuple_GET_ITEM(items[i], 0), &op_type))
            return MFV2D_BAD_ARGUMENT;

        matrix_op_t *const op_ptr = ops + i;
        op_ptr->type = op_type;

        mfv2d_result_t res = MFV2D_SUCCESS;
        switch (op_type)
        {
        case MATOP_IDENTITY:
            break;

        case MATOP_PUSH:
            stack_load += 1;
            if (stack_load > max_load)
            {
                max_load = stack_load;
            }
            break;

        case MATOP_MATMUL:
            if (stack_load == 0)
            {
                PyErr_SetString(PyExc_ValueError, "Matmul instruction with nothing on the stack.");
                return MFV2D_BAD_ARGUMENT;
            }
            stack_load -= 1;
            break;

        case MATOP_SCALE:
            res = convert_scale(items[i], &op_ptr->scale);
            break;

        case MATOP_SUM:
            res = convert_sum(items[i], &op_ptr->sum, &stack_load);
            break;

        case MATOP_INCIDENCE:
            res = convert_incidence(items[i], &op_ptr->incidence);
            break;

        case MATOP_MASS:
            res = convert_mass(items[i], &op_ptr->mass);
            break;

        case MATOP_INTERPROD:
            res = convert_interprod(items[i], &op_ptr->interprod, field_cnt, p_form_orders);
            break;

        default:
            PyErr_Format(PyExc_ValueError, "Invalid error code %u.", (unsigned)op_type);
            return MFV2D_BAD_ENUM;
        }

        if (res != MFV2D_SUCCESS)
            return res;
    }

    *p_max_stack = max_load;
    return MFV2D_SUCCESS;
}

int matrix_op_type_from_object(PyObject *const o, matrix_op_type_t *const out)
{
    const long val = PyLong_AsLong(o);
    if (PyErr_Occurred())
    {
        raise_exception_from_current(PyExc_ValueError, "Could not convert value \"%R\" to the correct type.", o);
        return 0;
    }
    if (val <= MATOP_INVALID || val >= MATOP_COUNT)
    {
        PyErr_Format(PyExc_ValueError, "Invalid operation code %ld (allowed range is %u to %u).", val, MATOP_INVALID,
                     MATOP_COUNT);
        return 0;
    }
    *out = (matrix_op_type_t)val;
    return 1;
}
