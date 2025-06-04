//
// Created by jan on 20.2.2025.
//

#include "bytecode.h"

#define MATRIX_OP_ENTRY(op) [op] = #op
static const char *matrix_op_strings[MATOP_COUNT] = {
    MATRIX_OP_ENTRY(MATOP_INVALID), MATRIX_OP_ENTRY(MATOP_IDENTITY),  MATRIX_OP_ENTRY(MATOP_MASS),
    MATRIX_OP_ENTRY(MATOP_MATMUL),  MATRIX_OP_ENTRY(MATOP_INCIDENCE), MATRIX_OP_ENTRY(MATOP_PUSH),
    MATRIX_OP_ENTRY(MATOP_SCALE),   MATRIX_OP_ENTRY(MATOP_SUM),
};
#undef MATRIX_OP_ENTRY

const char *matrix_op_str(const matrix_op_t op)
{
    if (op >= MATOP_COUNT)
        return "UNKNOWN";
    return matrix_op_strings[op];
}

MFV2D_INTERNAL
int convert_bytecode(const unsigned n, bytecode_t bytecode[restrict n + 1], PyObject *items[static n],
                     unsigned *p_max_stack, const unsigned n_vec_fields)
{
    bytecode[0].u32 = n;
    unsigned stack_load = 0, max_load = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const long val = PyLong_AsLong(items[i]);
        if (PyErr_Occurred())
        {
            return 0;
        }
        if (val <= MATOP_INVALID || val >= MATOP_COUNT)
        {
            PyErr_Format(PyExc_ValueError, "Invalid operation code %ld at position %zu.", val, i);
            return 0;
        }

        const matrix_op_t op = (matrix_op_t)val;
        bytecode[i + 1].op = op;

        int out_of_bounds = 0, bad_value = 0;
        switch (op)
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
                return 0;
            }
            stack_load -= 1;
            break;

        case MATOP_SCALE:
            if (n - i < 1)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].f64 = PyFloat_AsDouble(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        case MATOP_SUM:
            if (n - i < 1)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                const size_t n_sum = PyLong_AsUnsignedLong(items[i]);
                bytecode[i + 1].u32 = n_sum;
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                if (n_sum > stack_load)
                {
                    PyErr_Format(PyExc_ValueError, "Sum instruction for %zu matrices, but only %u are on stack", n_sum,
                                 stack_load);
                    return 0;
                }
                stack_load -= 1;
            }
            break;

        case MATOP_INCIDENCE:
            if (n - i < 2)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        case MATOP_MASS:
            if (n - i < 2)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        case MATOP_INTERPROD:
            if (n - i < 4)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                if (bytecode[i + 1].u32 >= n_vec_fields)
                {
                    out_of_bounds = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i + 1].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
            }
            break;

        default:
            PyErr_Format(PyExc_ValueError, "Invalid error code %u.", (unsigned)op);
            return 0;
        }

        if (out_of_bounds)
        {
            PyErr_Format(PyExc_ValueError, "Out of bounds for the required item.");
            return 0;
        }

        if (bad_value)
        {
            return 0;
        }
    }
    *p_max_stack = max_load;
    return 1;
}
