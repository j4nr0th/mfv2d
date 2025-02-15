//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#include "../common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Internal headers
#include "evaluation.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyObject *test_method(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;

    return PyUnicode_FromString("Test successful!\n");
}

static int convert_bytecode(unsigned n, bytecode_t bytecode[restrict n], PyObject *items[static n])
{
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
        bytecode[i].op = op;

        int out_of_bounds = 0, bad_value = 0;
        switch (op)
        {
        case MATOP_IDENTITY:
            break;

        case MATOP_PUSH:
            break;

        case MATOP_TRANSPOSE:
            break;

        case MATOP_MATMUL:
            break;

        case MATOP_SCALE:
            if (n - i < 1)
            {
                out_of_bounds = 1;
            }
            else
            {
                i += 1;
                bytecode[i].f64 = PyFloat_AsDouble(items[i]);
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
                bytecode[i].u32 = PyLong_AsUnsignedLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
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
                bytecode[i].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i].u32 = PyLong_AsLong(items[i]);
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
                bytecode[i].u32 = PyLong_AsLong(items[i]);
                if (PyErr_Occurred())
                {
                    bad_value = 1;
                    break;
                }
                i += 1;
                bytecode[i].u32 = PyLong_AsLong(items[i]);
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

    return 1;
}

static PyObject *compute_element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *in_expression;
    PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
    PyObject *element_orders;
    PyObject *cache_contents;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOOOOO",
            (char *[8]){"expression", "pos_bl", "pos_br", "pos_tr", "pos_tl", "element_orders", "cache_contents", NULL},
            &in_expression, &pos_bl, &pos_br, &pos_tr, &pos_tl, &element_orders))
    {
        return NULL;
    }
    size_t n_expr;
    bytecode_t *bytecode = NULL;
    // Convert into bytecode
    {
        PyObject *const expression = PySequence_Fast(in_expression, "Can not convert expression to sequence.");
        if (!expression)
            return NULL;

        n_expr = PySequence_Fast_GET_SIZE(expression);
        PyObject **const p_exp = PySequence_Fast_ITEMS(expression);

        bytecode = PyMem_RawMalloc(sizeof(*bytecode) * n_expr);
        if (!bytecode)
        {
            Py_DECREF(expression);
            return NULL;
        }

        if (!convert_bytecode(n_expr, bytecode, p_exp))
        {
            PyMem_RawFree(bytecode);
            Py_DECREF(expression);
            return NULL;
        }
        Py_DECREF(expression);
    }

    // TODO: complete

    PyMem_RawFree(bytecode);

    return NULL;
}

static PyObject *element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    double x0, x1, x2, x3;
    double y0, y1, y2, y3;
    int order, n_int;
    PyObject *int_nodes, *node_precomp, *edge_00_precomp, *edge_01_precomp, *edge_11_precomp, *surface_precomp;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddddddddiiOOOOOO",
                                     (char *[17]){"x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3", "order", "n_int",
                                                  "int_nodes", "node_precomp", "edge_00_precomp", "edge_01_precomp",
                                                  "edge_11_precomp", "surface_precomp", NULL},
                                     &x0, &x1, &x2, &x3, &y0, &y1, &y2, &y3, &order, &n_int, &int_nodes, &node_precomp,
                                     &edge_00_precomp, &edge_01_precomp, &edge_11_precomp, &surface_precomp))
    {
        return NULL;
    }

    PyArrayObject *const arr_int_nodes = (PyArrayObject *)PyArray_FromAny(
        int_nodes, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_node_precomp = (PyArrayObject *)PyArray_FromAny(
        node_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    ASSERT(PyArray_DIM(arr_int_nodes, 0) == n_int, "Integration order mismatch is too low.");
    PyArrayObject *const arr_edge_00_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_00_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_edge_01_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_01_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_edge_11_precomp = (PyArrayObject *)PyArray_FromAny(
        edge_11_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    PyArrayObject *const arr_surface_precomp = (PyArrayObject *)PyArray_FromAny(
        surface_precomp, PyArray_DescrFromType(NPY_DOUBLE), 4, 4, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);

    if (!arr_int_nodes || !arr_node_precomp || !arr_edge_00_precomp || !arr_edge_01_precomp || !arr_edge_11_precomp ||
        !arr_surface_precomp)
    {
        Py_XDECREF(arr_int_nodes);
        Py_XDECREF(arr_node_precomp);
        Py_XDECREF(arr_edge_00_precomp);
        Py_XDECREF(arr_edge_01_precomp);
        Py_XDECREF(arr_edge_11_precomp);
        Py_XDECREF(arr_surface_precomp);
        return NULL;
    }

    const basis_precomp_t basis_precomp = {
        .order = order,
        .n_int = n_int,
        .nodes_int = PyArray_DATA(arr_int_nodes),
        .mass_nodal = PyArray_DATA(arr_node_precomp),
        .mass_edge_00 = PyArray_DATA(arr_edge_00_precomp),
        .mass_edge_01 = PyArray_DATA(arr_edge_01_precomp),
        .mass_edge_11 = PyArray_DATA(arr_edge_11_precomp),
        .mass_surf = PyArray_DATA(arr_surface_precomp),
    };

    precompute_t out;
    const int res = precompute_create(&basis_precomp, x0, x1, x2, x3, y0, y1, y2, y3, &out, &SYSTEM_ALLOCATOR);
    Py_DECREF(arr_int_nodes);
    Py_DECREF(arr_node_precomp);
    Py_DECREF(arr_edge_00_precomp);
    Py_DECREF(arr_edge_01_precomp);
    Py_DECREF(arr_edge_11_precomp);
    Py_DECREF(arr_surface_precomp);

    if (!res)
    {
        return NULL;
    }

    PyObject *ret_val = PyTuple_Pack(
        6, matrix_full_to_array(out.mass_matrices + MASS_0), matrix_full_to_array(out.mass_matrices + MASS_1),
        matrix_full_to_array(out.mass_matrices + MASS_2), matrix_full_to_array(out.mass_matrices + MASS_0_I),
        matrix_full_to_array(out.mass_matrices + MASS_1_I), matrix_full_to_array(out.mass_matrices + MASS_2_I));

    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2_I].data);

    return ret_val;
}

static PyObject *check_bytecode(PyObject *Py_UNUSED(module), PyObject *in_expression)
{
    size_t n_expr;
    bytecode_t *bytecode = NULL;
    // Convert into bytecode
    {
        PyObject *const expression = PySequence_Fast(in_expression, "Can not convert expression to sequence.");
        if (!expression)
            return NULL;

        n_expr = PySequence_Fast_GET_SIZE(expression);
        PyObject **const p_exp = PySequence_Fast_ITEMS(expression);

        bytecode = PyMem_RawMalloc(sizeof(*bytecode) * n_expr);
        if (!bytecode)
        {
            Py_DECREF(expression);
            return NULL;
        }

        if (!convert_bytecode(n_expr, bytecode, p_exp))
        {
            PyMem_RawFree(bytecode);
            Py_DECREF(expression);
            return NULL;
        }
        Py_DECREF(expression);
    }

    PyTupleObject *out_tuple = (PyTupleObject *)PyTuple_New(n_expr);
    for (unsigned i = 0; i < n_expr; ++i)
    {
        switch (bytecode[i].op)
        {
        case MATOP_IDENTITY:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_IDENTITY));
            break;
        case MATOP_TRANSPOSE:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_TRANSPOSE));
            break;
        case MATOP_MATMUL:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_MATMUL));
            break;
        case MATOP_SCALE:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_SCALE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyFloat_FromDouble(bytecode[i].f64));
            break;
        case MATOP_SUM:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_SUM));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyFloat_FromDouble(bytecode[i].u32));
            break;
        case MATOP_INCIDENCE:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_INCIDENCE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_MASS:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_MASS));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_PUSH:
            PyTuple_SET_ITEM(out_tuple, i, PyLong_FromLong(MATOP_PUSH));
            break;
        default:
            ASSERT(0, "Invalid operation.");
            break;
        }
    }

    PyMem_RawFree(bytecode);

    return (PyObject *)out_tuple;
}

static PyMethodDef module_methods[] = {
    {"test", test_method, METH_NOARGS, "Test method that only returns a string."},
    {"compute_element_matrices", (void *)compute_element_matrices, METH_VARARGS | METH_KEYWORDS,
     "Compute element matrices."},
    {"element_matrices", (void *)element_matrices, METH_VARARGS | METH_KEYWORDS, "Compute element matrices."},
    {"check_bytecode", check_bytecode, METH_O, "Convert bytecode to C-values, then back to Python."},
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "interplib._eval",
                             .m_doc = "Internal C-extension implementing element matrix evaluation.",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__eval(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject *mod = PyModule_Create(&module);
    if (!mod)
    {
        return NULL;
    }

    return mod;
}
