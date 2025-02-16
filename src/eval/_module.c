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

static void caches_array_destroy(unsigned n, basis_precomp_t array[static n])
{
    for (unsigned i = n; i > 0; --i)
    {
        basis_precomp_destroy(array + (i - 1));
    }
}

static PyObject *compute_element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *ret_val = NULL;
    PyObject *in_form_orders;
    PyObject *in_expressions;
    PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
    PyObject *element_orders;
    PyObject *cache_contents;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO",
                                     (char *[9]){"form_orders", "expressions", "pos_bl", "pos_br", "pos_tr", "pos_tl",
                                                 "element_orders", "cache_contents", NULL},
                                     &in_form_orders, &in_expressions, &pos_bl, &pos_br, &pos_tr, &pos_tl,
                                     &element_orders, &cache_contents))
    {
        return NULL;
    }

    // Create the system template
    system_template_t system_template;
    if (!system_template_create(&system_template, in_form_orders, in_expressions, &SYSTEM_ALLOCATOR))
        return NULL;

    // Create caches
    PyObject *cache_seq = PySequence_Fast(cache_contents, "BasisCaches must be a sequence of tuples.");
    if (!cache_seq)
    {
        goto after_template;
    }
    const unsigned n_cache = PySequence_Fast_GET_SIZE(cache_seq);

    basis_precomp_t *cache_array = allocate(&SYSTEM_ALLOCATOR, sizeof *cache_array * n_cache);
    if (!cache_array)
    {
        Py_DECREF(cache_seq);
        goto after_template;
    }

    for (unsigned i = 0; i < n_cache; ++i)
    {
        int failed = !basis_precomp_create(PySequence_Fast_GET_ITEM(cache_seq, i), cache_array + i);
        if (!failed)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                if (cache_array[i].order == cache_array[j].order)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Cache contains the values for order as entries with indices %u and %u.", i, j);
                    failed = 1;
                    break;
                }
            }
        }

        if (failed)
        {
            caches_array_destroy(i, cache_array);
            deallocate(&SYSTEM_ALLOCATOR, cache_array);
            Py_DECREF(cache_seq);
            goto after_template;
        }
    }
    Py_DECREF(cache_seq);

    // Convert coordinate arrays
    PyArrayObject *const bl_array = (PyArrayObject *)PyArray_FromAny(pos_bl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const br_array = (PyArrayObject *)PyArray_FromAny(pos_br, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tr_array = (PyArrayObject *)PyArray_FromAny(pos_tr, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tl_array = (PyArrayObject *)PyArray_FromAny(pos_tl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const orders_array = (PyArrayObject *)PyArray_FromAny(
        element_orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!bl_array || !br_array || !tr_array || !tl_array || !orders_array)
    {
        Py_XDECREF(orders_array);
        Py_XDECREF(tl_array);
        Py_XDECREF(tr_array);
        Py_XDECREF(br_array);
        Py_XDECREF(bl_array);
        goto after_cache;
    }
    size_t n_elements;
    {
        n_elements = PyArray_DIM(orders_array, 0);
        const npy_intp *dims_bl = PyArray_DIMS(bl_array);
        const npy_intp *dims_br = PyArray_DIMS(br_array);
        const npy_intp *dims_tr = PyArray_DIMS(tr_array);
        const npy_intp *dims_tl = PyArray_DIMS(tl_array);
        if (dims_bl[0] != n_elements || (dims_bl[0] != dims_br[0] || dims_bl[1] != dims_br[1]) ||
            (dims_bl[0] != dims_tr[0] || dims_bl[1] != dims_tr[1]) ||
            (dims_bl[0] != dims_tl[0] || dims_bl[1] != dims_tl[1]) || dims_bl[1] != 2)
        {
            PyErr_SetString(PyExc_ValueError,
                            "All coordinate input arrays must be have same number of 2 component vectors.");
            goto after_arrays;
        }
    }

    // Extract C pointers

    const double *restrict const coord_bl = PyArray_DATA(bl_array);
    const double *restrict const coord_br = PyArray_DATA(br_array);
    const double *restrict const coord_tr = PyArray_DATA(tr_array);
    const double *restrict const coord_tl = PyArray_DATA(tl_array);
    const unsigned *restrict const orders = PyArray_DATA(orders_array);

    // Prepare output arrays

    ret_val = PyTuple_New((Py_ssize_t)n_elements);
    if (!ret_val)
    {
        goto after_arrays;
    }

    for (unsigned i = 0; i < n_elements; ++i)
    {
        size_t element_size = 0;
        for (unsigned j = 0; j < system_template.n_forms; ++j)
        {
            element_size += form_degrees_of_freedom_count(system_template.form_orders[j], orders[i]);
        }
        const npy_intp dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
        PyArrayObject *const a = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!a)
        {
            Py_DECREF(ret_val);
            ret_val = NULL;
            goto after_arrays;
        }
        PyTuple_SET_ITEM(ret_val, i, a);
    }

    // Clean up the coordinate arrays
after_arrays:
    Py_DECREF(orders_array);
    Py_DECREF(tl_array);
    Py_DECREF(tr_array);
    Py_DECREF(br_array);
    Py_DECREF(bl_array);

    // Clean up the basis caches
after_cache:
    caches_array_destroy(n_cache, cache_array);
    deallocate(&SYSTEM_ALLOCATOR, cache_array);

    // Clean up the template
after_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    return ret_val;
}

static PyObject *element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    double x0, x1, x2, x3;
    double y0, y1, y2, y3;
    PyObject *serialized;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddddddddO",
                                     (char *[10]){"x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3", "serialized", NULL},
                                     &x0, &x1, &x2, &x3, &y0, &y1, &y2, &y3, &serialized))
    {
        return NULL;
    }

    basis_precomp_t basis_precomp;
    if (!basis_precomp_create(serialized, &basis_precomp))
    {
        return NULL;
    }

    precompute_t out;
    const int res = precompute_create(&basis_precomp, x0, x1, x2, x3, y0, y1, y2, y3, &out, &SYSTEM_ALLOCATOR);
    basis_precomp_destroy(&basis_precomp);

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
    bytecode_val_t *bytecode = NULL;
    // Convert into bytecode
    {
        PyObject *const expression = PySequence_Fast(in_expression, "Can not convert expression to sequence.");
        if (!expression)
            return NULL;

        n_expr = PySequence_Fast_GET_SIZE(expression);
        PyObject **const p_exp = PySequence_Fast_ITEMS(expression);

        bytecode = PyMem_RawMalloc(sizeof(*bytecode) * (n_expr + 1));
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

    PyTupleObject *out_tuple = (PyTupleObject *)PyTuple_New((Py_ssize_t)n_expr);
    for (unsigned i = 1; i <= n_expr; ++i)
    {
        switch (bytecode[i].op)
        {
        case MATOP_IDENTITY:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_IDENTITY));
            break;
        case MATOP_TRANSPOSE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_TRANSPOSE));
            break;
        case MATOP_MATMUL:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MATMUL));
            break;
        case MATOP_SCALE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SCALE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].f64));
            break;
        case MATOP_SUM:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SUM));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].u32));
            break;
        case MATOP_INCIDENCE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_INCIDENCE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_MASS:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MASS));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_PUSH:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_PUSH));
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
