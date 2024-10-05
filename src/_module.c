//
// Created by jan on 29.9.2024.
//

//  Common definitions
#include "common_defines.h"



//  Python
#include <Python.h>
//  Numpy
#include <numpy/npy_no_deprecated_api.h>
#include <numpy/ndarrayobject.h>

//  Internal headers
#include "lagrange.h"


#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": "  fmt "\n", (expr))

static PyObject *interp_lagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    PyArrayObject *yp;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x, &PyArray_Type, &xp, &PyArray_Type, &yp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");
    ASSERT(PyArray_TYPE(yp) == NPY_DOUBLE, "Incorrect type for array yp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");
    ASSERT(PyArray_NDIM(yp) == 1, "Incorrect shape for array yp");


    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_SHAPE(x), PyArray_TYPE(x));
    if (!out)
    {
        return NULL;
    }
    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);
    double* working_memory = PyMem_Malloc(2 * n_nodes * sizeof(*working_memory));
    if (working_memory == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    interp_error_t interp_res = lagrange_interpolation_init(n_nodes, PyArray_DATA(xp), PyArray_DATA(yp), working_memory);

    if (interp_res != INTERP_SUCCESS)
    {
        printf("%s:%u\n", __FILE__, __LINE__);
        PyMem_Free(working_memory);
        Py_DECREF(out);
        return PyErr_Format(
            PyExc_RuntimeError,
            "Error occurred during interpolation (%s): %s",
            interp_error_str(interp_res),
            interp_error_msg(interp_res)
        );
    }

    const double* const p_x = (double*)PyArray_DATA(x);
    double* const p_out = (double*)PyArray_DATA(out);

    for (unsigned i = 0; i < n_pts; i++)
    {
        interp_res = lagrange_interpolation_evaluate(
            n_nodes,
            PyArray_DATA(xp),
            working_memory,
            p_x[i],
            p_out + i,
            working_memory + 1 * n_nodes
        );
        ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");
    }

    PyMem_Free(working_memory);
    return (PyObject*)PyArray_Return(out);
}

static PyObject *interp_dlagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    PyArrayObject *yp;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x, &PyArray_Type, &xp, &PyArray_Type, &yp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");
    ASSERT(PyArray_TYPE(yp) == NPY_DOUBLE, "Incorrect type for array yp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");
    ASSERT(PyArray_NDIM(yp) == 1, "Incorrect shape for array yp");


    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_SHAPE(x), PyArray_TYPE(x));
    if (!out)
    {
        return NULL;
    }
    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);


    const double* const p_x = (double*)PyArray_DATA(x);
    double* const p_out = (double*)PyArray_DATA(out);
    for (unsigned i = 0; i < n_pts; i++)
    {
        const interp_error_t interp_res = dlagrange_interpolation(
            n_nodes,
            PyArray_DATA(xp),
            PyArray_DATA(yp),
            p_x[i],
            p_out + i
        );
        ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");
    }

    return (PyObject*)PyArray_Return(out);
}

PyDoc_STRVAR(interp_lagrange_doc,"lagrange1d(x: array, xp: array, yp: array) -> array");


static PyObject *test_method(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;

    return PyUnicode_FromString("Test successful!\n");
}

static PyMethodDef module_methods[] =
{
    {"test", test_method, METH_NOARGS, "Test method that only returns a string."},
    {"lagrange1d", interp_lagrange, METH_VARARGS, interp_lagrange_doc},
    {"dlagrange1d", interp_dlagrange, METH_VARARGS, interp_lagrange_doc},
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_interp",
    .m_doc = "Internal C-extension implementing interpolation functions",
    .m_size = -1,
    .m_methods = module_methods,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};


PyMODINIT_FUNC PyInit__interp(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject* mod = PyModule_Create(&module);

    return mod;
}

