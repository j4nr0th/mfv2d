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
#include "geoidobject.h"
#include "lineobject.h"
#include "surfaceobject.h"

#include "manifold.h"
#include "manifold1d.h"
#include "manifold2d.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyObject *test_method(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;

    return PyUnicode_FromString("Test successful!\n");
}

static PyMethodDef module_methods[] = {
    {"test", test_method, METH_NOARGS, "Test method that only returns a string."}, {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "interplib._mimetic",
                             .m_doc = "Internal C-extension implementing mimetic related functionality",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__mimetic(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject *mod = NULL;
    if (!((mod = PyModule_Create(&module))) || PyModule_AddType(mod, &geo_id_type_object) < 0 ||
        PyModule_AddType(mod, &line_type_object) < 0 || PyModule_AddType(mod, &surface_type_object) < 0 ||
        PyModule_AddType(mod, &manifold_type_object) < 0 || PyModule_AddType(mod, &manifold1d_type_object) < 0 ||
        PyModule_AddType(mod, &manifold2d_type_object) < 0)
    {
        Py_XDECREF(mod);
        return NULL;
    }

    return mod;
}
