//
// Created by jan on 22.10.2024.
//

#include "basis1d.h"

#include <numpy/arrayobject.h>

#include "common.h"

static PyObject *basis1d_call(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args), PyObject *Py_UNUSED(kwargs))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyObject *basis1d_derivative(PyObject *Py_UNUSED(self), void *Py_UNUSED(args))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyObject *basis1d_antiderivative(PyObject *Py_UNUSED(self), void *Py_UNUSED(args))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyGetSetDef basis1d_get_set_def[] = {
    {.name = "derivative",
     .get = basis1d_derivative,
     .set = NULL,
     .doc = "Basis1D : Return the derivative of the basis.",
     .closure = NULL},
    {.name = "antiderivative",
     .get = basis1d_antiderivative,
     .set = NULL,
     .doc = "Basis1D : Return the derivative of the basis.",
     .closure = NULL},
    {NULL, NULL, NULL, NULL, NULL}, // sentinel
};

PyDoc_STRVAR(basis1d_docstr, "Basis1D\n"
                             "\n"
                             "Abstract class for 1D basis objects.\n");

INTERPLIB_INTERNAL
PyTypeObject basis1d_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._interp.Basis1D",
    .tp_basicsize = sizeof(PyObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DEFAULT,
    .tp_doc = basis1d_docstr,
    .tp_getset = basis1d_get_set_def,
    .tp_new = PyType_GenericNew,
    .tp_call = basis1d_call,
};
