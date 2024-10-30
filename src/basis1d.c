//
// Created by jan on 22.10.2024.
//

#include "basis1d.h"
#include <numpy/arrayobject.h>
#include "common.h"

static PyObject *basis_call(PyObject *self, PyObject *Py_UNUSED(args), PyObject *Py_UNUSED(kwargs))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyObject *spline1d_derivative(PyObject *self, void* Py_UNUSED(args))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyObject *spline1d_antiderivative(PyObject *self, void* Py_UNUSED(args))
{
    PyErr_Format(PyExc_NotImplementedError, "Base type does not implement this method");
    return NULL;
}

static PyGetSetDef basis1d_get_set_def[] =
    {
        {.name = "derivative",  .get = spline1d_derivative, .set = NULL, .doc = "Return the derivative of the basis.", .closure = NULL},
        {.name = "antiderivative",  .get = spline1d_antiderivative, .set = NULL, .doc = "Return the derivative of the basis.", .closure = NULL},
        {NULL, NULL, NULL, NULL, NULL}, // sentinel
    };

static PyMethodDef poly_basis_methods[] =
    {
        {NULL, NULL, 0, NULL}, // sentinel
    };



static PyType_Slot basis1d_slots[] =
    {
    {.slot = Py_tp_call, .pfunc = basis_call},
    {.slot = Py_tp_methods, .pfunc = poly_basis_methods},
    {.slot = Py_tp_getset, .pfunc = basis1d_get_set_def},
    {.slot = 0, .pfunc = NULL}, // sentinel
    };

PyType_Spec basis1d_type_spec =
    {
        .name = "_interp.Basis1D",
        .basicsize = sizeof(PyObject),
        .itemsize = 0,
        .flags = Py_TPFLAGS_BASETYPE|Py_TPFLAGS_DEFAULT|Py_TPFLAGS_IMMUTABLETYPE,
        .slots = basis1d_slots,
    };
