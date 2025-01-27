//
// Created by jan on 23.11.2024.
//

#include "lineobject.h"

#include <stddef.h>

#include "geoidobject.h"
// This should be after other includes
#include <numpy/ndarrayobject.h>

static PyObject *line_object_repr(PyObject *self)
{
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("Line(GeoID(%u, %u), GeoID(%u, %u))", this->value.begin.index,
                                this->value.begin.reverse, this->value.end.index, this->value.end.reverse);
}

static PyObject *line_object_str(PyObject *self)
{
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("(%c%u -> %c%u)", this->value.begin.reverse ? '-' : '+', this->value.begin.index,
                                this->value.end.reverse ? '-' : '+', this->value.end.index);
}

line_object_t *line_from_indices(geo_id_t begin, geo_id_t end)
{
    line_object_t *const this = (line_object_t *)line_type_object.tp_alloc(&line_type_object, 0);
    if (!this)
        return NULL;
    this->value.begin = begin;
    this->value.end = end;

    return this;
}

static PyObject *line_object_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *a1, *a2;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char *[3]){"begin", "end", NULL}, &a1, &a2))
    {
        return NULL;
    }
    geo_id_t begin, end;
    if (geo_id_from_object(a1, &begin) < 0 || geo_id_from_object(a2, &end) < 0)
        return NULL;

    line_object_t *const this = (line_object_t *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    this->value.begin = begin;
    this->value.end = end;

    return (PyObject *)this;
}

static PyObject *line_object_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const line_object_t *const this = (line_object_t *)self;
    if (!PyObject_TypeCheck(other, &line_type_object))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const line_object_t *const that = (line_object_t *)other;
    const int val =
        geo_id_compare(this->value.begin, that->value.begin) && geo_id_compare(this->value.end, that->value.end);
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

PyDoc_STRVAR(line_object_type_docstring, "Line(begin: GeoID | int, end: GeoID | int)\n"
                                         "Geometrical object, which connects two points.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "begin : GeoID or int\n"
                                         "    ID of the point where the line beings.\n"
                                         "end : GeoID or int\n"
                                         "    ID of the point where the line ends.\n");

static PyObject *line_object_get_begin(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.begin);
}

static PyObject *line_object_get_end(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.end);
}

static PyGetSetDef line_object_getset[] = {
    {.name = "begin",
     .get = line_object_get_begin,
     .set = NULL,
     .doc = "GeoID : ID of the point where the line beings.",
     .closure = NULL},
    {.name = "end",
     .get = line_object_get_end,
     .set = NULL,
     .doc = "GeoID : ID of the point where the line ends.",
     .closure = NULL},
    {},
};

static PyObject *line_object_as_array(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *dtype = NULL;
    int b_copy = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Op", (char *[3]){"dtype", "copy", NULL}, &dtype, &b_copy))
    {
        return NULL;
    }

    if (!b_copy)
    {
        PyErr_SetString(PyExc_ValueError, "A copy is always created when converting to NDArray.");
        return NULL;
    }

    const line_object_t *this = (line_object_t *)self;
    const npy_intp size = 2;

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_INT);
    if (!out)
        return NULL;

    int *const ptr = PyArray_DATA(out);
    ptr[0] = geo_id_unpack(this->value.begin);
    ptr[1] = geo_id_unpack(this->value.end);

    if (dtype)
    {
        PyObject *const new_out = PyArray_CastToType(out, dtype, 0);
        Py_DECREF(out);
        return new_out;
    }

    return (PyObject *)out;
}

static PyMethodDef line_methods[] = {
    {.ml_name = "__array__",
     .ml_meth = (void *)line_object_as_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray"},
    {},
};

PyTypeObject line_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.Line",
    .tp_basicsize = sizeof(line_object_t),
    .tp_itemsize = 0,
    .tp_repr = line_object_repr,
    .tp_str = line_object_str,
    .tp_doc = line_object_type_docstring,
    .tp_new = line_object_new,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_richcompare = line_object_rich_compare,
    .tp_getset = line_object_getset,
    .tp_methods = line_methods,
};
