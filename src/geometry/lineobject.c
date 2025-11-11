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
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(self, state->type_line))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s.", state->type_line->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("Line(GeoID(%u, %u), GeoID(%u, %u))", this->value.begin.index,
                                this->value.begin.reverse, this->value.end.index, this->value.end.reverse);
}

static PyObject *line_object_str(PyObject *self)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(self, state->type_line))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s.", state->type_line->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("(%c%u -> %c%u)", this->value.begin.reverse ? '-' : '+', this->value.begin.index,
                                this->value.end.reverse ? '-' : '+', this->value.end.index);
}

line_object_t *line_from_indices(PyTypeObject *line_type_object, const geo_id_t begin, const geo_id_t end)
{
    line_object_t *const this = (line_object_t *)line_type_object->tp_alloc(line_type_object, 0);
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
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(type);
    if (!state)
        return NULL;

    if (geo_id_from_object(state->type_geoid, a1, &begin) < 0 || geo_id_from_object(state->type_geoid, a2, &end) < 0)
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
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(self, state->type_line))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s.", state->type_line->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }

    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const line_object_t *const this = (line_object_t *)self;
    if (!PyObject_TypeCheck(other, state->type_line))
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
                                         "Lines can also be converted into :mod:`numpy` arrays directly,\n"
                                         "which essentially converts their beginning and end indices into\n"
                                         "integers equivalent to their :class:`GeoID` values.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "begin : GeoID or int\n"
                                         "    ID of the point where the line beings.\n"
                                         "end : GeoID or int\n"
                                         "    ID of the point where the line ends.\n"
                                         "\n"
                                         "Examples\n"
                                         "--------\n"
                                         "This section just serves to briefly illustrate how a line can be used.\n"
                                         "\n"
                                         ".. jupyter-execute::\n"
                                         "\n"
                                         "    >>> import numpy as np\n"
                                         "    >>> from mfv2d._mfv2d import Line\n"
                                         "    >>> ln = Line(1, 2)\n"
                                         "    >>> print(ln)\n"
                                         "    >>> # this one has an invalid point\n"
                                         "    >>> ln2 = Line(0, 3)\n"
                                         "    >>> print(np.array(ln2))\n"
                                         "    >>> print(bool(ln2.begin))\n"
                                         "\n");

static PyObject *line_object_get_begin(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;
    return (PyObject *)geo_id_object_from_value(state->type_geoid, this->value.begin);
}

static PyObject *line_object_get_end(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;
    return (PyObject *)geo_id_object_from_value(state->type_geoid, this->value.end);
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

static PyObject *line_object_as_array(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                      const Py_ssize_t nargs, const PyObject *kwnames)
{
    PyArray_Descr *dtype = NULL;
    int b_copy = 1;
    if (parse_arguments_check(
            (argument_t[]){
                {.type = ARG_TYPE_PYTHON, .optional = 1, .p_val = (void *)&dtype, .kwname = "dtype"},
                {.type = ARG_TYPE_BOOL, .optional = 1, .kwname = "copy", .p_val = &b_copy},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    if (!b_copy)
    {
        PyErr_SetString(PyExc_ValueError, "A copy is always created when converting to NDArray.");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);

    if (!PyObject_TypeCheck(self, state->type_line))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s.", state->type_line->tp_name, Py_TYPE(self)->tp_name);
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
    {
        .ml_name = "__array__",
        .ml_meth = (void *)line_object_as_array,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray",
    },
    {},
};

static PyType_Slot line_type_slots[] = {
    {.slot = Py_tp_repr, .pfunc = line_object_repr},
    {.slot = Py_tp_str, .pfunc = line_object_str},
    {.slot = Py_tp_doc, .pfunc = (void *)line_object_type_docstring},
    {.slot = Py_tp_new, .pfunc = line_object_new},
    {.slot = Py_tp_richcompare, .pfunc = line_object_rich_compare},
    {.slot = Py_tp_getset, .pfunc = line_object_getset},
    {.slot = Py_tp_methods, .pfunc = line_methods},
    {}, // sentinel
};

PyType_Spec line_type_spec = {
    .name = "mfv2d._mfv2d.Line",
    .basicsize = sizeof(line_object_t),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE,
    .slots = line_type_slots,
};
