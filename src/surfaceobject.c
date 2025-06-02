//
// Created by jan on 24.11.2024.
//

#include "surfaceobject.h"
#include "geoidobject.h"
#include "lineobject.h"

// This should be last
#include <numpy/ndarrayobject.h>

static PyObject *surface_object_repr(PyObject *self)
{
    const surface_object_t *this = (surface_object_t *)self;

    PyObject *current_out = PyUnicode_FromString("Surface(");
    if (!current_out)
        return NULL;

    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        const geo_id_t id = this->lines[i];
        const char end = i + 1 == this->n_lines ? ')' : ' ';
        PyObject *repr = PyUnicode_FromFormat("GeoID(%u, %u),%c", id.index, id.reverse, end);
        if (!repr)
        {
            Py_DECREF(current_out);
            return NULL;
        }
        PyObject *new = PyUnicode_Concat(current_out, repr);
        Py_DECREF(repr);
        if (!new)
        {
            Py_DECREF(current_out);
            return NULL;
        }
        current_out = new;
    }

    return current_out;
}

static PyObject *surface_object_str(PyObject *self)
{
    const surface_object_t *this = (surface_object_t *)self;

    PyObject *current_out = PyUnicode_FromString("(");
    if (!current_out)
        return NULL;

    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        const geo_id_t id = this->lines[i];
        const char end = i + 1 == this->n_lines ? ')' : ' ';
        PyObject *repr = PyUnicode_FromFormat("%c%u ->%c", id.reverse ? '-' : '+', id.index, end);
        if (!repr)
        {
            Py_DECREF(current_out);
            return NULL;
        }
        PyObject *new = PyUnicode_Concat(current_out, repr);
        Py_DECREF(repr);
        if (!new)
        {
            Py_DECREF(current_out);
            return NULL;
        }
        current_out = new;
    }

    return current_out;
}

PyDoc_STRVAR(surface_object_type_docstring, "Surface(*ids)\n"
                                            "Two dimensional geometrical object, which is bound by lines.\n"
                                            "\n"
                                            "Parameters\n"
                                            "----------\n"
                                            "*ids : GeoID or int\n"
                                            "    Ids of the lines which are the boundary of the surface.\n");

static PyObject *surface_object_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const surface_object_t *const this = (surface_object_t *)self;
    if (!PyObject_TypeCheck(other, &surface_type_object))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const surface_object_t *const that = (surface_object_t *)other;
    int val = 1;
    if (this->n_lines != that->n_lines)
    {
        val = 0;
        goto ret;
    }
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        if (!geo_id_compare(this->lines[i], that->lines[i]))
        {
            val = 0;
            break;
            // goto ret;
        }
    }

ret:
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

static PyObject *surface_object_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Keywords were given to the constructor, but it takes none.");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(args, "Argument could not be converted into a sequence.");
    if (!seq)
    {
        return NULL;
    }
    const size_t n = PySequence_Fast_GET_SIZE(seq);

    surface_object_t *const this = (surface_object_t *)type->tp_alloc(type, (Py_ssize_t)n);

    if (!this)
    {
        Py_DECREF(seq);
        return NULL;
    }
    this->n_lines = n;

    for (unsigned i = 0; i < n; ++i)
    {
        if (geo_id_from_object(PySequence_Fast_GET_ITEM(seq, i), this->lines + i) < 0)
        {
            Py_DECREF(this);
            Py_DECREF(seq);
            return NULL;
        }
    }

    Py_DECREF(seq);
    return (PyObject *)this;
}

MFV2D_INTERNAL
surface_object_t *surface_object_empty(size_t count)
{
    return (surface_object_t *)surface_type_object.tp_alloc(&surface_type_object, (Py_ssize_t)count);
}

MFV2D_INTERNAL
surface_object_t *surface_object_from_value(size_t count, geo_id_t ids[static count])
{
    surface_object_t *const this = surface_object_empty(count);
    if (!this)
    {
        return NULL;
    }
    for (unsigned i = 0; i < count; ++i)
    {
        this->lines[i] = ids[i];
    }

    return this;
}

static PyObject *surface_object_as_array(PyObject *self, PyObject *args, PyObject *kwds)
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

    const surface_object_t *this = (surface_object_t *)self;
    const npy_intp size = (npy_intp)this->n_lines;

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_INT);
    if (!out)
        return NULL;

    int *const ptr = PyArray_DATA(out);
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        ptr[i] = geo_id_unpack(this->lines[i]);
    }

    if (dtype)
    {
        PyObject *const new_out = PyArray_CastToType(out, dtype, 0);
        Py_DECREF(out);
        return new_out;
    }

    return (PyObject *)out;
}

static PyMethodDef surface_methods[] = {
    {.ml_name = "__array__",
     .ml_meth = (void *)surface_object_as_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray[int]\n"
               "Convert to numpy array.\n"},
    {},
};

static Py_ssize_t surface_sequence_length(PyObject *self)
{
    const surface_object_t *this = (surface_object_t *)self;
    return (Py_ssize_t)this->n_lines;
}
static PyObject *surface_sequence_item(PyObject *self, Py_ssize_t idx)
{
    const surface_object_t *this = (surface_object_t *)self;
    // Correction: Python don't give a fuck, it just keeps on chucking idx in here until index error.
    if (idx < 0)
    {
        idx = (Py_ssize_t)(this->n_lines + 1) - idx;
    }
    if (idx >= this->n_lines)
    {
        PyErr_SetString(PyExc_IndexError, "Index is out of bounds.");
        return NULL;
    }

    return (PyObject *)geo_id_object_from_value(this->lines[idx]);
}

static PySequenceMethods surface_sequence_methods = {
    .sq_length = surface_sequence_length,
    .sq_item = surface_sequence_item,
};

MFV2D_INTERNAL
PyTypeObject surface_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.Surface",
    .tp_basicsize = sizeof(surface_object_t),
    .tp_itemsize = sizeof(geo_id_t),
    .tp_repr = surface_object_repr,
    .tp_str = surface_object_str,
    .tp_doc = surface_object_type_docstring,
    .tp_richcompare = surface_object_rich_compare,
    .tp_new = surface_object_new,
    .tp_methods = surface_methods,
    .tp_as_sequence = &surface_sequence_methods,
};
