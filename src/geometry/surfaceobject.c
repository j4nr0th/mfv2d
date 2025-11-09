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

    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        const geo_id_t id = this->lines[i];
        const char end = i + 1 == Py_SIZE(this) ? ')' : ' ';
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

    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        const geo_id_t id = this->lines[i];
        const char end = i + 1 == Py_SIZE(this) ? ')' : ' ';
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

PyDoc_STRVAR(surface_object_type_docstring,
             "Surface(*ids: GeoID | int)\n"
             "Two dimensional geometrical object, which is bound by lines.\n"
             "\n"
             "Since surface can contain a variable number of lines, it has methods\n"
             "based on containers, such as ``len``, which allow for iterations.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*ids : GeoID or int\n"
             "    Ids of the lines which are the boundary of the surface.\n"
             "\n"
             "Examples\n"
             "--------\n"
             "Some examples of what can be done with surfaces are presented here.\n"
             "\n"
             "First, the length of the surface can be obtained by using :func:`len` build-in.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> import numpy as np\n"
             "    >>> from mfv2d._mfv2d import GeoID, Surface\n"
             "    >>> \n"
             "    >>> surf = Surface(1, 2, 3, -4)\n"
             "    >>> len(surf)\n"
             "    4\n"
             "\n"
             "Next, the surface can be iterated over:\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> for gid in surf:\n"
             "    ...     print(gid)\n"
             "\n"
             "The surface can also be converted into a :mod:`numpy` array directly.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> print(np.array(surf))\n"
             "\n");

static PyObject *surface_object_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const surface_object_t *const this = (surface_object_t *)self;
    if (!PyObject_TypeCheck(other, Py_TYPE(self)))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const surface_object_t *const that = (surface_object_t *)other;
    int val = 1;
    if (Py_SIZE(this) != Py_SIZE(that))
    {
        val = 0;
        goto ret;
    }
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
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

static PyObject *surface_object_new(PyTypeObject *type, PyObject *args, const PyObject *kwds)
{
    const mfv2d_module_state_t *state = PyType_GetModuleState(type);
    if (!state)
        return NULL;

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
    Py_SET_SIZE(this, n);

    for (unsigned i = 0; i < n; ++i)
    {
        if (geo_id_from_object(state->type_geoid, PySequence_Fast_GET_ITEM(seq, i), this->lines + i) < 0)
        {
            Py_DECREF(this);
            Py_DECREF(seq);
            return NULL;
        }
    }

    Py_DECREF(seq);
    return (PyObject *)this;
}

static surface_object_t *surface_object_empty(PyTypeObject *surface_type_object, const size_t count)
{
    return (surface_object_t *)surface_type_object->tp_alloc(surface_type_object, (Py_ssize_t)count);
}

MFV2D_INTERNAL
surface_object_t *surface_object_from_value(PyTypeObject *surface_type_object, const size_t count,
                                            geo_id_t ids[static count])
{
    surface_object_t *const this = surface_object_empty(surface_type_object, count);
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
    const npy_intp size = Py_SIZE(this);

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_INT);
    if (!out)
        return NULL;

    int *const ptr = PyArray_DATA(out);
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
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
    return Py_SIZE(this);
}
static PyObject *surface_sequence_item(PyObject *self, Py_ssize_t idx)
{
    const surface_object_t *this = (surface_object_t *)self;
    const mfv2d_module_state_t *state = PyType_GetModuleState(Py_TYPE(self));
    if (!state)
        return NULL;

    // Correction: Python don't give a fuck, it just keeps on chucking idx in here until the index error.
    if (idx < 0)
    {
        idx = Py_SIZE(this) + 1 - idx;
    }
    if (idx >= Py_SIZE(this))
    {
        PyErr_SetString(PyExc_IndexError, "Index is out of bounds.");
        return NULL;
    }

    return (PyObject *)geo_id_object_from_value(state->type_geoid, this->lines[idx]);
}

// static PySequenceMethods surface_sequence_methods = {
//     .sq_length = surface_sequence_length,
//     .sq_item = surface_sequence_item,
// };
//
// MFV2D_INTERNAL
// PyTypeObject surface_type_object = {
//     .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.Surface",
//     .tp_basicsize = sizeof(surface_object_t),
//     .tp_itemsize = sizeof(geo_id_t),
//     .tp_repr = surface_object_repr,
//     .tp_str = surface_object_str,
//     .tp_doc = surface_object_type_docstring,
//     .tp_richcompare = surface_object_rich_compare,
//     .tp_new = surface_object_new,
//     .tp_methods = surface_methods,
//     .tp_as_sequence = &surface_sequence_methods,
// };

static PyType_Slot surface_slots[] = {
    {.slot = Py_tp_repr, .pfunc = surface_object_repr},
    {.slot = Py_tp_str, .pfunc = surface_object_str},
    {.slot = Py_tp_doc, .pfunc = (void *)surface_object_type_docstring},
    {.slot = Py_tp_richcompare, .pfunc = surface_object_rich_compare},
    {.slot = Py_tp_new, .pfunc = surface_object_new},
    {.slot = Py_tp_methods, .pfunc = surface_methods},
    {.slot = Py_sq_length, .pfunc = surface_sequence_length},
    {.slot = Py_sq_item, .pfunc = surface_sequence_item},
    {} // sentinel
};

PyType_Spec surface_type_spec = {
    .name = "mfv2d._mfv2d.Surface",
    .basicsize = sizeof(surface_object_t),
    .itemsize = sizeof(geo_id_t),
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = surface_slots,
};
