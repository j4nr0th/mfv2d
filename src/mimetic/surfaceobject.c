//
// Created by jan on 24.11.2024.
//

#include "surfaceobject.h"

#include "geoidobject.h"
#include "lineobject.h"

static PyObject *surface_object_repr(PyObject *self)
{
    const surface_object_t *this = (surface_object_t *)self;

    return PyUnicode_FromFormat("Surface(GeoID(%u, %u), GeoID(%u, %u), GeoID(%u, %u), GeoID(%u, %u))",
                                this->value.bottom.index, this->value.bottom.reverse, this->value.right.index,
                                this->value.right.reverse, this->value.top.index, this->value.top.reverse,
                                this->value.left.index, this->value.left.reverse);
}

static PyObject *surface_object_str(PyObject *self)
{
    const surface_object_t *this = (surface_object_t *)self;
    return PyUnicode_FromFormat("(%c%u -> %c%u -> %c%u -> %c%u)", this->value.bottom.reverse ? '-' : '+',
                                this->value.bottom.index, this->value.right.reverse ? '-' : '+',
                                this->value.right.index, this->value.top.reverse ? '-' : '+', this->value.top.index,
                                this->value.left.reverse ? '-' : '+', this->value.left.index);
}

PyDoc_STRVAR(surface_object_type_docstring, "Geometrical object, which is bound by four lines.\n"
                                            "\n"
                                            "Parameters\n"
                                            "----------\n"
                                            "bottom : GeoID\n"
                                            "    Bottom boundary of the surface.\n"
                                            "right : GeoID\n"
                                            "    Right boundary of the surface.\n"
                                            "top : GeoID\n"
                                            "    Top boundary of the surface.\n"
                                            "left : GeoID\n"
                                            "    Left boundary of the surface.\n");

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
    const int val = geo_id_compare(this->value.bottom, that->value.bottom) &&
                    geo_id_compare(this->value.right, that->value.right) &&
                    geo_id_compare(this->value.top, that->value.top) &&
                    geo_id_compare(this->value.left, that->value.left);

    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

static PyObject *surface_object_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *arg_vals[4];
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", (char *[5]){"bottom", "right", "top", "left", NULL},
                                     arg_vals + 0, arg_vals + 1, arg_vals + 2, arg_vals + 3))
    {
        return NULL;
    }
    surface_object_t *const this = (surface_object_t *)type->tp_alloc(type, 0);

    if (!this)
        return NULL;

    if (geo_id_from_object(arg_vals[0], &this->value.bottom) < 0 ||
        geo_id_from_object(arg_vals[1], &this->value.right) < 0 ||
        geo_id_from_object(arg_vals[2], &this->value.top) < 0 || geo_id_from_object(arg_vals[3], &this->value.left) < 0)
    {
        Py_DECREF(this);
        return NULL;
    }

    return (PyObject *)this;
}

static PyObject *surface_object_get_bottom(PyObject *self, void *Py_UNUSED(closure))
{
    const surface_object_t *this = (surface_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.bottom);
}

static PyObject *surface_object_get_right(PyObject *self, void *Py_UNUSED(closure))
{
    const surface_object_t *this = (surface_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.right);
}

static PyObject *surface_object_get_top(PyObject *self, void *Py_UNUSED(closure))
{
    const surface_object_t *this = (surface_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.top);
}

static PyObject *surface_object_get_left(PyObject *self, void *Py_UNUSED(closure))
{
    const surface_object_t *this = (surface_object_t *)self;
    return (PyObject *)geo_id_object_from_value(this->value.left);
}

static PyGetSetDef pyvl_surface_getset[] = {
    {.name = "bottom",
     .get = surface_object_get_bottom,
     .set = NULL,
     .doc = "Bottom boundary of the surface.",
     .closure = NULL},
    {.name = "right",
     .get = surface_object_get_right,
     .set = NULL,
     .doc = "Right boundary of the surface.",
     .closure = NULL},
    {.name = "top", .get = surface_object_get_top, .set = NULL, .doc = "Top boundary of the surface.", .closure = NULL},
    {.name = "left",
     .get = surface_object_get_left,
     .set = NULL,
     .doc = "Left boundary of the surface.",
     .closure = NULL},
    {0},
};

INTERPLIB_INTERNAL
PyTypeObject surface_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.Surface",
    .tp_basicsize = sizeof(surface_object_t),
    .tp_itemsize = 0,
    .tp_repr = surface_object_repr,
    .tp_str = surface_object_str,
    .tp_doc = surface_object_type_docstring,
    .tp_richcompare = surface_object_rich_compare,
    .tp_getset = pyvl_surface_getset,
    .tp_new = surface_object_new,
};
