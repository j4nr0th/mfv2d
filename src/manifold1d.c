//
// Created by jan on 18.1.2025.
//

#include "manifold1d.h"

#include "geoidobject.h"

/**********************************************************************************************************************
 *                                                                                                                    *
 *                  Get - set definitions                                                                             *
 *                                                                                                                    *
 **********************************************************************************************************************/

static PyObject *manifold1d_get_dimension(PyObject *Py_UNUSED(self), void *Py_UNUSED(closure))
{
    return PyLong_FromLong(1);
}

static PyObject *manifold1d_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold1d_object_t *this = (manifold1d_object_t *)self;
    return PyLong_FromSize_t(this->n_lines);
}

static PyObject *manifold1d_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold1d_object_t *this = (manifold1d_object_t *)self;
    return PyLong_FromSize_t(this->n_points);
}

static PyGetSetDef manifold1d_get_set[] = {
    {.name = "dimension",
     .get = manifold1d_get_dimension,
     .set = NULL,
     .doc = "int : Dimension of the manifold.",
     .closure = NULL},
    {.name = "n_lines",
     .get = manifold1d_get_n_lines,
     .set = NULL,
     .doc = "int : Number of lines in the manifold.",
     .closure = NULL},
    {.name = "n_points",
     .get = manifold1d_get_n_points,
     .set = NULL,
     .doc = "int : Number of points in the manifold.",
     .closure = NULL},
    {},
};

/**********************************************************************************************************************
 *                                                                                                                    *
 *                  Type method definitions.                                                                          *
 *                                                                                                                    *
 **********************************************************************************************************************/

static PyObject *manifold1d_get_line(PyObject *self, PyObject *arg)
{
    const manifold1d_object_t *const this = (manifold1d_object_t *)self;
    geo_id_t id;
    if (geo_id_from_object(arg, &id) < 0)
    {
        return NULL;
    }
    if (id.index >= this->n_lines)
    {
        PyErr_Format(PyExc_IndexError, "The manifold has %zu lines, but index %u was specified.", this->n_lines,
                     (unsigned)id.index);
        return NULL;
    }
    const line_t *p_line = this->lines + id.index;
    if (id.reverse)
    {
        return (PyObject *)line_from_indices(p_line->end, p_line->begin);
    }
    return (PyObject *)line_from_indices(p_line->begin, p_line->end);
}

static PyObject *manifold1d_find_line(PyObject *self, PyObject *arg)
{
    const manifold1d_object_t *const this = (manifold1d_object_t *)self;
    if (!PyObject_TypeCheck(arg, &line_type_object))
    {
        PyErr_Format(PyExc_TypeError, "Parameter was not a Line, but was instead %R.", Py_TYPE(arg));
        return NULL;
    }
    const line_object_t *line_ob = (line_object_t *)arg;
    const line_t line = line_ob->value;
    if (line.begin.index >= this->n_points || line.end.index >= this->n_points)
    {
        return (PyObject *)geo_id_object_from_value((geo_id_t){.index = GEO_ID_INVALID, .reverse = 0});
    }

    for (size_t line_id = 0; line_id < this->n_lines; ++line_id)
    {
        const line_t *p_line = this->lines + line_id;
        if (geo_id_compare(p_line->begin, line.begin) && geo_id_compare(p_line->end, line.end))
        {
            return (PyObject *)geo_id_object_from_value((geo_id_t){.index = line_id, .reverse = 0});
        }
        if (geo_id_compare(p_line->begin, line.end) && geo_id_compare(p_line->end, line.begin))
        {
            return (PyObject *)geo_id_object_from_value((geo_id_t){.index = line_id, .reverse = 1});
        }
    }
    return (PyObject *)geo_id_object_from_value((geo_id_t){.index = GEO_ID_INVALID, .reverse = 0});
}

static PyObject *manifold1d_line_mesh(PyObject *cls, PyObject *arg)
{
    const size_t n = PyLong_AsSize_t(arg);
    if (PyErr_Occurred())
        return NULL;
    if (n == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Can not create a manifold of a line with no segments.");
        return NULL;
    }
    PyTypeObject *const type = (PyTypeObject *)cls;
    manifold1d_object_t *const this = (manifold1d_object_t *)type->tp_alloc(type, (Py_ssize_t)n);
    if (!this)
        return NULL;
    this->n_points = n + 1;
    this->n_lines = n;
    for (size_t i = 0; i < n; ++i)
    {
        this->lines[i] = (line_t){.begin = {.index = i, .reverse = 0}, .end = {.index = i + 1, .reverse = 0}};
    }
    return (PyObject *)this;
}

static PyObject *manifold1d_compute_dual(PyObject *self, PyObject *Py_UNUSED(args))
{
    const manifold1d_object_t *this = (manifold1d_object_t *)self;

    manifold1d_object_t *dual =
        (manifold1d_object_t *)manifold1d_type_object.tp_alloc(&manifold1d_type_object, (Py_ssize_t)this->n_points);
    if (!dual)
        return NULL;
    dual->n_points = this->n_lines;
    dual->n_lines = this->n_points;

    for (index_t point_idx = 0; point_idx < this->n_points; ++point_idx)
    {
        geo_id_t left = {.index = GEO_ID_INVALID, .reverse = 0};
        geo_id_t right = {.index = GEO_ID_INVALID, .reverse = 0};
        for (index_t line_idx = 0;
             line_idx < this->n_lines && (left.index == GEO_ID_INVALID || right.index == GEO_ID_INVALID); ++line_idx)
        {
            const line_t *p_line = this->lines + line_idx;
            if (p_line->begin.index == point_idx)
            {
                left.index = line_idx;
            }
            if (p_line->end.index == point_idx)
            {
                right.index = line_idx;
            }
        }
        dual->lines[point_idx] = (line_t){.begin = left, .end = right};
    }
    return (PyObject *)dual;
}

static PyMethodDef manifold1d_methods[] = {
    {.ml_name = "get_line",
     .ml_meth = manifold1d_get_line,
     .ml_flags = METH_O,
     .ml_doc = "get_line(index: GeoID | int, /) -> Line\n"
               "Get the line of the specified ID.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "index : GeoID or int\n"
               "    Index of the line to get.\n"
               "Returns\n"
               "-------\n"
               "Line\n"
               "    Line specified by the index.\n"},
    {.ml_name = "find_line",
     .ml_meth = manifold1d_find_line,
     .ml_flags = METH_O,
     .ml_doc = "find_line(line: Line, /) -> GeoID\n"
               "Find the ID of the specified line.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "line : Line\n"
               "    Line object to identify in the manifold.\n"
               "Returns\n"
               "-------\n"
               "GeoID\n"
               "    ID of the line which was to be found. If the line does not exist in the manifold,\n"
               "    then an invalid ID will be returned (will evaluate to ``False``).\n"},
    {.ml_name = "line_mesh",
     .ml_meth = manifold1d_line_mesh,
     .ml_flags = METH_O | METH_CLASS,
     .ml_doc = "line_mesh(segments : int, /) -> Manifold1D\n"
               "Create a new Manifold1D which represents a line.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "segments : int\n"
               "    Number of segments the line is split into. There will be one more point.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Manifold1D\n"
               "    Manifold that represents the topology of the line.\n"},
    {.ml_name = "compute_dual",
     .ml_meth = manifold1d_compute_dual,
     .ml_flags = METH_NOARGS,
     .ml_doc = "compute_dual() -> Manifold1D\n"
               "Compute the dual to the manifold.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Manifold1D\n"
               "    The dual to the manifold.\n"},
    {},
};

/**********************************************************************************************************************
 *                                                                                                                    *
 *                  The type object itself.                                                                           *
 *                                                                                                                    *
 **********************************************************************************************************************/

PyDoc_STRVAR(manifold1d_type_docstr, "One dimensional manifold.");

MFV2D_INTERNAL
PyTypeObject manifold1d_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.Manifold1D",
    .tp_basicsize = sizeof(manifold1d_object_t),
    .tp_itemsize = sizeof(line_t),
    // .tp_repr = ,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = manifold1d_type_docstr,
    .tp_methods = manifold1d_methods,
    .tp_getset = manifold1d_get_set,
    .tp_base = &manifold_type_object,
};
