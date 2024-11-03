//
// Created by jan on 21.10.2024.
//

#include "spline1d.h"

#include <numpy/arrayobject.h>
#include <stddef.h>

#include "basis1d.h"
#include "common.h"

static PyObject *spline1d_vectorcall(PyObject *self, PyObject *const *args, size_t nargs, PyObject *Py_UNUSED(kwargs))
{
    if (PyVectorcall_NARGS(nargs) != 1)
    {
        PyErr_Format(PyExc_ValueError, "Spline1D can only be called with a single argument.");
        return NULL;
    }
    const spline1d_t *this = (spline1d_t *)self;
    PyObject *input = *args;
    PyObject *array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY, NULL);
    if (!array)
    {
        return NULL;
    }

    const npy_intp m = PyArray_SIZE((PyArrayObject *)array);
    const unsigned n_nodes = this->n_nodes;
    const double *restrict nodes = this->data + 0;
    const double *restrict coefficients = this->data + this->n_nodes;
    double *restrict const pv = PyArray_DATA((PyArrayObject *)array);

    //  Could make this OpenMP?
    unsigned left_node = 0;
    for (npy_intp i = 0; i < m; ++i)
    {
        const double v = pv[i];
        unsigned span;
        if (v < nodes[left_node])
        {
            // reset if the new node was to the left
            left_node = 0;
        }
        span = this->n_nodes - left_node;
        while (span > 1)
        {
            const unsigned pivot = left_node + span / 2;
            const double g = nodes[pivot];
            if (g < v)
            {
                left_node = pivot;
                span = span - (pivot - left_node);
            }
            else
            {
                span = pivot - left_node;
            }
        }
        // left_node is now less than or equal to v
        unsigned right_node = left_node + 1;

        if (!INTERPLIB_EXPECT_CONDITION(right_node < n_nodes))
        {
            left_node = n_nodes - 2;
            // right_node = n_nodes - 1;
        }

        const double t = v - nodes[left_node];
        const double *restrict coeffs = coefficients + this->n_coefficients * left_node;

        double vv = 1.0;
        double sum = coeffs[0];
        for (unsigned j = 1; j < this->n_coefficients; ++j)
        {
            vv *= t;
            sum += coeffs[j] * vv;
        }
        pv[i] = sum;
    }

    return array;
}

static PyObject *spline1d_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *nodes, *coefficients;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char *[3]){"", "", NULL}, &nodes, &coefficients))
    {
        return NULL;
    }

    PyObject *node_array =
        PyArray_FromAny(nodes, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!node_array)
    {
        return NULL;
    }
    const unsigned n_nodes = PyArray_Size(node_array);
    if (n_nodes < 2)
    {
        Py_DECREF(node_array);
        PyErr_Format(PyExc_ValueError,
                     "There must be at least two nodes to define a spline, but "
                     "%u were given.",
                     n_nodes);
        return NULL;
    }
    PyObject *coeff_array =
        PyArray_FromAny(coefficients, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!coeff_array)
    {
        Py_DECREF(node_array);
        return NULL;
    }
    spline1d_t *this = NULL;
    const npy_intp *restrict coeff_dims = PyArray_DIMS((PyArrayObject *)coeff_array);
    if (coeff_dims[0] + 1 != n_nodes)
    {
        PyErr_Format(PyExc_ValueError,
                     "The number of nodes must be one more than the larges "
                     "dimension of coefficient array,"
                     " instead got %u nodes and %u coefficient groups.",
                     n_nodes, coeff_dims[0]);
        goto end;
    }

    allocfunc alloc = PyType_GetSlot(type, Py_tp_alloc);
    this = (spline1d_t *)alloc(type, (Py_ssize_clean_t)((PyArray_SIZE((PyArrayObject *)node_array) +
                                                         PyArray_SIZE((PyArrayObject *)coeff_array))));
    if (!this)
    {
        goto end;
    }

    this->call_spline = spline1d_vectorcall;
    this->n_nodes = n_nodes;
    this->n_coefficients = coeff_dims[1];
    const double *k = PyArray_DATA((PyArrayObject *)node_array);
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        this->data[i] = k[i];
    }
    const double *coeffs = PyArray_DATA((PyArrayObject *)coeff_array);
    for (unsigned i = 0; i < coeff_dims[0] * coeff_dims[1]; ++i)
    {
        this->data[n_nodes + i] = coeffs[i];
    }

end:
    Py_DECREF(coeff_array);
    Py_DECREF(node_array);
    return (PyObject *)this;
}

static PyObject *spline1d_derivative(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;

    spline1d_t *out;

    if (this->n_coefficients <= 1)
    {
        out = PyObject_NewVar(spline1d_t, &spline1d_type_object, this->n_nodes + 1 * (this->n_nodes - 1));
        if (!out)
        {
            return NULL;
        }
        out->call_spline = spline1d_vectorcall;
        out->n_nodes = this->n_nodes;
        out->n_coefficients = 1;
        for (unsigned i = 0; i < this->n_nodes; ++i)
        {
            out->data[i] = this->data[i];
        }
        for (unsigned i = 0; i < this->n_nodes - 1; ++i)
        {
            out->data[this->n_nodes + i] = 0.0;
        }

        return (PyObject *)out;
    }
    out = PyObject_NewVar(spline1d_t, &spline1d_type_object,
                          this->n_nodes + (this->n_nodes - 1) * (this->n_coefficients - 1));
    if (!out)
    {
        return NULL;
    }
    out->n_nodes = this->n_nodes;
    out->call_spline = spline1d_vectorcall;
    for (unsigned i = 0; i < this->n_nodes; ++i)
    {
        out->data[i] = this->data[i];
    }
    out->n_coefficients = this->n_coefficients - 1;
    for (unsigned i = 0; i < this->n_nodes - 1; ++i)
    {
        for (unsigned j = 1; j < this->n_coefficients; ++j)
        {
            out->data[this->n_nodes + out->n_coefficients * i + j - 1] =
                this->data[this->n_nodes + this->n_coefficients * i + j] * (double)j;
        }
    }
    return (PyObject *)out;
}

static PyObject *spline1d_antiderivative(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;

    spline1d_t *out;

    out = PyObject_NewVar(spline1d_t, &spline1d_type_object,
                          this->n_nodes + (this->n_nodes - 1) * (this->n_coefficients + 1));
    if (!out)
    {
        return NULL;
    }
    out->n_nodes = this->n_nodes;
    out->call_spline = spline1d_vectorcall;
    for (unsigned i = 0; i < this->n_nodes; ++i)
    {
        out->data[i] = this->data[i];
    }
    out->n_coefficients = this->n_coefficients + 1;
    double antiderivative = 0.0;
    for (unsigned i = 0; i < this->n_nodes - 1; ++i)
    {
        out->data[this->n_nodes + out->n_coefficients * i] = antiderivative;
        const double dx = this->data[i + 1] - this->data[i];
        double d = dx;
        for (unsigned j = 0; j < this->n_coefficients; ++j)
        {
            const double k = this->data[this->n_nodes + this->n_coefficients * i + j] / (double)(j + 1);
            out->data[this->n_nodes + out->n_coefficients * i + j + 1] = k;
            antiderivative += k * d;
            d *= dx;
        }
    }
    return (PyObject *)out;
}

static PyObject *spline1d_get_coefficients(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;
    npy_intp ndims[2] = {this->n_nodes - 1, this->n_coefficients};
    PyObject *array = PyArray_SimpleNewFromData(2, ndims, NPY_DOUBLE, (void *)(this->data + this->n_nodes));
    if (!array)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)array, self))
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    return array;
}

static int spline1d_set_coefficients(PyObject *self, PyObject *v, void *Py_UNUSED(closure))
{
    spline1d_t *this = (spline1d_t *)self;
    PyObject *array = PyArray_FromAny(v, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return -1;
    }

    const npy_intp *ndims = PyArray_DIMS((PyArrayObject *)array);
    if (ndims[0] != this->n_nodes - 1 || ndims[1] != this->n_coefficients)
    {
        Py_DECREF(array);
        (void)PyErr_Format(PyExc_ValueError,
                           "Spline has %u groups of %u coefficients, but %u "
                           "groups of %u were given instead.",
                           ndims[0], ndims[1]);
        return -1;
    }

    const double *k = PyArray_DATA((PyArrayObject *)array);
    for (unsigned i = 0; i < (this->n_nodes - 1) * this->n_coefficients; ++i)
    {
        this->data[this->n_nodes + i] = k[i];
    }

    Py_DECREF(array);
    return 0;
}

static PyObject *spline1d_get_nodes(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;
    npy_intp n = this->n_nodes;
    PyObject *array = PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, (void *)this->data);
    if (!array)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)array, self))
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    return array;
}

static PyGetSetDef spline1d_getset[] = {
    {.name = "coefficients",
     .get = spline1d_get_coefficients,
     .set = spline1d_set_coefficients,
     .doc = "Coefficients of the polynomials",
     .closure = NULL},
    {.name = "nodes", .get = spline1d_get_nodes, .set = NULL, .doc = "Nodes of the spline", .closure = NULL},
    {.name = "derivative",
     .get = spline1d_derivative,
     .set = NULL,
     .doc = "Return derivative of the spline.",
     .closure = NULL},
    {.name = "antiderivative",
     .get = spline1d_antiderivative,
     .set = NULL,
     .doc = "Return antiderivative of the spline.",
     .closure = NULL},
    {NULL, NULL, NULL, NULL, NULL} // sentinel
};

PyDoc_STRVAR(spline1d_docstring, "Spline1D(nodes: Sequence, coefficients: Sequence[Sequence], /)\n"
                                 "\n"
                                 "Piecewise polynomial function, defined between nodes.\n"
                                 "\n"
                                 "Parameters\n"
                                 "----------\n"
                                 "nodes : (N + 1,) Sequence\n"
                                 "    Sequence of ``N + 1`` values, which mark where a different set of\n"
                                 "    coefficients is to be used.\n"
                                 "coefficients : (N, M) Sequence\n"
                                 "    Sequence of ``M`` coefficients for each of the ``N`` elements.\n");

INTERPLIB_INTERNAL
PyTypeObject spline1d_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_interp.Spline1D",
    .tp_basicsize = sizeof(spline1d_t),
    .tp_itemsize = sizeof(double),
    .tp_vectorcall_offset = offsetof(spline1d_t, call_spline),
    // .tp_repr = ,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_doc = spline1d_docstring,
    .tp_getset = spline1d_getset,
    .tp_base = &basis1d_type_object,
    .tp_new = spline1d_new,
    // .tp_vectorcall = ,
    .tp_call = PyVectorcall_Call,
};

static PyObject *spline1di_vectorcall(PyObject *self, PyObject *const *args, size_t nargs, PyObject *Py_UNUSED(kwargs))
{
    if (PyVectorcall_NARGS(nargs) != 1)
    {
        PyErr_Format(PyExc_ValueError, "Spline1Di can only be called using a single argument.");
        return NULL;
    }
    const spline1d_t *this = (spline1d_t *)self;
    PyObject *input = *args;
    PyObject *array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY, NULL);
    if (!array)
    {
        return NULL;
    }

    const npy_intp m = PyArray_SIZE((PyArrayObject *)array);
    const unsigned n_elm = this->n_nodes - 1;
    const double *restrict coefficients = this->data;
    double *restrict const pv = PyArray_DATA((PyArrayObject *)array);

    //  Could make this OpenMP?
    for (npy_intp i = 0; i < m; ++i)
    {
        const double v = pv[i];
        double integral;
        double t = modf(v, &integral);
        int idx = (int)integral;
        //  Adjust if too small
        if (idx < 0)
        {
            t += integral;
            idx = 0;
        }
        else if (idx >= n_elm)
        {
            idx = (int)(n_elm - 1);
            t += integral - (double)idx;
        }

        const double *restrict coeffs = coefficients + this->n_coefficients * idx;

        double vv = 1.0;
        double sum = coeffs[0];
        for (unsigned j = 1; j < this->n_coefficients; ++j)
        {
            vv *= t;
            sum += coeffs[j] * vv;
        }
        pv[i] = sum;
    }

    return array;
}

static PyObject *spline1di_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *coefficients;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char *[2]){"", NULL}, &coefficients))
    {
        return NULL;
    }

    PyObject *coeff_array =
        PyArray_FromAny(coefficients, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!coeff_array)
    {
        return NULL;
    }

    const npy_intp *restrict coeff_dims = PyArray_DIMS((PyArrayObject *)coeff_array);
    const unsigned n_nodes = coeff_dims[0] + 1;

    allocfunc alloc = PyType_GetSlot(type, Py_tp_alloc);
    spline1d_t *const this = (spline1d_t *)alloc(type, (Py_ssize_clean_t)(PyArray_SIZE((PyArrayObject *)coeff_array)));
    if (!this)
    {
        goto end;
    }

    this->call_spline = spline1di_vectorcall;
    this->n_nodes = n_nodes;
    this->n_coefficients = coeff_dims[1];
    const double *coeffs = PyArray_DATA((PyArrayObject *)coeff_array);
    for (unsigned i = 0; i < coeff_dims[0] * coeff_dims[1]; ++i)
    {
        this->data[i] = coeffs[i];
    }

end:
    Py_DECREF(coeff_array);
    return (PyObject *)this;
}

static PyObject *spline1di_derivative(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;

    spline1d_t *out;

    if (this->n_coefficients <= 1)
    {
        out = PyObject_NewVar(spline1d_t, &spline1di_type_object, this->n_nodes - 1);
        if (!out)
        {
            return NULL;
        }
        out->call_spline = spline1di_vectorcall;
        out->n_nodes = this->n_nodes;
        out->n_coefficients = 1;
        for (unsigned i = 0; i < this->n_nodes - 1; ++i)
        {
            out->data[i] = 0.0;
        }
        return (PyObject *)out;
    }
    out = PyObject_NewVar(spline1d_t, &spline1di_type_object, (this->n_nodes - 1) * (this->n_coefficients - 1));
    if (!out)
    {
        return NULL;
    }
    out->n_nodes = this->n_nodes;
    out->call_spline = spline1di_vectorcall;
    out->n_coefficients = this->n_coefficients - 1;
    for (unsigned i = 0; i < this->n_nodes - 1; ++i)
    {
        for (unsigned j = 1; j < this->n_coefficients; ++j)
        {
            out->data[out->n_coefficients * i + j - 1] = this->data[this->n_coefficients * i + j] * (double)j;
        }
    }
    return (PyObject *)out;
}

static PyObject *spline1di_antiderivative(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;

    spline1d_t *out;

    out = PyObject_NewVar(spline1d_t, &spline1d_type_object, (this->n_nodes - 1) * (this->n_coefficients + 1));
    if (!out)
    {
        return NULL;
    }
    out->call_spline = spline1di_vectorcall;
    out->n_nodes = this->n_nodes;
    out->n_coefficients = this->n_coefficients + 1;
    double prev_anti_derivative = 0.0;
    for (unsigned i = 0; i < this->n_nodes - 1; ++i)
    {
        out->data[out->n_coefficients * i] = prev_anti_derivative;

        for (unsigned j = 0; j < this->n_coefficients; ++j)
        {
            const double coef = this->data[this->n_coefficients * i + j] / (double)(j + 1);
            out->data[out->n_coefficients * i + j + 1] = coef;
            prev_anti_derivative += coef;
        }
    }
    return (PyObject *)out;
}

static PyObject *spline1di_get_coefficients(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;
    npy_intp ndims[2] = {this->n_nodes - 1, this->n_coefficients};
    PyObject *array = PyArray_SimpleNewFromData(2, ndims, NPY_DOUBLE, (void *)(this->data));
    if (!array)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)array, self))
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    return array;
}

static int spline1di_set_coefficients(PyObject *self, PyObject *v, void *Py_UNUSED(closure))
{
    spline1d_t *this = (spline1d_t *)self;
    PyObject *array = PyArray_FromAny(v, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return -1;
    }

    const npy_intp *ndims = PyArray_DIMS((PyArrayObject *)array);
    if (ndims[0] != this->n_nodes - 1 || ndims[1] != this->n_coefficients)
    {
        Py_DECREF(array);
        (void)PyErr_Format(PyExc_ValueError,
                           "Spline has %u groups of %u coefficients, but %u "
                           "groups of %u were given instead.",
                           ndims[0], ndims[1]);
        return -1;
    }

    const double *k = PyArray_DATA((PyArrayObject *)array);
    for (unsigned i = 0; i < (this->n_nodes - 1) * this->n_coefficients; ++i)
    {
        this->data[i] = k[i];
    }

    Py_DECREF(array);
    return 0;
}

static PyObject *spline1di_get_nodes(PyObject *self, void *Py_UNUSED(closure))
{
    const spline1d_t *this = (spline1d_t *)self;
    const npy_intp n = this->n_nodes;
    PyObject *array = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    if (!array)
    {
        return NULL;
    }
    double *restrict const p = PyArray_DATA((PyArrayObject *)array);
    for (unsigned i = 0; i < this->n_nodes; ++i)
    {
        p[i] = (double)i;
    }
    return array;
}

static PyGetSetDef spline1di_getset[] = {
    {.name = "coefficients",
     .get = spline1di_get_coefficients,
     .set = spline1di_set_coefficients,
     .doc = "Coefficients of the polynomials",
     .closure = NULL},
    {.name = "nodes", .get = spline1di_get_nodes, .set = NULL, .doc = "Nodes of the spline", .closure = NULL},
    {.name = "derivative",
     .get = spline1di_derivative,
     .set = NULL,
     .doc = "Return derivative of the spline.",
     .closure = NULL},
    {.name = "antiderivative",
     .get = spline1di_antiderivative,
     .set = NULL,
     .doc = "Return antiderivative of the spline.",
     .closure = NULL},
    {NULL, NULL, NULL, NULL, NULL} // sentinel
};

PyDoc_STRVAR(spline1di_docstring, "Spline1Di(coefficients: Sequence[Sequence], /)\n"
                                  "\n"
                                  "Piecewise polynomial function, defined between integer nodes.\n"
                                  "\n"
                                  "Parameters\n"
                                  "----------\n"
                                  "coefficients : (N, M) Sequence\n"
                                  "    Sequence of ``M`` coefficients for each of the ``N`` elements.\n");

INTERPLIB_INTERNAL
PyTypeObject spline1di_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_interp.Spline1Di",
    .tp_basicsize = sizeof(spline1d_t),
    .tp_itemsize = sizeof(double),
    .tp_vectorcall_offset = offsetof(spline1d_t, call_spline),
    // .tp_repr = ,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_doc = spline1di_docstring,
    .tp_getset = spline1di_getset,
    .tp_base = &spline1d_type_object,
    .tp_new = spline1di_new,
    // .tp_vectorcall = ,
    .tp_call = PyVectorcall_Call,
};
