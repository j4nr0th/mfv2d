//
// Created by jan on 27.1.2025.
//

/**
 * Implementation based on Jupyter in:
 *
 * Ciardelli, C., Bozdağ, E., Peter, D., and Van der Lee, S., 2022. SphGLLTools: A toolbox for visualization of large
 * seismic model files based on 3D spectral-element meshes. Computer & Geosciences, v. 159, 105007,
 * doi: https://doi.org/10.1016/j.cageo.2021.105007
 */

#include "integration_rule.h"

#include "../polynomials/gauss_lobatto.h"

#include <numpy/ndarrayobject.h>

static PyObject *integration_rule_1d_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    int order;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", (char *const[2]){"order", NULL}, &order))
        return NULL;
    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be non-negative, got %d.", order);
        return NULL;
    }

    double *nodal_vals, *weight_vals;
    int non_converged = 0;
    Py_BEGIN_ALLOW_THREADS;

    nodal_vals = PyMem_RawMalloc(sizeof(double) * (order + 1));
    weight_vals = PyMem_RawMalloc(sizeof(double) * (order + 1));
    if (nodal_vals && weight_vals)
    {
        non_converged = gauss_lobatto_nodes_weights(order + 1, 1e-15, 10, nodal_vals, weight_vals);
    }
    Py_END_ALLOW_THREADS;
    if (!nodal_vals || !weight_vals)
    {
        PyMem_RawFree(nodal_vals);
        PyMem_RawFree(weight_vals);
        return NULL;
    }
    if (non_converged)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to compute Gauss-Lobatto nodes and weights (%d nodes did not converge).", non_converged);

        PyMem_RawFree(nodal_vals);
        PyMem_RawFree(weight_vals);
        return NULL;
    }

    integration_rule_1d_t *const self = (integration_rule_1d_t *)type->tp_alloc(type, 0);

    if (!self)
        return NULL;

    self->order = order;
    self->nodes = nodal_vals;
    self->weights = weight_vals;

    return (PyObject *)self;
}

static void integration_rule_1d_dealloc(integration_rule_1d_t *self)
{
    PyMem_RawFree(self->nodes);
    PyMem_RawFree(self->weights);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *integration_rule_1d_get_order(const integration_rule_1d_t *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->order);
}

static PyObject *integration_rule_1d_get_nodes(const integration_rule_1d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp n = self->order + 1;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, self->nodes);
    if (out)
    {
        if (PyArray_SetBaseObject(out, (PyObject *)self) < 0)
        {
            Py_DECREF(out);
            return NULL;
        }
        Py_INCREF(self);
    }
    return (PyObject *)out;
}

static PyObject *integration_rule_1d_get_weights(const integration_rule_1d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp n = self->order + 1;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, self->weights);
    if (out)
    {
        if (PyArray_SetBaseObject(out, (PyObject *)self) < 0)
        {
            Py_DECREF(out);
            return NULL;
        }
        Py_INCREF(self);
    }
    return (PyObject *)out;
}

static PyObject *integration_rule_1d_repr(const integration_rule_1d_t *self)
{
    char buffer[128];
    (void)snprintf(buffer, sizeof(buffer), "IntegrationRule1D(order=%u)", self->order);
    return PyUnicode_FromString(buffer);
}

static PyGetSetDef integration_rule_1d_getset[] = {
    {.name = "order",
     .get = (getter)integration_rule_1d_get_order,
     .set = NULL,
     .doc = "int : order of the rule",
     .closure = NULL},
    {.name = "nodes",
     .get = (getter)integration_rule_1d_get_nodes,
     .set = NULL,
     .doc = "array : Position of integration nodes on the reference domain [-1, +1]\n"
            "    where the integrated function should be evaluated.\n",
     .closure = NULL},
    {.name = "weights",
     .get = (getter)integration_rule_1d_get_weights,
     .set = NULL,
     .doc = "array : Weight values by which the values of evaluated function should be\n"
            "    multiplied by.\n",
     .closure = NULL},
    {NULL}};

PyDoc_STRVAR(integration_rule_1d_docstr, "IntegrationRule1D(order: int)\n"
                                         "Type used to contain integration rule information.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "order : int\n"
                                         "    Order of integration rule used. Can not be negative.\n"
                                         "\n");

PyTypeObject integration_rule_1d_type = {
    .tp_new = integration_rule_1d_new,
    .tp_name = "mfv2d._mfv2d.IntegrationRule1D",
    .tp_basicsize = sizeof(integration_rule_1d_t),
    .tp_dealloc = (destructor)integration_rule_1d_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = integration_rule_1d_docstr,
    .tp_getset = integration_rule_1d_getset,
    .tp_repr = (reprfunc)integration_rule_1d_repr,
};
