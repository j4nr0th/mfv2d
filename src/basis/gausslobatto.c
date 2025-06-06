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

#include "gausslobatto.h"

#include <numpy/ndarrayobject.h>

static void legendre_eval_bonnet_two(const unsigned n, const double x, double MFV2D_ARRAY_ARG(out, 2))
{
    // n >= 2
    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = (i - 1);
        const double new = (k1 * v2 - k2 * v1) / (double)(i);
        v1 = v2;
        v2 = new;
    }
    out[0] = v1;
    out[1] = v2;
}

MFV2D_INTERNAL
int gauss_lobatto_nodes_weights(const unsigned n, const double tol, const unsigned max_iter,
                                double MFV2D_ARRAY_ARG(x, restrict n), double MFV2D_ARRAY_ARG(w, restrict n))
{
    int non_converged = 0;
    // n >= 2
    x[0] = -1.0;
    x[n - 1] = +1.0;
    w[n - 1] = w[0] = 2.0 / (double)(n * (n - 1));
    const double kx_1 = (1.0 - 3.0 * (n - 2) / (double)(8.0 * (n - 1) * (n - 1) * (n - 1)));
    const double kx_2 = M_PI / (4.0 * (n - 1) + 1);
    for (unsigned i = 2; i < n; ++i)
    {
        double new_x = kx_1 * cos(kx_2 * (4 * i - 3));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]) / denominator;
            const double d2y = (2 * new_x * dy - (n - 1) * n * leg_poly[1]) / denominator;
            const double d3y = (4 * new_x * d2y - ((n - 1) * n - 2) * dy) / denominator;
            const double dx = 2 * dy * d2y / (2 * d2y * d2y - dy * d3y);
            new_x -= dx;
            error = fabs(dx);
        }
        // this is done like this to catch any NaNs
        non_converged += 1 - (error <= tol);
        x[n - i] = new_x;
        legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
        w[n - i] = 2.0 / (n * (n - 1) * leg_poly[1] * leg_poly[1]);
    }
    return non_converged;
}

MFV2D_INTERNAL
const char compute_gll_docstring[] =
    "compute_gll(order: int, max_iter: int = 10, tol: float = 1e-15) -> tuple[array, array]\n"
    "Compute Gauss-Legendre-Lobatto integration nodes and weights.\n"
    "\n"
    "If you are often re-using these, consider caching them.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "   Order of the scheme. The number of node-weight pairs is one more.\n"
    "max_iter : int, default: 10\n"
    "   Maximum number of iterations used to further refine the values.\n"
    "tol : float, default: 1e-15\n"
    "   Tolerance for stopping the refinement of the nodes.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "   Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.\n"
    "array\n"
    "   Array of integration weights which correspond to the nodes.\n";

MFV2D_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    int order, max_iter = 10;
    double tol = 1e-15;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|id", (char *[4]){"", "max_iter", "tol", NULL}, &order, &max_iter,
                                     &tol))
    {
        return NULL;
    }
    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be positive, but was given as %i.", order);
        return NULL;
    }
    if (max_iter < 0)
    {
        PyErr_Format(PyExc_ValueError, "Number of maximum iterations must be positive, but was given as %i.", max_iter);
        return NULL;
    }
    if (tol < 0)
    {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%g", tol);
        PyErr_Format(PyExc_ValueError, "Tolerance must be positive %s", buffer);
        return NULL;
    }

    const npy_intp array_size = order + 1;
    PyArrayObject *const nodes = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!nodes)
    {
        return NULL;
    }
    PyArrayObject *const weights = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!weights)
    {
        Py_DECREF(nodes);
        return NULL;
    }
    double *const p_x = PyArray_DATA(nodes);
    double *const p_w = PyArray_DATA(weights);
    if (order != 0)
    {
        const int non_converged = gauss_lobatto_nodes_weights(order + 1, tol, max_iter, p_x, p_w);
        if (non_converged != 0)
        {
            PyErr_Format(PyExc_RuntimeWarning,
                         "A total of %i nodes were non-converged. Consider changing"
                         " the tolerance or increase the number of iterations.",
                         non_converged);
        }
    }
    else
    {
        // Corner case
        p_x[0] = 0.0;
        p_w[0] = 2.0;
    }

    return PyTuple_Pack(2, nodes, weights);
}

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

static PyGetSetDef integration_rule_1d_getset[] = {{.name = "order",
                                                    .get = (getter)integration_rule_1d_get_order,
                                                    .set = NULL,
                                                    .doc = "int : order of the rule",
                                                    .closure = NULL},
                                                   {.name = "nodes",
                                                    .get = (getter)integration_rule_1d_get_nodes,
                                                    .set = NULL,
                                                    .doc = "array : Nodes for integration.",
                                                    .closure = NULL},
                                                   {.name = "weights",
                                                    .get = (getter)integration_rule_1d_get_weights,
                                                    .set = NULL,
                                                    .doc = "array : Weights for integration.",
                                                    .closure = NULL},
                                                   {NULL}};

PyDoc_STRVAR(integration_rule_1d_docstr, "IntegrationRule1D(order: int)\n"
                                         "Type used to contain integration rule information.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "order : int\n"
                                         "    Order of integration rule used. Can not be negative.\n"
                                         "\n"
                                         "Attributes\n"
                                         "----------\n"
                                         "nodes : array\n"
                                         "    Position of integration nodes on the reference domain [-1, +1]\n"
                                         "    where the integrated function should be evaluated.\n"
                                         "\n"
                                         "weights : array\n"
                                         "    Weight values by which the values of evaluated function should be\n"
                                         "    multiplied by.\n");

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
