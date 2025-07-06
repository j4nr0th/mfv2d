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

/**
 * Evaluates the Legendre polynomial of degree n and its derivative using Bonnet's recursion formula.
 * Stores the results in the provided output array.
 *
 * @param n The degree of the Legendre polynomial. Must be greater than or equal to 2.
 * @param x The point at which the Legendre polynomial is evaluated.
 * @param out A two-element array where the result is stored.
 *            out[0] receives the value of the Legendre polynomial of degree n-1.
 *            out[1] receives the value of the Legendre polynomial of degree n.
 */
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
    "compute_gll(order: int, /, max_iter: int = 10, tol: float = 1e-15) -> tuple[array, array]\n"
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
    "   Array of integration weights which correspond to the nodes.\n"
    "\n"
    "Examples\n"
    "--------\n"
    "Gauss-Legendre-Lobatto nodes computed using this function, along with\n"
    "the weights.\n"
    "\n"
    ".. jupyter-execute::\n"
    "\n"
    "    >>> import numpy as np\n"
    "    >>> from mfv2d._mfv2d import compute_gll\n"
    "    >>> from matplotlib import pyplot as plt\n"
    "    >>>\n"
    "    >>> n = 5\n"
    "    >>> nodes, weights = compute_gll(n)\n"
    "    >>>\n"
    "    >>> # Plot these\n"
    "    >>> plt.figure()\n"
    "    >>> plt.scatter(nodes, weights)\n"
    "    >>> plt.xlabel(\"$\\\\xi$\")\n"
    "    >>> plt.ylabel(\"$w$\")\n"
    "    >>> plt.grid()\n"
    "    >>> plt.show()\n"
    "\n"
    "Since these are computed in an iterative way, giving a tolerance\n"
    "which is too strict or not allowing for sufficient iterations\n"
    "might cause an exception to be raised to do failiure to converge.\n"
    "\n";

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

/**
 * Evaluates the Legendre polynomial of degree n and its derivative using Bonnet's recursion formula.
 * Stores the results in the provided output array.
 *
 * @param n The degree of the Legendre polynomial.
 * @param x The point at which the Legendre polynomial is evaluated.
 * @param out An array where the result is stored.
 *            out[i] receives the value of the Legendre polynomial of degree i.
 */
static void legendre_eval_bonnet_all(const unsigned n, const double x, double MFV2D_ARRAY_ARG(out, n + 1))
{
    // Always there
    out[0] = 1.0;

    if (n > 0)
    {
        out[1] = x;
    }

    if (n > 1)
    {
        double v1 = 1.0;
        double v2 = x;
        for (unsigned i = 2; i < n + 1; ++i)
        {
            const double k1 = (2 * i - 1) * x;
            const double k2 = (i - 1);
            const double new = (k1 * v2 - k2 * v1) / (double)(i);
            out[i] = new;
            v1 = v2;
            v2 = new;
        }
    }
}

MFV2D_INTERNAL
PyObject *compute_legendre_polynomials(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    int order;
    PyObject *positions;
    PyArrayObject *out = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO|O!", (char *[4]){"order", "positions", "out", NULL}, &order,
                                     &positions, &PyArray_Type, &out))
    {
        return NULL;
    }

    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order can not be negative, but was given as %i.", order);
        return NULL;
    }

    PyArrayObject *const positions_array =
        (PyArrayObject *)PyArray_FROMANY(positions, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!positions_array)
        return NULL;

    const unsigned ndim = PyArray_NDIM(positions_array);
    const npy_intp *const shape = PyArray_DIMS(positions_array);

    enum
    {
        MAXIMUM_DIMENSIONS = 32
    };
    npy_intp array_size[MAXIMUM_DIMENSIONS + 1];
    if (ndim > MAXIMUM_DIMENSIONS)
    {
        PyErr_Format(PyExc_ValueError, "Too many dimensions for positions array, got %i, but can only support %i.",
                     ndim, MAXIMUM_DIMENSIONS);
        Py_DECREF(positions_array);
        return NULL;
    }
    memcpy(array_size + 1, shape, sizeof(npy_intp) * ndim);
    array_size[0] = order + 1;

    if (out == NULL)
    {
        out = (PyArrayObject *)PyArray_SimpleNew(ndim + 1, array_size, NPY_DOUBLE);
        if (!out)
        {
            Py_DECREF(positions_array);
            return NULL;
        }
    }
    else
    {
        if (check_input_array(out, ndim + 1, array_size, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "output") < 0)
        {
            Py_DECREF(positions_array);
            return NULL;
        }
        Py_INCREF(out);
    }
    double *const p_out = PyArray_DATA(out);
    double *const p_positions = PyArray_DATA(positions_array);
    double *const buffer = (double *)PyMem_RawMalloc(sizeof(double) * (order + 1));
    if (!buffer)
    {
        Py_DECREF(positions_array);
        Py_DECREF(out);
        return NULL;
    }

    // Calculations are done here, so no need to hold GIL.
    Py_BEGIN_ALLOW_THREADS;

    unsigned n_positions = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        n_positions *= shape[idim];
    }

    for (unsigned i = 0; i < n_positions; ++i)
    {
        const double x = p_positions[i];
        legendre_eval_bonnet_all(order, x, buffer);
        for (unsigned j = 0; j < (order + 1); ++j)
        {
            p_out[j * n_positions + i] = buffer[j];
        }
    }

    PyMem_RawFree(buffer);
    Py_END_ALLOW_THREADS;

    return (PyObject *)out;
}

MFV2D_INTERNAL
const char compute_legendre_polynomials_docstring[] =
    "compute_lagrange_polynomials(order: int, positions: array_like, out: array|None = None) -> array\n"
    "Compute Legendre polynomials at given nodes.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "    Order of the scheme. The number of node-weight pairs is one more.\n"
    "\n"
    "positions : array_like\n"
    "    Positions where the polynomials should be evaluated at.\n"
    "\n"
    "out : array, optional\n"
    "    Output array to write to. If not specified, then a new array is allocated.\n"
    "    Must have the exact correct shape (see return value) and data type\n"
    "    (double/float64).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array with the same shape as ``positions`` parameter, except with an\n"
    "    additional first dimension, which determines which Legendre polynomial\n"
    "    it is.\n"
    "\n"
    "Examples\n"
    "--------\n"
    "To quickly illustrate how this function can be used to work with Legendre polynomials,\n"
    "some simple examples are shown.\n"
    "\n"
    "First things first, the function can be called for any order of polynomials, with\n"
    "about any shape of input array (though if you put too many dimensions you will get an\n"
    "exception). Also, you can supply an optional output parameter, such that an output\n"
    "array need not be newly allocated.\n"
    "\n"
    ".. jupyter-execute::\n"
    "\n"
    "    >>> import numpy as np\n"
    "    >>> from mfv2d._mfv2d import compute_legendre\n"
    "    >>>\n"
    "    >>> n = 5\n"
    "    >>> positions = np.linspace(-1, +1, 101)\n"
    "    >>> vals = compute_legendre(n, positions)\n"
    "    >>> assert vals is compute_legendre(n, positions, vals)\n"
    "\n"
    "The output array will always have the same shape as the input array, with the only\n"
    "difference being that a new axis is added for the first dimension, which can be\n"
    "indexed to distinguish between the different Legendre polynomials.\n"
    "\n"
    ".. jupyter-execute::\n"
    "\n"
    "    >>> from matplotlib import pyplot as plt\n"
    "    >>>\n"
    "    >>> fig, ax = plt.subplots(1, 1)\n"
    "    >>>\n"
    "    >>> for i in range(n + 1):\n"
    "    >>>     ax.plot(positions, vals[i, ...], label=f\"$y = \\\\mathcal{{L}}_{{{i:d}}}$\")\n"
    "    >>>\n"
    "    >>> ax.set(xlabel=\"$x$\", ylabel=\"$y$\")\n"
    "    >>> ax.grid()\n"
    "    >>> ax.legend()\n"
    "    >>>\n"
    "    >>> fig.tight_layout()\n"
    "    >>> plt.show()\n"
    "\n"
    "Lastly, these polynomials are all orthogonal under the :math:`L^2` norm. This can\n"
    "be shown numerically as well.\n"
    "\n"
    ".. jupyter-execute::\n"
    "\n"
    "    >>> from mfv2d._mfv2d import IntegrationRule1D\n"
    "    >>>\n"
    "    >>> rule = IntegrationRule1D(n + 1)\n"
    "    >>>\n"
    "    >>> vals = compute_legendre(n, rule.nodes)\n"
    "    >>>\n"
    "    >>> for i1 in range(n + 1):\n"
    "    >>>     p1 = vals[i1, ...]\n"
    "    >>>     for i2 in range(n + 1):\n"
    "    >>>         p2 = vals[i2, ...]\n"
    "    >>>\n"
    "    >>>         integral = np.sum(p1 * p2 * rule.weights)\n"
    "    >>>\n"
    "    >>>         if i1 != i2:\n"
    "    >>>             assert abs(integral) < 1e-16\n"
    "\n";

MFV2D_INTERNAL
const char legendre_l2_to_h1_coefficients_docstring[] =
    "legendre_l2_to_h1_coefficients(c: array_like, /) -> array\n"
    "Convert Legendre polynomial coefficients to H1 coefficients.\n"
    "\n"
    "The :math:`H^1` coefficients are based on being expansion coefficients of hierarchical\n"
    "basis which are orthogonal in the :math:`H^1` norm instead of in the :math:`L^2` norm,\n"
    "which holds for Legendre polynomials instead.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "c : array_like\n"
    "    Coefficients of the Legendre polynomials.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Coefficients of integrated Legendre polynomial basis.\n";

MFV2D_INTERNAL
PyObject *legendre_l2_to_h1_coefficients(PyObject *Py_UNUSED(self), PyObject *coefficients)
{
    PyArrayObject *const coeffs_in =
        (PyArrayObject *)PyArray_FROMANY(coefficients, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!coeffs_in)
        return NULL;

    if (PyArray_NDIM(coeffs_in) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Input must be a 1D array.");
        return NULL;
    }

    const npy_intp order = PyArray_SIZE(coeffs_in) - 1;
    if (order < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Order must be >= 0.");
        return NULL;
    }

    npy_intp n_coeffs = order + 1;
    double *const coeffs = (double *)PyArray_DATA(coeffs_in);

    // Compute end and beginning
    double end = 0.0, beginning = 0.0;
    for (npy_intp i = 0; i < n_coeffs; ++i)
    {
        end += coeffs[i];
        beginning += coeffs[i] * ((i & 1) ? -1.0 : 1.0);
    }

    // Output array
    const npy_intp dims[1] = {n_coeffs};
    PyArrayObject *const out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (!out)
    {
        Py_DECREF(coeffs_in);
        return NULL;
    }
    double *out_data = (double *)PyArray_DATA(out);

    // First two coefficients
    out_data[0] = (end + beginning) / 2.0;
    if (order > 0)
        out_data[1] = (end - beginning) / 2.0;

    // Main loop
    for (npy_intp n = 2; n < n_coeffs; ++n)
    {
        double carry = 0.0;
        const npy_intp m = n / 2;
        for (npy_intp j = 1; j <= m; ++j)
        {
            const npy_intp idx = n - 2 * j;
            // norms[idx] = 2.0 / (double)(2 * idx + 1);
            carry += 2 * (double)(2 * n - 4 * j + 1) * coeffs[idx] / (double)(2 * idx + 1);
        }
        double k;
        if (n & 1) // Odd
        {
            k = (end - beginning) - carry;
        }
        else // Even
        {
            k = (end + beginning) - carry;
        }

        // const double scale = (double)(2 * (n - 1) + 1) / 2.0;
        const double scale = (double)(n - 1) - 0.5; // avoids division
        out_data[n] = scale * k;
    }

    Py_DECREF(coeffs_in);

    return (PyObject *)out;
}
