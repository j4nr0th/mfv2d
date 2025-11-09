#include "legendre.h"

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
        for (unsigned j = 0; j < (unsigned)(order + 1); ++j)
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
