//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#include "../common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Internal headers
#include "basis1d.h"
#include "bernstein.h"
#include "cubic_splines.h"
#include "gausslobatto.h"
#include "lagrange.h"
#include "polynomial1d.h"
#include "spline1d.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyObject *interp_lagrange(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyObject *arg1, *arg2;
    PyArrayObject *out = NULL;
    if (!PyArg_ParseTuple(args, "OO|O!", &arg1, &arg2, &PyArray_Type, &out))
    {
        return NULL;
    }
    PyArrayObject *const roots = (PyArrayObject *)PyArray_FromAny(arg1, PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                                                  NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!roots)
    {
        return NULL;
    }
    PyArrayObject *const positions = (PyArrayObject *)PyArray_FromAny(arg2, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!positions)
    {
        Py_DECREF(roots);
        return NULL;
    }
    const npy_intp n_roots = PyArray_SIZE(roots);
    const npy_intp n_dim = PyArray_NDIM(positions);
    const npy_intp *const p_dim = PyArray_DIMS(positions);
    if (out)
    {
        if (PyArray_TYPE(out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Output array must have the correct type (numpy.double/numpy.float64).");
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        int matches = n_dim + 1 == PyArray_NDIM(out) && PyArray_DIM(out, (int)n_dim) == n_roots;
        for (int n = 0; matches && n < n_dim; ++n)
        {
            matches = p_dim[n] == PyArray_DIM(out, n);
        }
        if (!matches)
        {
            PyErr_SetString(PyExc_ValueError, "Output must have same shape as input array, except for one more"
                                              " dimension, which must be the same length as the array of Lagrange"
                                              " nodes.");
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {
        npy_intp *const buffer = PyMem_Malloc(sizeof *buffer * (n_dim + 1));
        if (!buffer)
        {
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        for (unsigned i = 0; i < n_dim; ++i)
        {
            buffer[i] = p_dim[i];
        }
        buffer[n_dim] = n_roots;
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim + 1, buffer, NPY_DOUBLE);
        PyMem_Free(buffer);
        if (!out)
        {
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
    }

    const npy_intp n_pos = PyArray_SIZE(positions);

    double *work = PyMem_Malloc(n_roots * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        Py_DECREF(roots);
        Py_DECREF(positions);
        return NULL;
    }

    const double *const p_x = (double *)PyArray_DATA(positions);
    const double *restrict nodes = PyArray_DATA(roots);
    double *const p_out = (double *)PyArray_DATA(out);

    lagrange_polynomial_values(n_pos, p_x, n_roots, nodes, p_out, work);

    PyMem_Free(work);
    return (PyObject *)out;
}

static PyObject *interp_dlagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(2 * n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res =
        lagrange_polynomial_first_derivative(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work, work + n_nodes);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

static PyObject *interp_d2lagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(2 * n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res =
        lagrange_polynomial_second_derivative(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work, work + n_nodes);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

static PyObject *interp_hermite_coefficients(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyObject *bc1;
    PyObject *bc2;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x, &PyTuple_Type, &bc1, &PyTuple_Type, &bc2))
    {
        return NULL;
    }
    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");

    cubic_spline_bc bc_left, bc_right;
    if (!PyArg_ParseTuple(bc1, "ddd", &bc_left.k1, &bc_left.k2, &bc_left.v))
    {
        return NULL;
    }
    if (!PyArg_ParseTuple(bc2, "ddd", &bc_right.k1, &bc_right.k2, &bc_right.v))
    {
        return NULL;
    }

    npy_intp n_pts = PyArray_SIZE(x);

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, &n_pts, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(n_pts * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res = interp_cubic_spline_init(n_pts, p_x, p_out, work, bc_left, bc_right);
    PyMem_Free(work);
    if ASSERT (interp_res == INTERP_SUCCESS, "Interpolation failed")
    {
        Py_DECREF(out);
        PyErr_Format(PyExc_RuntimeError, "Could not compute Hermite coefficients, error code %s (%s)",
                     interp_error_str(interp_res), interp_error_msg(interp_res));
        return NULL;
    }

    return (PyObject *)PyArray_Return(out);
}

PyDoc_STRVAR(interp_lagrange_doc,
             "lagrange1d(roots: array_like, x: array_like, out: array|None = None, /) -> array\n"
             "Evaluate Lagrange polynomials.\n"
             "\n"
             "This function efficiently evaluates Lagrange basis polynomials, defined by\n"
             "\n"
             ".. math::\n"
             "\n"
             "   \\mathcal{L}^n_i (x) = \\prod\\limits_{j=1, j \\neq i}^{n} \\frac{x - x_j}{x_i - x_j},\n"
             "\n"
             "where the ``roots`` specifies the zeros of the Polynomials :math:`\\{x_1, \\dots, x_n\\}`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "roots : array_like\n"
             "   Roots of Lagrange polynomials.\n"
             "x : array_like\n"
             "   Points where the polynomials should be evaluated.\n"
             "out : array, optional\n"
             "   Array where the results should be written to. If not given, a new one will be\n"
             "   created and returned. It should have the same shape as ``x``, but with an extra\n"
             "   dimension added, the length of which is ``len(roots)``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "   Array of Lagrange polynomial values at positions specified by ``x``.\n"

             "Examples\n"
             "--------\n"
             "This example here shows the most basic use of the function to evaluate Lagrange\n"
             "polynomials. First, let us define the roots.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> import numpy as np\n"
             "    >>>\n"
             "    >>> order = 7\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "\n"
             "Next, we can evaluate the polynomials at positions. Here the interval between the\n"
             "roots is chosen.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from interplib import lagrange1d\n"
             "    >>>\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)\n"
             "    >>> yvals = lagrange1d(roots, xpos)\n"
             "\n"
             "Note that if we were to give an output array to write to, it would also be the\n"
             "return value of the function (as in no copy is made).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> yvals is lagrange1d(roots, xpos, yvals)\n"
             "    True\n"
             "\n"
             "Now we can plot these polynomials.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from matplotlib import pyplot as plt\n"
             "    >>>\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> plt.legend()\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n"
             "\n"

             "Accuracy is retained even at very high polynomial order. The following\n"
             "snippet shows that even at absurdly high order of 51, the results still\n"
             "have high accuracy and don't suffer from rounding errors. It also performs\n"
             "well (in this case, the 52 polynomials are each evaluated at 1025 points).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from time import perf_counter\n"
             "    >>> order = 51\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)\n"
             "    >>> t0 = perf_counter()\n"
             "    >>> yvals = lagrange1d(roots, xpos)\n"
             "    >>> t1 = perf_counter()\n"
             "    >>> print(f\"Calculations took {t1 - t0: e} seconds.\")\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> # plt.legend() # No, this is too long\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n");
PyDoc_STRVAR(interp_dlagrange_doc, "dlagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");
PyDoc_STRVAR(interp_d2lagrange_doc, "d2lagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");
PyDoc_STRVAR(interp_hermite_doc, "hermite(x: np.ndarray, bc1: tuple[float, float, float], bc2: "
                                 "tuple[float, float, float]) -> np.ndarray");

static PyObject *test_method(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;

    return PyUnicode_FromString("Test successful!\n");
}

static PyMethodDef module_methods[] = {
    {"test", test_method, METH_NOARGS, "Test method that only returns a string."},
    {"lagrange1d", interp_lagrange, METH_VARARGS, interp_lagrange_doc},
    {"dlagrange1d", interp_dlagrange, METH_VARARGS, interp_dlagrange_doc},
    {"d2lagrange1d", interp_d2lagrange, METH_VARARGS, interp_d2lagrange_doc},
    {"hermite", interp_hermite_coefficients, METH_VARARGS, interp_hermite_doc},
    {"bernstein1d", (PyCFunction)bernstein_interpolation_matrix, METH_FASTCALL, bernstein_interpolation_matrix_doc},
    {"bernstein_coefficients", (PyCFunction)bernstein_coefficients, METH_O, bernstein_coefficients_doc},
    {.ml_name = "compute_gll",
     .ml_meth = (void *)compute_gauss_lobatto_nodes,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = compute_gll_docstring},
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "interplib._interp",
                             .m_doc = "Internal C-extension implementing interpolation functions",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__interp(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        goto failed;
    }

    if (PyType_Ready(&basis1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&polynomial1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&spline1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&spline1di_type_object) < 0)
    {
        goto failed;
    }

    PyObject *mod = PyModule_Create(&module);
    if (!mod)
    {
        goto failed;
    }

    int res = PyModule_AddObjectRef(mod, "Basis1D", (PyObject *)&basis1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Polynomial1D", (PyObject *)&polynomial1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Spline1D", (PyObject *)&spline1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Spline1Di", (PyObject *)&spline1di_type_object);
    if (res)
    {
        goto failed;
    }

    return mod;

failed:
    // Py_XDECREF(spline1d_type);
    // Py_XDECREF(poly_basis_type);
    // Py_XDECREF(basis1d_type);
    Py_XDECREF(mod);
    return NULL;
}
