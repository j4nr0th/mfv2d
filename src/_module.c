//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _mfv2d
#include "common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Topology
#include "topology/geoidobject.h"
#include "topology/lineobject.h"
#include "topology/manifold.h"
#include "topology/manifold2d.h"
#include "topology/surfaceobject.h"

// Evaluation
#include "eval/element_cache.h"
#include "eval/element_system.h"
#include "eval/incidence.h"

// Solver
#include "solve/givens.h"
#include "solve/lil_matrix.h"
#include "solve/svector.h"

// Basis

#include "basis/gausslobatto.h"
#include "basis/lagrange.h"
#include "eval/basis.h"
#include "eval/bytecode.h"
#include "eval/fem_space.h"

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

static PyObject *interp_dlagrange(PyObject *Py_UNUSED(module), PyObject *args)
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

    size_t size_work = sizeof(double) * 2 * n_roots;
    size_t size_buffet = sizeof(npy_intp) * (n_dim + 1);
    void *const mem_buffer = PyMem_Malloc(size_buffet > size_work ? size_buffet : size_work);
    if (!mem_buffer)
    {
        Py_DECREF(roots);
        Py_DECREF(positions);
        return NULL;
    }
    double *const work1 = (double *)mem_buffer + 0;
    double *const work2 = (double *)mem_buffer + n_roots;
    npy_intp *const dim_buffer = (npy_intp *)mem_buffer;
    if (out)
    {
        if (PyArray_TYPE(out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Output array must have the correct type (numpy.double/numpy.float64).");
            PyMem_Free(mem_buffer);
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
            PyMem_Free(mem_buffer);
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {

        for (unsigned i = 0; i < n_dim; ++i)
        {
            dim_buffer[i] = p_dim[i];
        }
        dim_buffer[n_dim] = n_roots;
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim + 1, dim_buffer, NPY_DOUBLE);
        if (!out)
        {
            PyMem_Free(mem_buffer);
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
    }

    const npy_intp n_pos = PyArray_SIZE(positions);

    const double *const p_x = (double *)PyArray_DATA(positions);
    const double *restrict nodes = PyArray_DATA(roots);
    double *const p_out = (double *)PyArray_DATA(out);

    lagrange_polynomial_first_derivative(n_pos, p_x, n_roots, nodes, p_out, work1, work2);
    PyMem_Free(mem_buffer);

    return (PyObject *)out;
}

PyDoc_STRVAR(interp_lagrange_doc,
             "lagrange1d(roots: array_like, x: array_like, out: array|None = None, /) -> array\n"
             "Evaluate Lagrange polynomials.\n"
             "\n"
             "This function efficiently evaluates Lagrange basis polynomials, defined by\n"
             "\n"
             ".. math::\n"
             "\n"
             "   \\mathcal{L}^n_i (x) = \\prod\\limits_{j=0, j \\neq i}^{n} \\frac{x - x_j}{x_i - x_j},\n"
             "\n"
             "where the ``roots`` specifies the zeros of the Polynomials :math:`\\{x_0, \\dots, x_n\\}`.\n"
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
             "    >>> from mfv2d._mfv2d import lagrange1d\n"
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
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i}}}$\"\n"
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
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i}}}$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> # plt.legend() # No, this is too long\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n");
PyDoc_STRVAR(interp_dlagrange_doc,
             "dlagrange1d(roots: array_like, x: array_like, out: array|None = None, /) -> array\n"
             "Evaluate derivatives of Lagrange polynomials.\n"
             "\n"
             "This function efficiently evaluates Lagrange basis polynomials derivatives, defined by\n"
             "\n"
             ".. math::\n"
             "\n"
             "   \\frac{d \\mathcal{L}^n_i (x)}{d x} =\n"
             "   \\sum\\limits_{j=0,j \\neq i}^n \\prod\\limits_{k=0, k \\neq i, k \\neq j}^{n}\n"
             "   \\frac{1}{x_i - x_j} \\cdot \\frac{x - x_k}{x_i - x_k},\n"
             "\n"
             "where the ``roots`` specifies the zeros of the Polynomials :math:`\\{x_0, \\dots, x_n\\}`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "roots : array_like\n"
             "   Roots of Lagrange polynomials.\n"
             "x : array_like\n"
             "   Points where the derivatives of polynomials should be evaluated.\n"
             "out : array, optional\n"
             "   Array where the results should be written to. If not given, a new one will be\n"
             "   created and returned. It should have the same shape as ``x``, but with an extra\n"
             "   dimension added, the length of which is ``len(roots)``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "   Array of Lagrange polynomial derivatives at positions specified by ``x``.\n"

             "Examples\n"
             "--------\n"
             "This example here shows the most basic use of the function to evaluate derivatives of Lagrange\n"
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
             "    >>> from mfv2d._mfv2d import dlagrange1d\n"
             "    >>>\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)\n"
             "    >>> yvals = dlagrange1d(roots, xpos)\n"
             "\n"
             "Note that if we were to give an output array to write to, it would also be the\n"
             "return value of the function (as in no copy is made).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> yvals is dlagrange1d(roots, xpos, yvals)\n"
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
             "    ...         label=f\"${{\\\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\\\prime$\"\n"
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
             "    >>> yvals = dlagrange1d(roots, xpos)\n"
             "    >>> t1 = perf_counter()\n"
             "    >>> print(f\"Calculations took {t1 - t0: e} seconds.\")\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"${{\\\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\\\prime$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> # plt.legend() # No, this is too long\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n");

static PyObject *check_bytecode(PyObject *Py_UNUSED(module), PyObject *in_expression)
{
    size_t n_expr;
    bytecode_t *bytecode = NULL;
    // Convert into bytecode
    {
        PyObject *const expression = PySequence_Fast(in_expression, "Can not convert expression to sequence.");
        if (!expression)
            return NULL;

        n_expr = PySequence_Fast_GET_SIZE(expression);
        PyObject **const p_exp = PySequence_Fast_ITEMS(expression);

        bytecode = PyMem_RawMalloc(sizeof(*bytecode) * (n_expr + 1));
        if (!bytecode)
        {
            Py_DECREF(expression);
            return NULL;
        }
        unsigned unused;
        if (!convert_bytecode(n_expr, bytecode, p_exp, &unused, 0))
        {
            PyMem_RawFree(bytecode);
            Py_DECREF(expression);
            return NULL;
        }
        Py_DECREF(expression);
    }

    PyTupleObject *out_tuple = (PyTupleObject *)PyTuple_New((Py_ssize_t)n_expr);
    for (unsigned i = 1; i <= n_expr; ++i)
    {
        switch (bytecode[i].op)
        {
        case MATOP_IDENTITY:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_IDENTITY));
            break;
        case MATOP_MATMUL:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MATMUL));
            break;
        case MATOP_SCALE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SCALE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].f64));
            break;
        case MATOP_SUM:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SUM));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].u32));
            break;
        case MATOP_INCIDENCE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_INCIDENCE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_MASS:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MASS));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_PUSH:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_PUSH));
            break;
        default:
            ASSERT(0, "Invalid operation.");
            break;
        }
    }

    PyMem_RawFree(bytecode);

    return (PyObject *)out_tuple;
}

static PyObject *check_incidence(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *in_x;
    unsigned order, form;
    int transpose, right;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OIIpp", (char *[6]){"", "order", "form", "transpose", "right", NULL},
                                     &in_x, &order, &form, &transpose, &right))
    {
        return NULL;
    }
    if (form > 1)
    {
        PyErr_Format(PyExc_ValueError, "Form specified is too high (%u, but only up to 1 is allowed).", order);
        return NULL;
    }
    const incidence_type_t t = ((incidence_type_t)form) + (transpose ? (INCIDENCE_TYPE_10_T - INCIDENCE_TYPE_10) : 0);
    PyArrayObject *const x = (PyArrayObject *)PyArray_FromAny(in_x, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!x)
        return NULL;

    const matrix_full_t in = {.base = {.type = MATRIX_TYPE_FULL, .rows = PyArray_DIM(x, 0), .cols = PyArray_DIM(x, 1)},
                              .data = PyArray_DATA(x)};
    matrix_full_t out;
    mfv2d_result_t res;
    if (right)
    {
        res = apply_incidence_to_full_right(t, order, &in, &out, &SYSTEM_ALLOCATOR);
    }
    else
    {
        res = apply_incidence_to_full_left(t, order, &in, &out, &SYSTEM_ALLOCATOR);
    }
    Py_DECREF(x);
    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not apply the incidence matrix %u (order %u) to a %u by %u matrix, reason: %s.", t, order,
                     in.base.rows, in.base.cols, mfv2d_result_str(res));
        return NULL;
    }
    PyArrayObject *const y = matrix_full_to_array(&out);
    deallocate(&SYSTEM_ALLOCATOR, out.data);
    return (PyObject *)y;
}

static PyMethodDef module_methods[] = {
    {"check_bytecode", check_bytecode, METH_O, "Convert bytecode to C-values, then back to Python."},
    {"check_incidence", (void *)check_incidence, METH_VARARGS | METH_KEYWORDS,
     "Apply the incidence matrix to the input matrix."},
    {"lagrange1d", interp_lagrange, METH_VARARGS, interp_lagrange_doc},
    {"dlagrange1d", interp_dlagrange, METH_VARARGS, interp_dlagrange_doc},
    {.ml_name = "compute_gll",
     .ml_meth = (void *)compute_gauss_lobatto_nodes,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = compute_gll_docstring},
    {.ml_name = "compute_element_matrix_test",
     .ml_meth = (void *)compute_element_mass_matrices,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = compute_element_mass_matrices_docstr},
    {.ml_name = "compute_element_matrix",
     .ml_meth = (void *)compute_element_matrix,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "compute_element_matrix(form_orders: Sequence[int], expressions: _CompiledCodeMatrix, corners: NDArray, "
               "vector_fields: Sequence[npt.NDArray[np.float64]], basis: Basis2D, stack_memory: int = 1 << 24,"
               ") -> NDArray\n"
               "Compute a single element matrix.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "form_orders : Sequence of int\n"
               "    Orders of differential forms for the degrees of freedom. Must be between 0 and 2.\n"
               "\n"
               "expressions\n"
               "    Compiled bytecode to execute.\n"
               "\n"
               "corners : (4, 2) array\n"
               "    Array of corners of the element.\n"
               "\n"
               "vector_fields : Sequence of arrays\n"
               "    Vector field arrays as required for interior product evaluations.\n"
               "\n"
               "basis : Basis2D\n"
               "    Basis functions with integration rules to use.\n"
               "\n"
               "stack_memory : int, default: 1 << 24\n"
               "    Amount of memory to use for the evaluation stack.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "array\n"
               "    Element matrix for the specified system.\n"

    },
    {.ml_name = "compute_element_projector",
     .ml_meth = (void *)compute_element_projector,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = compute_element_projector_docstr},
    {.ml_name = "compute_element_mass_matrix",
     .ml_meth = (void *)compute_element_mass_matrix,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = compute_element_mass_matrix_docstr},
    {.ml_name = "compute_element_vector",
     .ml_meth = (void *)compute_element_vector,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc =
         "compute_element_vector(form_orders: Sequence[int], expressions: _CompiledCodeMatrix, corners: array, "
         "vector_fields: Sequence[array], basis: Basis2D, solution: array, stack_memory: int = 1 << 24) -> array\n"
         "Compute a single element forcing.\n"
         "\n"
         "Parameters\n"
         "----------\n"
         "form_orders : Sequence of int\n"
         "    Orders of differential forms for the degrees of freedom. Must be between 0 and 2.\n"
         "\n"
         "expressions\n"
         "    Compiled bytecode to execute.\n"
         "\n"
         "corners : (4, 2) array\n"
         "    Array of corners of the element.\n"
         "\n"
         "vector_fields : Sequence of arrays\n"
         "    Vector field arrays as required for interior product evaluations.\n"
         "\n"
         "basis : Basis2D\n"
         "    Basis functions with integration rules to use.\n"
         "\n"
         "solution : array\n"
         "    Array with degrees of freedom for the element.\n"
         "\n"
         "stack_memory : int, default: 1 << 24\n"
         "    Amount of memory to use for the evaluation stack.\n"
         "\n"
         "Returns\n"
         "-------\n"
         "array\n"
         "    Element vector for the specified system.\n"

    },
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "mfv2d._mfv2d",
                             .m_doc = "Internal C-extension implementing required functionality.",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__mfv2d(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject *mod = NULL;
    if (!((mod = PyModule_Create(&module))) || PyModule_AddType(mod, &geo_id_type_object) < 0 ||
        PyModule_AddType(mod, &line_type_object) < 0 || PyModule_AddType(mod, &surface_type_object) < 0 ||
        PyModule_AddType(mod, &manifold_type_object) < 0 || PyModule_AddType(mod, &manifold2d_type_object) < 0 ||
        PyModule_AddType(mod, &svec_type_object) < 0 || PyModule_AddType(mod, &givens_rotation_type_object) < 0 ||
        PyModule_AddType(mod, &lil_mat_type_object) < 0 || PyModule_AddType(mod, &givens_series_type_object) < 0 ||
        PyModule_AddType(mod, &integration_rule_1d_type) < 0 || PyModule_AddType(mod, &basis_1d_type) < 0 ||
        PyModule_AddType(mod, &basis_2d_type) < 0 || PyModule_AddType(mod, &element_mass_matrix_cache_type) < 0)
    {
        Py_XDECREF(mod);
        return NULL;
    }

    return mod;
}
