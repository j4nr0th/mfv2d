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
#include "topology/manifold1d.h"
#include "topology/manifold2d.h"
#include "topology/surfaceobject.h"

// Evaluation
#include "eval/allocator.h"
#include "eval/element_system.h"
#include "eval/evaluation.h"
#include "eval/incidence.h"
#include "eval/precomp.h"

// Solver
#include "solve/givens.h"
#include "solve/lil_matrix.h"
#include "solve/svector.h"

// Basis

#include "basis/gausslobatto.h"
#include "basis/lagrange.h"
#include "eval/basis.h"
#include "eval/fem_space.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static void caches_array_destroy(unsigned n, basis_precomp_t array[static n])
{
    for (unsigned i = n; i > 0; --i)
    {
        basis_precomp_destroy(array + (i - 1));
    }
}

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
             "    >>> from interplib import dlagrange1d\n"
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
static PyObject *compute_element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *ret_val = NULL;
    PyObject *in_form_orders;
    PyObject *in_expressions;
    PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
    PyObject *element_orders;
    PyObject *cache_contents;
    PyTupleObject *vector_field_tuple;
    PyObject *element_offsets;
    Py_ssize_t thread_stack_size = (1 << 24);
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOOOOOO!OO|n",
            (char *[12]){"form_orders", "expressions", "pos_bl", "pos_br", "pos_tr", "pos_tl", "element_orders",
                         "vector_fields", "element_field_offsets", "cache_contents", "thread_stack_size", NULL},
            &in_form_orders, &in_expressions, &pos_bl, &pos_br, &pos_tr, &pos_tl, &element_orders, &PyTuple_Type,
            &vector_field_tuple, &element_offsets, &cache_contents, &thread_stack_size))
    {
        return NULL;
    }

    // Check that the number of vector fields is not too high
    if (PyTuple_GET_SIZE(vector_field_tuple) >= VECTOR_FIELDS_MAX)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Number of vector fields given (%zu) is over maximum supported value (VECTOR_FIELDS_MAX = %u).",
                     (size_t)PyTuple_GET_SIZE(vector_field_tuple), (unsigned)VECTOR_FIELDS_MAX);
        return NULL;
    }

    if (thread_stack_size < 0)
    {
        PyErr_Format(PyExc_ValueError, "Thread stack size can not be negative (%lld).",
                     (long long int)thread_stack_size);
        return NULL;
    }

    // Round the stack size up to the nearest 8.
    if ((thread_stack_size & 7) != 0)
    {
        thread_stack_size += 8 - (thread_stack_size & 7);
    }

    // Create the system template
    const size_t field_count = PyTuple_GET_SIZE(vector_field_tuple);
    system_template_t system_template;
    if (!system_template_create(&system_template, in_form_orders, in_expressions, (unsigned)field_count,
                                &SYSTEM_ALLOCATOR))
        return NULL;

    // Create caches
    PyObject *cache_seq = PySequence_Fast(cache_contents, "BasisCaches must be a sequence of tuples.");
    if (!cache_seq)
    {
        goto after_template;
    }
    const unsigned n_cache = PySequence_Fast_GET_SIZE(cache_seq);

    basis_precomp_t *cache_array = allocate(&SYSTEM_ALLOCATOR, sizeof *cache_array * n_cache);
    if (!cache_array)
    {
        Py_DECREF(cache_seq);
        goto after_template;
    }

    for (unsigned i = 0; i < n_cache; ++i)
    {
        int failed = !basis_precomp_create(PySequence_Fast_GET_ITEM(cache_seq, i), cache_array + i);
        if (!failed)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                if (cache_array[i].order == cache_array[j].order)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Cache contains the values for order as entries with indices %u and %u.", i, j);
                    failed = 1;
                    break;
                }
            }
        }

        if (failed)
        {
            caches_array_destroy(i, cache_array);
            deallocate(&SYSTEM_ALLOCATOR, cache_array);
            Py_DECREF(cache_seq);
            goto after_template;
        }
    }
    Py_DECREF(cache_seq);

    // Convert coordinate arrays
    PyArrayObject *const bl_array = (PyArrayObject *)PyArray_FromAny(pos_bl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const br_array = (PyArrayObject *)PyArray_FromAny(pos_br, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tr_array = (PyArrayObject *)PyArray_FromAny(pos_tr, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tl_array = (PyArrayObject *)PyArray_FromAny(pos_tl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const orders_array = (PyArrayObject *)PyArray_FromAny(
        element_orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    PyArrayObject *const offset_array = (PyArrayObject *)PyArray_FromAny(
        element_offsets, PyArray_DescrFromType(NPY_UINT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    if (!bl_array || !br_array || !tr_array || !tl_array || !orders_array || !offset_array)
    {
        Py_XDECREF(offset_array);
        Py_XDECREF(orders_array);
        Py_XDECREF(tl_array);
        Py_XDECREF(tr_array);
        Py_XDECREF(br_array);
        Py_XDECREF(bl_array);
        goto after_cache;
    }
    size_t n_elements, n_field_points;
    {
        n_elements = PyArray_DIM(orders_array, 0);
        const npy_intp *dims_bl = PyArray_DIMS(bl_array);
        const npy_intp *dims_br = PyArray_DIMS(br_array);
        const npy_intp *dims_tr = PyArray_DIMS(tr_array);
        const npy_intp *dims_tl = PyArray_DIMS(tl_array);
        if (dims_bl[0] != n_elements || (dims_bl[0] != dims_br[0] || dims_bl[1] != dims_br[1]) ||
            (dims_bl[0] != dims_tr[0] || dims_bl[1] != dims_tr[1]) ||
            (dims_bl[0] != dims_tl[0] || dims_bl[1] != dims_tl[1]) || dims_bl[1] != 2 ||
            PyArray_SIZE(offset_array) != n_elements + 1)
        {
            PyErr_SetString(PyExc_ValueError, "All coordinate input arrays, orders array, and offset array must be "
                                              "have same number of 2 component vectors.");
            goto after_arrays;
        }
        n_field_points = ((const npy_uint64 *)PyArray_DATA(offset_array))[n_elements];
    }

    field_information_t vector_fields = {.n_fields = field_count, .offsets = PyArray_DATA(offset_array)};
    // Check that the vector field arrays have the correct shape
    for (unsigned i = 0; i < field_count; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(vector_field_tuple, i);
        // Check type
        if (!PyArray_Check(o))
        {
            PyErr_Format(PyExc_ValueError, "Vector field tuple entry %u was not a Numpy array, but %R.", i, Py_TYPE(o));
            goto after_arrays;
        }
        PyArrayObject *const vec_field = (PyArrayObject *)o;

        // Check data type
        if (PyArray_TYPE(vec_field) != NPY_FLOAT64)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had an incorrect data type with dtype %u (should be float64 - %u).", i,
                         PyArray_TYPE(vec_field), NPY_FLOAT64);
            goto after_arrays;
        }

        // Check dim count
        if (PyArray_NDIM(vec_field) != 2)
        {
            PyErr_Format(PyExc_ValueError, "Vector field %u did not have two axis, but had %u instead.", i,
                         (unsigned)PyArray_NDIM(vec_field));
            goto after_arrays;
        }
        const npy_intp *dims = PyArray_DIMS(vec_field);

        // Check dims
        if (dims[0] != n_field_points || dims[1] != 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had the incorrect shape (%zu, %zu). Based on the offset array it was "
                         "expected to be (%zu, %zu) instead.",
                         i, (size_t)dims[0], (size_t)dims[1], n_field_points, (size_t)2);
            goto after_arrays;
        }

        const unsigned required_flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
        const unsigned flags = PyArray_FLAGS(vec_field);
        if ((flags & required_flags) != required_flags)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u did not have the required flags %04x, but instead had flags %04x.", i,
                         required_flags, flags);
            goto after_arrays;
        }

        vector_fields.fields[i] = (const double *)PyArray_DATA(vec_field);
    }

    // Extract C pointers

    const double *restrict const coord_bl = PyArray_DATA(bl_array);
    const double *restrict const coord_br = PyArray_DATA(br_array);
    const double *restrict const coord_tr = PyArray_DATA(tr_array);
    const double *restrict const coord_tl = PyArray_DATA(tl_array);
    const unsigned *restrict const orders = PyArray_DATA(orders_array);

    // Prepare output arrays
    double **p_out = allocate(&SYSTEM_ALLOCATOR, sizeof(*p_out) * n_elements);
    if (!p_out)
    {
        goto after_arrays;
    }
    ret_val = PyTuple_New((Py_ssize_t)n_elements);
    if (!ret_val)
    {
        deallocate(&SYSTEM_ALLOCATOR, p_out);
        goto after_arrays;
    }

    // Create an error stack for reporting issues

    for (unsigned i = 0; i < n_elements; ++i)
    {
        size_t element_size = 0;
        for (unsigned j = 0; j < system_template.n_forms; ++j)
        {
            element_size += form_degrees_of_freedom_count(system_template.form_orders[j], orders[i], orders[i]);
        }
        const npy_intp dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
        PyArrayObject *const a = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!a)
        {
            Py_DECREF(ret_val);
            ret_val = NULL;
            deallocate(&SYSTEM_ALLOCATOR, p_out);
            goto after_arrays;
        }
        PyTuple_SET_ITEM(ret_val, i, a);
        p_out[i] = PyArray_DATA(a);
        memset(p_out[i], 0, sizeof(*p_out[i]) * dims[0] * dims[1]);
    }

    mfv2d_result_t common_res = MFV2D_SUCCESS;
    Py_BEGIN_ALLOW_THREADS

#pragma omp parallel default(none)                                                                                     \
    shared(SYSTEM_ALLOCATOR, system_template, stderr, common_res, n_elements, orders, cache_array, n_cache, coord_bl,  \
               coord_br, coord_tr, coord_tl, p_out, thread_stack_size, vector_fields)
    {
        error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
        // Allocate the stack through system allocator
        matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
        allocator_stack_t *const allocator_stack = allocator_stack_create(thread_stack_size, &SYSTEM_ALLOCATOR);
        mfv2d_result_t res = matrix_stack && err_stack ? MFV2D_SUCCESS : MFV2D_FAILED_ALLOC;
        /* Heavy calculations here */
#pragma omp for nowait
        for (unsigned i_elem = 0; i_elem < n_elements; ++i_elem)
        {
            if (!(common_res == MFV2D_SUCCESS && err_stack && matrix_stack && allocator_stack))
            {
                continue;
            }
            allocator_stack_reset(allocator_stack);
            const unsigned order = orders[i_elem];
            size_t element_size = 0;
            for (unsigned j = 0; j < system_template.n_forms; ++j)
            {
                element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order, order);
            }
            precompute_t precomp;
            unsigned i;
            // Find cached values for the current order
            for (i = 0; i < n_cache; ++i)
            {
                if (order == cache_array[i].order)
                {
                    break;
                }
            }
            if (i == n_cache)
            {
                // Failed, not in cache!
                continue;
            }
            // Compute matrices for the element
            if (!precompute_create(cache_array + i, coord_bl[2 * i_elem + 0], coord_br[2 * i_elem + 0],
                                   coord_tr[2 * i_elem + 0], coord_tl[2 * i_elem + 0], coord_bl[2 * i_elem + 1],
                                   coord_br[2 * i_elem + 1], coord_tr[2 * i_elem + 1], coord_tl[2 * i_elem + 1],
                                   &precomp, &allocator_stack->base))
            {
                // Failed, could not compute precomp
                continue;
            }

            double *restrict const output_mat = p_out[i_elem];

            // Compute the individual entries
            size_t row_offset = 0;
            for (unsigned row = 0; row < system_template.n_forms && res == MFV2D_SUCCESS; ++row)
            {
                const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order, order);
                size_t col_offset = 0;
                for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
                {
                    const unsigned col_len =
                        form_degrees_of_freedom_count(system_template.form_orders[col], order, order);
                    const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
                    if (!bytecode)
                    {
                        // Zero entry, we do nothing since arrays start zeroed out (I think).
                        col_offset += col_len;
                        continue;
                    }

                    // Offset the fields to the element
                    field_information_t element_field_information = vector_fields;
                    for (unsigned idx = 0; idx < element_field_information.n_fields; ++idx)
                    {
                        element_field_information.fields[idx] += 2 * element_field_information.offsets[i_elem];
                    }

                    matrix_full_t mat;
                    res = evaluate_element_term_sibling(err_stack, system_template.form_orders[row], order, bytecode,
                                                        &precomp, &element_field_information, system_template.max_stack,
                                                        matrix_stack, &allocator_stack->base, &mat, NULL);
                    if (res != MFV2D_SUCCESS)
                    {
                        MFV2D_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                        break;
                    }
                    if (row_len != mat.base.rows || col_len != mat.base.cols)
                    {
                        MFV2D_ERROR(err_stack, MFV2D_DIMS_MISMATCH,
                                    "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                                    mat.base.rows, mat.base.cols, row_len, col_len);
                        res = MFV2D_DIMS_MISMATCH;
                        break;
                    }

                    for (unsigned i_out = 0; i_out < row_len; ++i_out)
                    {
                        for (unsigned j_out = 0; j_out < col_len; ++j_out)
                        {
                            output_mat[(i_out + row_offset) * element_size + (j_out + col_offset)] =
                                mat.data[i_out * mat.base.cols + j_out];
                        }
                    }

                    deallocate(&allocator_stack->base, mat.data);
                    // SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, mat.data);
                    col_offset += col_len;
                }
                row_offset += row_len;
            }
        }

        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);

        // Clean error stack
        if (err_stack && err_stack->position != 0)
        {
            fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);

            for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
            {
                const error_message_t *msg = err_stack->messages + i_e;
                fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                        mfv2d_result_str(msg->code), msg->message);
                deallocate(err_stack->allocator, msg->message);
            }
        }
        if (err_stack)
            deallocate(err_stack->allocator, err_stack);
        if (allocator_stack)
        {
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        }
#pragma omp critical
        {
            if (res != MFV2D_SUCCESS)
            {
                common_res = res;
            }
        }
    }

    Py_END_ALLOW_THREADS if (common_res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Execution failed with error code %s.", mfv2d_result_str(common_res));
        // Failed allocation of matrix stack.
        Py_DECREF(ret_val);
        ret_val = NULL;
    }

    // Clean up the array of output pointers
    deallocate(&SYSTEM_ALLOCATOR, p_out);

    // Clean up the coordinate arrays
after_arrays:
    Py_DECREF(offset_array);
    Py_DECREF(orders_array);
    Py_DECREF(tl_array);
    Py_DECREF(tr_array);
    Py_DECREF(br_array);
    Py_DECREF(bl_array);

    // Clean up the basis caches
after_cache:
    caches_array_destroy(n_cache, cache_array);
    deallocate(&SYSTEM_ALLOCATOR, cache_array);

    // Clean up the template
after_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    return ret_val;
}

static PyObject *compute_element_explicit(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *ret_val = NULL;
    PyObject *in_dofs;
    PyObject *in_offsets;
    PyObject *in_form_orders;
    PyObject *in_expressions;
    PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
    PyObject *element_orders;
    PyObject *cache_contents;
    PyTupleObject *vector_field_tuple;
    PyObject *element_offsets;
    Py_ssize_t thread_stack_size = (1 << 24);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOOO!OO|n",
                                     (char *[14]){"dofs", "offsets", "form_orders", "expressions", "pos_bl", "pos_br",
                                                  "pos_tr", "pos_tl", "element_orders", "vector_fields",
                                                  "element_field_offsets", "cache_contents", "thread_stack_size", NULL},
                                     &in_dofs, &in_offsets, &in_form_orders, &in_expressions, &pos_bl, &pos_br, &pos_tr,
                                     &pos_tl, &element_orders, &PyTuple_Type, &vector_field_tuple, &element_offsets,
                                     &cache_contents, &thread_stack_size))
    {
        return NULL;
    }

    // Check that the number of vector fields is not too high
    if (PyTuple_GET_SIZE(vector_field_tuple) >= VECTOR_FIELDS_MAX)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Number of vector fields given (%zu) is over maximum supported value (VECTOR_FIELDS_MAX = %u).",
                     (size_t)PyTuple_GET_SIZE(vector_field_tuple), (unsigned)VECTOR_FIELDS_MAX);
        return NULL;
    }

    if (thread_stack_size < 0)
    {
        PyErr_Format(PyExc_ValueError, "Thread stack size can not be negative (%lld).",
                     (long long int)thread_stack_size);
        return NULL;
    }

    // Round the stack size up to the nearest 8.
    if ((thread_stack_size & 7) != 0)
    {
        thread_stack_size += 8 - (thread_stack_size & 7);
    }

    // Create the system template
    const size_t field_count = PyTuple_GET_SIZE(vector_field_tuple);
    system_template_t system_template;
    if (!system_template_create(&system_template, in_form_orders, in_expressions, (unsigned)field_count,
                                &SYSTEM_ALLOCATOR))
        return NULL;

    // Create caches
    PyObject *cache_seq = PySequence_Fast(cache_contents, "BasisCaches must be a sequence of tuples.");
    if (!cache_seq)
    {
        goto after_template;
    }
    const unsigned n_cache = PySequence_Fast_GET_SIZE(cache_seq);

    basis_precomp_t *cache_array = allocate(&SYSTEM_ALLOCATOR, sizeof *cache_array * n_cache);
    if (!cache_array)
    {
        Py_DECREF(cache_seq);
        goto after_template;
    }

    for (unsigned i = 0; i < n_cache; ++i)
    {
        int failed = !basis_precomp_create(PySequence_Fast_GET_ITEM(cache_seq, i), cache_array + i);
        if (!failed)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                if (cache_array[i].order == cache_array[j].order)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Cache contains the values for order as entries with indices %u and %u.", i, j);
                    failed = 1;
                    break;
                }
            }
        }

        if (failed)
        {
            caches_array_destroy(i, cache_array);
            deallocate(&SYSTEM_ALLOCATOR, cache_array);
            Py_DECREF(cache_seq);
            goto after_template;
        }
    }
    Py_DECREF(cache_seq);

    // Convert coordinate arrays
    PyArrayObject *const bl_array = (PyArrayObject *)PyArray_FromAny(pos_bl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const br_array = (PyArrayObject *)PyArray_FromAny(pos_br, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tr_array = (PyArrayObject *)PyArray_FromAny(pos_tr, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tl_array = (PyArrayObject *)PyArray_FromAny(pos_tl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const orders_array = (PyArrayObject *)PyArray_FromAny(
        element_orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    // Other element arrays
    PyArrayObject *const vec_field_offset_array = (PyArrayObject *)PyArray_FromAny(
        element_offsets, PyArray_DescrFromType(NPY_UINT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    PyArrayObject *const offset_array = (PyArrayObject *)PyArray_FromAny(
        in_offsets, PyArray_DescrFromType(NPY_UINT32), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const dofs_array = (PyArrayObject *)PyArray_FromAny(
        in_dofs, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    if (!bl_array || !br_array || !tr_array || !tl_array || !orders_array || !vec_field_offset_array || !offset_array ||
        !dofs_array)
    {
        Py_XDECREF(dofs_array);
        Py_XDECREF(offset_array);
        Py_XDECREF(vec_field_offset_array);
        Py_XDECREF(orders_array);
        Py_XDECREF(tl_array);
        Py_XDECREF(tr_array);
        Py_XDECREF(br_array);
        Py_XDECREF(bl_array);
        goto after_cache;
    }
    size_t n_elements, n_field_points;
    {
        n_elements = PyArray_DIM(orders_array, 0);
        const npy_intp *dims_bl = PyArray_DIMS(bl_array);
        const npy_intp *dims_br = PyArray_DIMS(br_array);
        const npy_intp *dims_tr = PyArray_DIMS(tr_array);
        const npy_intp *dims_tl = PyArray_DIMS(tl_array);
        if (dims_bl[0] != n_elements || (dims_bl[0] != dims_br[0] || dims_bl[1] != dims_br[1]) ||
            (dims_bl[0] != dims_tr[0] || dims_bl[1] != dims_tr[1]) ||
            (dims_bl[0] != dims_tl[0] || dims_bl[1] != dims_tl[1]) || dims_bl[1] != 2 ||
            PyArray_SIZE(vec_field_offset_array) != n_elements + 1)
        {
            PyErr_SetString(PyExc_ValueError, "All coordinate input arrays, orders array, and offset array must be "
                                              "have same number of 2 component vectors.");
            goto after_arrays;
        }
        n_field_points = ((const npy_uint64 *)PyArray_DATA(vec_field_offset_array))[n_elements];
    }

    field_information_t vector_fields = {.n_fields = field_count, .offsets = PyArray_DATA(vec_field_offset_array)};
    // Check that the vector field arrays have the correct shape
    for (unsigned i = 0; i < field_count; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(vector_field_tuple, i);
        // Check type
        if (!PyArray_Check(o))
        {
            PyErr_Format(PyExc_ValueError, "Vector field tuple entry %u was not a Numpy array, but %R.", i, Py_TYPE(o));
            goto after_arrays;
        }
        PyArrayObject *const vec_field = (PyArrayObject *)o;

        // Check data type
        if (PyArray_TYPE(vec_field) != NPY_FLOAT64)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had an incorrect data type with dtype %u (should be float64 - %u).", i,
                         PyArray_TYPE(vec_field), NPY_FLOAT64);
            goto after_arrays;
        }

        // Check dim count
        if (PyArray_NDIM(vec_field) != 2)
        {
            PyErr_Format(PyExc_ValueError, "Vector field %u did not have two axis, but had %u instead.", i,
                         (unsigned)PyArray_NDIM(vec_field));
            goto after_arrays;
        }
        const npy_intp *dims = PyArray_DIMS(vec_field);

        // Check dims
        if (dims[0] != n_field_points || dims[1] != 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had the incorrect shape (%zu, %zu). Based on the offset array it was "
                         "expected to be (%zu, %zu) instead.",
                         i, (size_t)dims[0], (size_t)dims[1], n_field_points, (size_t)2);
            goto after_arrays;
        }

        const unsigned required_flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
        const unsigned flags = PyArray_FLAGS(vec_field);
        if ((flags & required_flags) != required_flags)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u did not have the required flags %04x, but instead had flags %04x.", i,
                         required_flags, flags);
            goto after_arrays;
        }

        vector_fields.fields[i] = (const double *)PyArray_DATA(vec_field);
    }

    // Check that the dof and offset arrays have the correct shapes
    if (PyArray_DIM(offset_array, 0) != n_elements)
    {
        PyErr_Format(PyExc_ValueError, "DoF offset array only has %zu entries, but there are %u elements to compute.",
                     (size_t)PyArray_DIM(offset_array, 0), n_elements);
        goto after_arrays;
    }
    const size_t total_dofs = PyArray_DIM(dofs_array, 0);

    // Extract C pointers
    const npy_uint32 *restrict const dof_offsets = PyArray_DATA(offset_array);
    const double *restrict const dofs = PyArray_DATA(dofs_array);
    const double *restrict const coord_bl = PyArray_DATA(bl_array);
    const double *restrict const coord_br = PyArray_DATA(br_array);
    const double *restrict const coord_tr = PyArray_DATA(tr_array);
    const double *restrict const coord_tl = PyArray_DATA(tl_array);
    const unsigned *restrict const orders = PyArray_DATA(orders_array);

    // Prepare output arrays
    double **p_out = allocate(&SYSTEM_ALLOCATOR, sizeof(*p_out) * n_elements);
    if (!p_out)
    {
        goto after_arrays;
    }
    ret_val = PyTuple_New((Py_ssize_t)n_elements);
    if (!ret_val)
    {
        deallocate(&SYSTEM_ALLOCATOR, p_out);
        goto after_arrays;
    }

    // Create an error stack for reporting issues

    for (unsigned i = 0; i < n_elements; ++i)
    {
        size_t element_size = 0;
        for (unsigned j = 0; j < system_template.n_forms; ++j)
        {
            element_size += form_degrees_of_freedom_count(system_template.form_orders[j], orders[i], orders[i]);
        }
        // Check if too many DoFs
        if (element_size + dof_offsets[i] > total_dofs)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Element %u has %u degrees of freedom with offset of %zu, while the total number of DoFs is %zu.", i,
                element_size, dof_offsets[i], total_dofs);
            Py_DECREF(ret_val);
            ret_val = NULL;
            deallocate(&SYSTEM_ALLOCATOR, p_out);
            goto after_arrays;
        }
        const npy_intp dims[1] = {(npy_intp)element_size};
        PyArrayObject *const a = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!a)
        {
            Py_DECREF(ret_val);
            ret_val = NULL;
            deallocate(&SYSTEM_ALLOCATOR, p_out);
            goto after_arrays;
        }
        PyTuple_SET_ITEM(ret_val, i, a);
        p_out[i] = PyArray_DATA(a);
        memset(p_out[i], 0, sizeof(*p_out[i]) * dims[0]);
    }

    mfv2d_result_t common_res = MFV2D_SUCCESS;
    Py_BEGIN_ALLOW_THREADS

#pragma omp parallel default(none)                                                                                     \
    shared(SYSTEM_ALLOCATOR, system_template, stderr, common_res, n_elements, orders, cache_array, n_cache, coord_bl,  \
               coord_br, coord_tr, coord_tl, p_out, thread_stack_size, vector_fields, dofs, dof_offsets)
    {
        error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
        // Allocate the stack through system allocator
        matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
        allocator_stack_t *const allocator_stack = allocator_stack_create(thread_stack_size, &SYSTEM_ALLOCATOR);
        mfv2d_result_t res = matrix_stack && err_stack ? MFV2D_SUCCESS : MFV2D_FAILED_ALLOC;
        /* Heavy calculations here */
#pragma omp for nowait
        for (unsigned i_elem = 0; i_elem < n_elements; ++i_elem)
        {
            if (!(common_res == MFV2D_SUCCESS && err_stack && matrix_stack && allocator_stack))
            {
                continue;
            }
            allocator_stack_reset(allocator_stack);
            const unsigned order = orders[i_elem];
            size_t element_size = 0;
            for (unsigned j = 0; j < system_template.n_forms; ++j)
            {
                element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order, order);
            }
            precompute_t precomp;
            unsigned i;
            // Find cached values for the current order
            for (i = 0; i < n_cache; ++i)
            {
                if (order == cache_array[i].order)
                {
                    break;
                }
            }
            if (i == n_cache)
            {
                // Failed, not in cache!
                continue;
            }
            // Compute matrices for the element
            if (!precompute_create(cache_array + i, coord_bl[2 * i_elem + 0], coord_br[2 * i_elem + 0],
                                   coord_tr[2 * i_elem + 0], coord_tl[2 * i_elem + 0], coord_bl[2 * i_elem + 1],
                                   coord_br[2 * i_elem + 1], coord_tr[2 * i_elem + 1], coord_tl[2 * i_elem + 1],
                                   &precomp, &allocator_stack->base))
            {
                // Failed, could not compute precomp
                continue;
            }
            const size_t element_dof_offsets = dof_offsets[i_elem];

            double *restrict const output_mat = p_out[i_elem];

            // Compute the individual entries
            size_t row_offset = 0;
            for (unsigned row = 0; row < system_template.n_forms && res == MFV2D_SUCCESS; ++row)
            {
                const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order, order);
                size_t col_offset = 0;
                for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
                {
                    const size_t local_dof_offsets = element_dof_offsets + col_offset;
                    const unsigned col_len =
                        form_degrees_of_freedom_count(system_template.form_orders[col], order, order);
                    const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
                    if (!bytecode)
                    {
                        // Zero entry, we do nothing since arrays start zeroed out (I think).
                        col_offset += col_len;
                        continue;
                    }

                    // Offset the fields to the element
                    field_information_t element_field_information = vector_fields;
                    for (unsigned idx = 0; idx < element_field_information.n_fields; ++idx)
                    {
                        element_field_information.fields[idx] += 2 * element_field_information.offsets[i_elem];
                    }

                    const double *restrict const dof = dofs + local_dof_offsets;
                    matrix_full_t mat;
                    const matrix_full_t input = {.base = {.type = MATRIX_TYPE_FULL, .cols = 1, .rows = col_len},
                                                 .data = (double *)dof};
                    res = evaluate_element_term_sibling(err_stack, system_template.form_orders[row], order, bytecode,
                                                        &precomp, &element_field_information, system_template.max_stack,
                                                        matrix_stack, &allocator_stack->base, &mat, &input);
                    if (res != MFV2D_SUCCESS)
                    {
                        MFV2D_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                        break;
                    }
                    if (row_len != mat.base.rows || 1 != mat.base.cols)
                    {
                        MFV2D_ERROR(err_stack, MFV2D_DIMS_MISMATCH,
                                    "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                                    mat.base.rows, mat.base.cols, row_len, 1);
                        res = MFV2D_DIMS_MISMATCH;
                        break;
                    }
                    // printf("Block (%u, %u) input:\n\t[", row, col);
                    // for (unsigned i_out = 0; i_out < col_len; ++i_out)
                    // {
                    //     printf("%g, ", input.data[i_out]);
                    // }
                    // printf("]\nBlock (%u, %u) output:\n\t[", row, col);
                    for (unsigned i_out = 0; i_out < row_len; ++i_out)
                    {
                        output_mat[i_out + row_offset] += mat.data[i_out];

                        // printf("%g, ", mat.data[i_out]);
                    }
                    // printf("]\n\n");

                    deallocate(&allocator_stack->base, mat.data);
                    col_offset += col_len;
                }
                row_offset += row_len;
            }
        }

        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);

        // Clean error stack
        if (err_stack && err_stack->position != 0)
        {
            fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);

            for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
            {
                const error_message_t *msg = err_stack->messages + i_e;
                fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                        mfv2d_result_str(msg->code), msg->message);
                deallocate(err_stack->allocator, msg->message);
            }
        }
        if (err_stack)
            deallocate(err_stack->allocator, err_stack);
        if (allocator_stack)
        {
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        }
#pragma omp critical
        {
            if (res != MFV2D_SUCCESS)
            {
                common_res = res;
            }
        }
    }

    Py_END_ALLOW_THREADS if (common_res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Execution failed with error code %s.", mfv2d_result_str(common_res));
        // Failed allocation of matrix stack.
        Py_DECREF(ret_val);
        ret_val = NULL;
    }

    // Clean up the array of output pointers
    deallocate(&SYSTEM_ALLOCATOR, p_out);

    // Clean up the coordinate arrays
after_arrays:
    Py_XDECREF(dofs_array);
    Py_XDECREF(offset_array);
    Py_DECREF(vec_field_offset_array);
    Py_DECREF(orders_array);
    Py_DECREF(tl_array);
    Py_DECREF(tr_array);
    Py_DECREF(br_array);
    Py_DECREF(bl_array);

    // Clean up the basis caches
after_cache:
    caches_array_destroy(n_cache, cache_array);
    deallocate(&SYSTEM_ALLOCATOR, cache_array);

    // Clean up the template
after_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    return ret_val;
}

static PyObject *element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    double x0, x1, x2, x3;
    double y0, y1, y2, y3;
    PyObject *serialized;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddddddddO",
                                     (char *[10]){"x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3", "serialized", NULL},
                                     &x0, &x1, &x2, &x3, &y0, &y1, &y2, &y3, &serialized))
    {
        return NULL;
    }

    basis_precomp_t basis_precomp;
    if (!basis_precomp_create(serialized, &basis_precomp))
    {
        return NULL;
    }

    precompute_t out;
    const int res = precompute_create(&basis_precomp, x0, x1, x2, x3, y0, y1, y2, y3, &out, &SYSTEM_ALLOCATOR);

    if (!res)
    {
        return NULL;
    }
    int failed = 0;
    for (mass_mtx_indices_t t = MASS_0; t < MASS_CNT; ++t)
    {
        const matrix_full_t *m = precompute_get_matrix(&out, t, &SYSTEM_ALLOCATOR);
        if (!m)
        {
            failed = 1;
            PyErr_Format(PyExc_ValueError, "Failed allocating and crating the mass matrix %u.", (unsigned)t);
            break;
        }
    }

    PyObject *ret_val = NULL;

    if (!failed)
    {
        ret_val = PyTuple_Pack(
            6, matrix_full_to_array(out.mass_matrices + MASS_0), matrix_full_to_array(out.mass_matrices + MASS_1),
            matrix_full_to_array(out.mass_matrices + MASS_2), matrix_full_to_array(out.mass_matrices + MASS_0_I),
            matrix_full_to_array(out.mass_matrices + MASS_1_I), matrix_full_to_array(out.mass_matrices + MASS_2_I));
    }

    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.jacobian);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2_I].data);

    basis_precomp_destroy(&basis_precomp);
    return ret_val;
}

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
    {"compute_element_matrices", (void *)compute_element_matrices, METH_VARARGS | METH_KEYWORDS,
     "Compute element matrices by sibling calls."},
    {"compute_element_explicit", (void *)compute_element_explicit, METH_VARARGS | METH_KEYWORDS,
     "Compute element values by sibling calls."},
    {"element_matrices", (void *)element_matrices, METH_VARARGS | METH_KEYWORDS, "Compute element matrices."},
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
     .ml_doc = "TODO" /* TODO */},
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
     .ml_doc = "TODO" /*TODO*/},
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
        PyModule_AddType(mod, &manifold_type_object) < 0 || PyModule_AddType(mod, &manifold1d_type_object) < 0 ||
        PyModule_AddType(mod, &manifold2d_type_object) < 0 || PyModule_AddType(mod, &svec_type_object) < 0 ||
        PyModule_AddType(mod, &givens_rotation_type_object) < 0 || PyModule_AddType(mod, &lil_mat_type_object) < 0 ||
        PyModule_AddType(mod, &givens_series_type_object) < 0 || PyModule_AddType(mod, &integration_rule_1d_type) < 0 ||
        PyModule_AddType(mod, &basis_1d_type) < 0 || PyModule_AddType(mod, &basis_2d_type) < 0)
    {
        Py_XDECREF(mod);
        return NULL;
    }

    return mod;
}
