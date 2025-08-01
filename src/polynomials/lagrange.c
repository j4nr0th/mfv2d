//
// Created by jan on 29.9.2024.
//

#include "lagrange.h"

MFV2D_INTERNAL
void lagrange_polynomial_denominators(unsigned n, const double MFV2D_ARRAY_ARG(nodes, restrict static n),
                                      double MFV2D_ARRAY_ARG(denominators, restrict n))
{
    denominators[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n; ++j)
    {
        const double dif = nodes[0] - nodes[j];
        denominators[0] *= dif;
        denominators[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n; ++i)
    {
        for (unsigned j = i + 1; j < n; ++j)
        {
            const double dif = nodes[i] - nodes[j];
            denominators[i] *= +dif;
            denominators[j] *= -dif;
        }
    }
}

MFV2D_INTERNAL
void lagrange_polynomial_coefficients(unsigned n, unsigned j, const double MFV2D_ARRAY_ARG(nodes, restrict static n),
                                      double MFV2D_ARRAY_ARG(coefficients, restrict n))
{
    coefficients[0] = 1.0;
    for (unsigned i = 0; i < j; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i + 1] = 0.0;
        for (unsigned k = i + 1; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
    for (unsigned i = j + 1; i < n; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i] = 0.0;
        for (unsigned k = i; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
}

MFV2D_INTERNAL
void lagrange_polynomial_values(unsigned n_in, const double MFV2D_ARRAY_ARG(pos, static n_in), unsigned n_nodes,
                                const double MFV2D_ARRAY_ARG(x, static n_nodes),
                                double MFV2D_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                double MFV2D_ARRAY_ARG(work, restrict n_nodes))
{
    lagrange_polynomial_denominators(n_nodes, x, work);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work[i] = 1.0 / work[i];
    }

    //  Compute the numerator now
    for (unsigned k = 0; k < n_in; ++k)
    {
        double *const row = weights + n_nodes * k;
        //  First loop can be used to initialize the row
        {
            const double dif = pos[k] - x[0];
            row[0] = 1.0;
            for (unsigned j = 1; j < n_nodes; ++j)
            {
                row[j] = +dif;
            }
        }
        for (unsigned i = 1; i < n_nodes; ++i)
        {
            const double dif = pos[k] - x[i];
            for (unsigned j = 0; j < i; ++j)
            {
                row[j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_nodes; ++j)
            {
                row[j] *= +dif;
            }
        }
        //  Multiply by 1/denominator
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            row[i] *= work[i];
        }
    }
}

MFV2D_INTERNAL
void lagrange_polynomial_first_derivative(unsigned n_in, const double MFV2D_ARRAY_ARG(pos, static n_in),
                                          unsigned n_nodes, const double MFV2D_ARRAY_ARG(x, static n_nodes),
                                          double MFV2D_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                          /* cache for denominators (once per fn) */
                                          double MFV2D_ARRAY_ARG(work1, restrict n_nodes),
                                          /* cache for differences (once per node) */
                                          double MFV2D_ARRAY_ARG(work2, restrict n_nodes))
{
    // compute denominators
    lagrange_polynomial_denominators(n_nodes, x, work1);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work1[i] = 1.0 / work1[i];
    }

    //  Now loop per node
    for (unsigned ipos = 0; ipos < n_in; ++ipos)
    {
        const double v = pos[ipos];
        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Compute the differences
            work2[j] = v - x[j];
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = i + 1; k < n_nodes; ++k)
                {
                    dlijdx *= work2[k];
                }
                //  L_i^j and L_j^i have same numerators
                weights[n_nodes * ipos + j] += dlijdx;
                weights[n_nodes * ipos + i] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] *= work1[j];
        }
    }
}

PyObject *interp_lagrange(PyObject *Py_UNUSED(module), PyObject *args)
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

MFV2D_INTERNAL
const char interp_lagrange_doc[] =
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
    "    >>> plt.show()\n";

PyObject *interp_dlagrange(PyObject *Py_UNUSED(module), PyObject *args)
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

MFV2D_INTERNAL
const char interp_dlagrange_doc[] =
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
    "    >>> plt.show()\n";
