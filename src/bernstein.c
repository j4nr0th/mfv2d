//
// Created by jan on 5.11.2024.
//

#include "bernstein.h"
#include "basis1d.h"
#include "polynomial1d.h"
#include <numpy/arrayobject.h>

INTERPLIB_INTERNAL
void bernstein_from_power_series(unsigned n, double INTERPLIB_ARRAY_ARG(coeffs, static n))
{
    unsigned base_coefficient = 1;
    for (unsigned k = 0; k < n; ++k)
    {
        const double beta = coeffs[k];

        // Update the remaining entries
        const unsigned diff = n - k - 1;
        int local = (int)(diff);
        for (int i = 1; i < diff + 1; ++i)
        {
            coeffs[k + i] += beta * (double)local;
            // Incorporate the (-1)^i into the binomial coefficient
            local = (local * ((int)i - (int)diff)) / (int)(i + 1);
        }

        coeffs[k] = beta / (double)base_coefficient;
        // Update the binomial coefficient of the polynomial
        base_coefficient = (base_coefficient * (diff)) / (k + 1);
    }
}

static inline void internal_bernstein_interpolation_vector(double t, unsigned n,
                                                           double INTERPLIB_ARRAY_ARG(out, restrict n + 1))
{
    //  Bernstein polynomials follow the following recursion:
    //
    //  B^{N+1}_k(t) = t B^{N}_{k-1}(t) + (1-t) B^{N}_k(t)
    const double a = t;
    const double b = 1.0 - t;
    //  this is to stor the value about to be overridden
    out[0] = 1.0;

    for (unsigned i = 0; i < n; ++i)
    {
        out[i + 1] = out[i] * b;
        for (unsigned j = i; j > 0; --j)
        {
            out[j] = b * out[j - 1] + a * out[j];
        }
        out[0] *= a;
    }
}

INTERPLIB_INTERNAL
void bernstein_interpolation_vector(double t, unsigned n, double INTERPLIB_ARRAY_ARG(out, restrict n))
{
    internal_bernstein_interpolation_vector(t, n, out);
}

INTERPLIB_INTERNAL
const char bernstein_interpolation_matrix_doc[] =
    "bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]\n"
    "Compute Bernstein polynomials of given order at given locations.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "n : int\n"
    "   Order of polynomials used.\n"
    "x : (M,) array_like\n"
    "   Flat array of locations where the values should be interpolated.\n"
    "\n"
    "Returns\n"
    "--------\n"
    "(M, n) array"
    "   Matrix containing values of Bernstein polynomial :math:`B^M_j(x_i)` as the\n"
    "   element ``array[i, j]``.\n";
INTERPLIB_INTERNAL
PyObject *bernstein_interpolation_matrix(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function only takes two arguments.");
        return NULL;
    }
    const unsigned long order = PyLong_AsUnsignedLong(args[0]);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    PyArrayObject *array_in = (PyArrayObject *)PyArray_FromAny(args[1], PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                                               NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array_in)
    {
        return NULL;
    }

    const npy_intp len[2] = {PyArray_SIZE(array_in), (npy_intp)order + 1};
    PyArrayObject *array_out = (PyArrayObject *)PyArray_SimpleNew(2, len, NPY_DOUBLE);
    if (!array_out)
    {
        Py_DECREF(array_in);
        return NULL;
    }

    const double *restrict p_in = PyArray_DATA(array_in);
    double *restrict p_out = PyArray_DATA(array_out);

    //  May be made parallel in another version of the function.
    for (npy_intp i = 0; i < len[0]; ++i)
    {
        internal_bernstein_interpolation_vector(p_in[i], order, p_out + i * len[1]);
    }

    Py_DECREF(array_in);

    return (PyObject *)array_out;
}

INTERPLIB_INTERNAL
const char bernstein_coefficients_doc[] = "bernstein_coefficients(x: array_like, /) -> array\n"
                                          "\n"
                                          "Compute Bernstein polynomial coefficients from a power series polynomial.\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "x : array_like\n"
                                          "   Coefficients of the polynomial from 0-th to the highest order.\n"
                                          "\n"
                                          "Returns\n"
                                          "-------\n"
                                          "array\n"
                                          "   Array of coefficients of Bernstein polynomial series.\n";

INTERPLIB_INTERNAL
PyObject *bernstein_coefficients(PyObject *Py_UNUSED(self), PyObject *arg)
{
    PyArrayObject *const input_coeffs =
        (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSURECOPY, NULL);
    if (!input_coeffs)
        return NULL;

    bernstein_from_power_series(PyArray_DIM(input_coeffs, 0), PyArray_DATA(input_coeffs));

    return (PyObject *)input_coeffs;
}
