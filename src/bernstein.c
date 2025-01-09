//
// Created by jan on 5.11.2024.
//

#include "bernstein.h"
#include "basis1d.h"
#include "polynomial1d.h"
#include <numpy/arrayobject.h>

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
const char bernstein_convert_polynomial_doc[] = "bernstein1d_convert(poly: Polynomial1D) -> npt.NDArray[np.float64]\n"
                                                "Compute Bernstein coefficients from power basis polynomial.\n"
                                                "\n"
                                                "Parameters\n"
                                                "----------\n"
                                                "poly : Polynomial1D\n"
                                                "   Polynomial which to convert.\n"
                                                "\n"
                                                "Returns\n"
                                                "--------\n"
                                                "array"
                                                "   Array of coefficients of Bernstein polynomial coefficients.\n";
PyObject *bernstein_convert_polynomial(PyObject *Py_UNUSED(self), PyObject *arg)
{

    if (!Py_IS_TYPE(arg, &polynomial1d_type_object))
    {
        PyErr_Format(PyExc_TypeError, "Only a Polynomial1D object can converted.");
        return NULL;
    }
    const polynomial1d_t *poly = (polynomial1d_t *)arg;
    const npy_intp n = poly->n;
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }

    double *restrict k = PyArray_DATA(out);
    // pre-compute factorials
    k[poly->n - 1] = 1.0;
    unsigned v = 1;
    for (unsigned i = 1; i < poly->n; ++i)
    {
        v *= i;
        k[poly->n - i - 1] = (double)v;
    }

    for (unsigned i = 0; i < poly->n; ++i)
    {
        const double factorial = k[i];
        for (unsigned j = 0; j < i; ++j)
        {
            // TODO
        }
    }

    return (PyObject *)out;
}
