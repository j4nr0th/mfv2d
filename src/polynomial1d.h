//
// Created by jan on 21.10.2024.
//

#ifndef POLYNOMIAL1D_H
#define POLYNOMIAL1D_H
#include <Python.h>

#include "common_defines.h"

typedef struct
{
    PyObject_HEAD unsigned n;
    vectorcallfunc call_poly;
    double k[];
} polynomial1d_t;

INTERPLIB_INTERNAL
extern PyTypeObject polynomial1d_type_object;

/**
 * Multiply coefficients of two polynomials and write result into array out, which is allowed to overlap with the second
 * array.
 *
 * @param n1 Number of terms in the first polynomial.
 * @param k1 Terms of the first polynomial in ascending powers of x. Array must not overlap with either of the other
 * two.
 * @param n2 Number of terms in the second polynomial.
 * @param k1 Terms of the second polynomial in ascending powers of x.
 * @param out Array, which receives the output terms in ascending powers of x.
 */
INTERPLIB_INTERNAL
void multiply_polynomials(unsigned n1, const double INTERPLIB_ARRAY_ARG(k1, restrict static n1), unsigned n2,
                          const double INTERPLIB_ARRAY_ARG(k2, static n2),
                          double INTERPLIB_ARRAY_ARG(out, ((n1 - 1) + (n2 - 1) + 1)));

#endif // POLYNOMIAL1D_H
