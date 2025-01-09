//
// Created by jan on 5.11.2024.
//

#ifndef BERNSTEIN_H
#define BERNSTEIN_H

#include "common_defines.h"

INTERPLIB_INTERNAL
void bernstein_interpolation_vector(double t, unsigned n, double INTERPLIB_ARRAY_ARG(out, restrict n));

INTERPLIB_INTERNAL
PyObject *bernstein_interpolation_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

INTERPLIB_INTERNAL
extern const char bernstein_interpolation_matrix_doc[];

#endif // BERNSTEIN_H
