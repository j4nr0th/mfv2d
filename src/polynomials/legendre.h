#ifndef LEGENDRE_H
#define LEGENDRE_H
#include "../common/common.h"

MFV2D_INTERNAL
PyObject *compute_legendre_polynomials(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_legendre_polynomials_docstring[];
#endif // LEGENDRE_H
