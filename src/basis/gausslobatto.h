//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common.h"

MFV2D_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_gll_docstring[];

MFV2D_INTERNAL
int gauss_lobatto_nodes_weights(unsigned n, double tol, unsigned max_iter, double MFV2D_ARRAY_ARG(x, restrict n),
                                double MFV2D_ARRAY_ARG(w, restrict n));

typedef struct
{
    PyObject_HEAD;
    unsigned order;
    double *nodes;
    double *weights;
} integration_rule_1d_t;

MFV2D_INTERNAL
PyObject *compute_gauss_lobatto_integration_rule(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern PyTypeObject integration_rule_1d_type;

MFV2D_INTERNAL
PyObject *compute_legendre_polynomials(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_legendre_polynomials_docstring[];

#endif // GAUSSLOBATTO_H
