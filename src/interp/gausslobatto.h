//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common.h"

INTERPLIB_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

INTERPLIB_INTERNAL
extern const char compute_gll_docstring[];

#endif // GAUSSLOBATTO_H
