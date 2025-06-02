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

#endif // GAUSSLOBATTO_H
