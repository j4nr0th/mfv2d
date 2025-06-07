#ifndef ELEMENT_SYSTEM_H
#define ELEMENT_SYSTEM_H

#include "../common.h"

MFV2D_INTERNAL
PyObject *compute_element_matrix(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
PyObject *compute_element_vector(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
PyObject *compute_element_projector(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_element_projector_docstr[];

MFV2D_INTERNAL
PyObject *compute_element_mass_matrix(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_element_mass_matrix_docstr[];

#endif // ELEMENT_SYSTEM_H
