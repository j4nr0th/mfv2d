//
// Created by jan on 18.1.2025.
//

#include "manifold.h"

PyDoc_STRVAR(manifold_type_docstr, "A manifold of a finite number of dimensions.");

INTERPLIB_INTERNAL
PyTypeObject manifold_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.Manifold",
    .tp_basicsize = sizeof(PyObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = manifold_type_docstr,
};
