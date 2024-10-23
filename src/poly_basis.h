//
// Created by jan on 21.10.2024.
//

#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H
#include "common_defines.h"
#include <Python.h>

typedef struct
{
    PyObject_HEAD
    unsigned n;
    double k[];
} polynomial_basis_t;

INTERPLIB_INTERNAL
extern PyType_Spec poly_basis_type_spec;

#endif //BASIS_FUNCTIONS_H
