//
// Created by jan on 21.10.2024.
//

#ifndef POLYNOMIAL1D_H
#define POLYNOMIAL1D_H
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

#endif //POLYNOMIAL1D_H
