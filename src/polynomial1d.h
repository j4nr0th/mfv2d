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
extern PyTypeObject polynomial1d_type_object;

#endif //POLYNOMIAL1D_H
