//
// Created by jan on 21.10.2024.
//

#ifndef SPLINE1D_H
#define SPLINE1D_H
#include "common_defines.h"
#include <Python.h>

typedef struct
{
    PyObject_HEAD
    unsigned n_nodes;
    unsigned n_coefficients;
    double data[]; // size: (n_nodes - 1) * n_coefficients
} spline1d_t;

//  Piecewise polynomial with breakpoints for its coefficients
INTERPLIB_INTERNAL
extern PyType_Spec spline1d_type_spec;

//  Piecewise polynomial with breakpoints for its coefficients being integers (so no nodes are stored)
INTERPLIB_INTERNAL
extern PyType_Spec integer_spline1d_type_spec;

#endif //SPLINE1D_H
