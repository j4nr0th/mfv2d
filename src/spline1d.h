//
// Created by jan on 21.10.2024.
//

#ifndef SPLINE1D_H
#define SPLINE1D_H
#include <Python.h>

#include "common_defines.h"

typedef struct
{
    PyObject_HEAD vectorcallfunc call_spline;
    unsigned n_nodes;
    unsigned n_coefficients;
    double data[]; // size: (n_nodes - 1) * n_coefficients for spline1di, extra
                   // n_nodes entries for spline1d
} spline1d_t;

//  Piecewise polynomial with breakpoints for its coefficients
INTERPLIB_INTERNAL
extern PyTypeObject spline1d_type_object;

//  Piecewise polynomial with breakpoints for its coefficients being integers
//  (so no nodes are stored)
INTERPLIB_INTERNAL
extern PyTypeObject spline1di_type_object;

#endif // SPLINE1D_H
