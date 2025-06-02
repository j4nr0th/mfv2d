//
// Created by jan on 18.1.2025.
//

#ifndef MANIFOLD1D_H
#define MANIFOLD1D_H

#include "lineobject.h"
#include "manifold.h"

typedef struct
{
    PyObject_VAR_HEAD size_t n_points;
    size_t n_lines;
    line_t lines[];
} manifold1d_object_t;

MFV2D_INTERNAL
extern PyTypeObject manifold1d_type_object;

#endif // MANIFOLD1D_H
