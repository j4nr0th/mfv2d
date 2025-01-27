//
// Created by jan on 24.11.2024.
//

#ifndef MANIFOLD2D_H
#define MANIFOLD2D_H

#include "lineobject.h"
#include "manifold.h"

typedef struct
{
    PyObject_HEAD size_t n_points;
    size_t n_lines;
    line_t *lines;
    size_t n_surfaces;
    size_t *surf_counts;
    geo_id_t *surf_lines;
} manifold2d_object_t;

INTERPLIB_INTERNAL
extern PyTypeObject manifold2d_type_object;

#endif // MANIFOLD2D_H
