//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "../common_defines.h"
#include "../module.h"

typedef struct
{
    PyObject_HEAD size_t n_lines;
    geo_id_t lines[];
} surface_object_t;

MFV2D_INTERNAL
extern PyTypeObject surface_type_object;

MFV2D_INTERNAL
surface_object_t *surface_object_empty(size_t count);

MFV2D_INTERNAL
surface_object_t *surface_object_from_value(size_t count, geo_id_t ids[static count]);

#endif // SURFACEOBJECT_H
