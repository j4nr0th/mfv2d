//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "../common_defines.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD surface_t value;
} surface_object_t;

INTERPLIB_INTERNAL
extern PyTypeObject surface_type_object;

INTERPLIB_INTERNAL
surface_object_t *pyvl_surface_from_value(geo_id_t bottom, geo_id_t right, geo_id_t top, geo_id_t left);

#endif // SURFACEOBJECT_H
