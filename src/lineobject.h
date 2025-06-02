//
// Created by jan on 23.11.2024.
//

#ifndef LINEOBJECT_H
#define LINEOBJECT_H

#include "common.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD line_t value;
} line_object_t;

MFV2D_INTERNAL
extern PyTypeObject line_type_object;

MFV2D_INTERNAL
line_object_t *line_from_indices(geo_id_t begin, geo_id_t end);

#endif // LINEOBJECT_H
