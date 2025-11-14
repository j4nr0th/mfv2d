//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "../common/common_defines.h"
#include "../module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    geo_id_t lines[];
} surface_object_t;

MFV2D_INTERNAL
// extern PyTypeObject surface_type_object;
extern PyType_Spec surface_type_spec;

MFV2D_INTERNAL
surface_object_t *surface_object_from_value(PyTypeObject *surface_type_object, size_t count,
                                            geo_id_t ids[static count]);

#endif // SURFACEOBJECT_H
