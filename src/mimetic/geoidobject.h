//
// Created by jan on 23.11.2024.
//

#ifndef GEOIDOBJECT_H
#define GEOIDOBJECT_H

#include "../common.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD geo_id_t id;
} geo_id_object_t;

INTERPLIB_INTERNAL
extern PyTypeObject geo_id_type_object;

INTERPLIB_INTERNAL
geo_id_object_t *geo_id_object_from_value(geo_id_t id);

INTERPLIB_INTERNAL
int geo_id_from_object(PyObject *o, geo_id_t *p_out);

static inline int geo_id_compare(const geo_id_t id1, const geo_id_t id2)
{
    return id1.index == id2.index && id1.reverse == id2.reverse;
}

#endif // GEOIDOBJECT_H
