//
// Created by jan on 23.11.2024.
//

#ifndef GEOIDOBJECT_H
#define GEOIDOBJECT_H

#include "../common/common.h"
#include "../module.h"

typedef struct
{
    PyObject_HEAD geo_id_t id;
} geo_id_object_t;

MFV2D_INTERNAL
// extern PyTypeObject geo_id_type_object;
extern PyType_Spec geo_id_type_spec;

MFV2D_INTERNAL
geo_id_object_t *geo_id_object_from_value(PyTypeObject *geo_id_type_object, geo_id_t id);

/**
 * @brief Try to convert PyObject into a geo_id_t value.
 *
 * Conversion will first check if the object was of type geo_id_object_t, which means the value is directly copied.
 * If the type does not match, it tries to convert using PyLong_AsLong and convert from the integer value. If that
 * conversion fails, it will set the Python exception state and function will return -1. If it succeeded instead,
 * 0 is returned.
 *
 * @param geoid_type Type object for the GeoID type.
 * @param o Object that should be converted.
 * @param p_out Pointer which receives the converted value.
 * @return 0 if successful, -1 if conversion failed.
 */
MFV2D_INTERNAL
int geo_id_from_object(PyTypeObject *geoid_type, PyObject *o, geo_id_t *p_out);

static inline int geo_id_compare(const geo_id_t id1, const geo_id_t id2)
{
    return id1.index == id2.index && id1.reverse == id2.reverse;
}

static inline int geo_id_unpack(const geo_id_t id)
{
    const int v = (int)(id.index + 1);
    return id.reverse ? -v : +v;
}

static inline geo_id_t geo_id_pack(const int v)
{
    if (v < 0)
    {
        return (geo_id_t){.reverse = 1, .index = -(v + 1)};
    }
    if (v > 0)
    {
        return (geo_id_t){.reverse = 0, .index = v - 1};
    }
    return (geo_id_t){.reverse = 0, .index = GEO_ID_INVALID};
}

#endif // GEOIDOBJECT_H
