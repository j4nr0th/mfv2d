#ifndef MFV2D_SYSTEM_OBJECTS_H
#define MFV2D_SYSTEM_OBJECTS_H
#include "sparse_system.h"

typedef struct
{
    PyObject_VAR_HEAD;
    system_t system;
    unsigned total_dense;
} system_object_t;

MFV2D_INTERNAL
// extern PyTypeObject system_object_type;
extern PyType_Spec system_object_spec;

static inline const system_t *system_object_as_system(const system_object_t *const this)
{
    return &this->system;
}

typedef struct
{
    PyObject_VAR_HEAD;
    system_object_t *parent;
    unsigned *offsets;
    double *values;
} dense_vector_object_t;

MFV2D_INTERNAL
// extern PyTypeObject dense_vector_object_type;
extern PyType_Spec dense_vector_object_type_spec;

static inline dense_vector_t dense_vector_object_as_dense_vector(const dense_vector_object_t *const this)
{
    return (dense_vector_t){
        .parent = system_object_as_system(this->parent), .offsets = this->offsets, .values = this->values};
}

typedef struct
{
    PyObject_VAR_HEAD;
    system_object_t *parent;
    svector_t *values;
} trace_vector_object_t;

MFV2D_INTERNAL
// extern PyTypeObject trace_vector_object_type;
extern PyType_Spec trace_vector_type_spec;

static inline trace_vector_t trace_vector_object_as_trace_vector(const trace_vector_object_t *const this)
{
    return (trace_vector_t){
        .parent = system_object_as_system(this->parent),
        .values = this->values,
    };
}

#endif // MFV2D_SYSTEM_OBJECTS_H
