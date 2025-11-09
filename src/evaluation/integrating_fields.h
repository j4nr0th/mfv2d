#ifndef INTEGRATING_FIELDS_H
#define INTEGRATING_FIELDS_H

#include "../common/common.h"
#include "../common/error.h"
#include "../fem_space/fem_space.h"
#include "forms.h"

enum
{
    INTEGRATING_FIELDS_MAX_COUNT = 16,
};

typedef enum
{
    FIELD_SPEC_UNKNOWN = 1,
    FIELD_SPEC_CALLABLE = 2,
    FIELD_SPEC_UNUSED = -1,
} field_spec_type_t;

typedef struct
{
    field_spec_type_t type;
    form_order_t form_order;
    unsigned index;
} field_spec_unknown_t;

typedef struct
{
    field_spec_type_t type;
    form_order_t form_order;
    PyObject *callable;
} field_spec_callable_t;

typedef union {
    struct
    {
        field_spec_type_t type;
        form_order_t form_order;
    };
    field_spec_unknown_t unknown;
    field_spec_callable_t callable;
} field_spec_t;

typedef struct
{
    unsigned n_fields;                                  // Number of vector fields provided.
    const double *fields[INTEGRATING_FIELDS_MAX_COUNT]; // Array of offsets for elements.
    void *buffer;
} field_values_t;

MFV2D_INTERNAL
void reconstruct_field_0_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict]);

MFV2D_INTERNAL
void reconstruct_field_1_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict]);

MFV2D_INTERNAL
void reconstruct_field_2_form(const fem_space_2d_t *space, const double degrees_of_freedom[restrict],
                              double output[restrict]);

MFV2D_INTERNAL
mfv2d_result_t compute_fields(const fem_space_2d_t *fem_space, const quad_info_t *quad, field_values_t *fields,
                              const unsigned n_fields, const field_spec_t field_specs[const static n_fields],
                              const allocator_callbacks *allocator, const element_form_spec_t *form_spec,
                              const double degrees_of_freedom[restrict]);

MFV2D_INTERNAL
PyObject *compute_integrating_fields(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_integrating_fields_docstring[];

#endif // INTEGRATING_FIELDS_H
