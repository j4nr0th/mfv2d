#ifndef INTEGRATING_FIELDS_H
#define INTEGRATING_FIELDS_H

#include "../common.h"
#include "../error.h"
#include "fem_space.h"
#include "forms.h"

enum
{
    INTEGRATING_FIELDS_MAX_COUNT = 16,
};

typedef struct
{
    unsigned n_fields;                                  // Number of vector fields provided.
    const double *fields[INTEGRATING_FIELDS_MAX_COUNT]; // Array of offsets for elements.
    void *buffer;
} field_information_t;

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
mfv2d_result_t compute_fields(const fem_space_2d_t *fem_space, const quad_info_t *quad, field_information_t *fields,
                              unsigned n_fields, const form_order_t field_orders[static n_fields],
                              const allocator_callbacks *allocator, unsigned n_unknowns,
                              const form_order_t unknown_orders[static n_unknowns],
                              const double degrees_of_freedom[restrict], PyObject *field_data);

MFV2D_INTERNAL
PyObject *compute_integrating_fields(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds);

MFV2D_INTERNAL
extern const char compute_integrating_fields_docstring[];

#endif // INTEGRATING_FIELDS_H
