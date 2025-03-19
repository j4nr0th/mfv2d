//
// Created by jan on 19.3.2025.
//

#ifndef LIL_MATRIX_H
#define LIL_MATRIX_H

#include "svector.h"

typedef struct
{
    uint64_t rows, cols;
    svector_t *row_data;
} lil_matrix_t;

typedef struct
{
    PyObject_HEAD;
    uint64_t rows, cols;
    svector_t row_data[];
} lil_mat_object_t;

extern PyTypeObject lil_mat_type_object;

#endif // LIL_MATRIX_H
