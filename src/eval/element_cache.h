#ifndef ELEMENT_CACHE_H
#define ELEMENT_CACHE_H

#include "../common.h"
#include "basis.h"
#include "fem_space.h"
#include "matrix.h"

typedef struct
{
    PyObject_HEAD;             // head of the type
    basis_2d_t *basis;         // Basis functions
    fem_space_2d_t *fem_space; // Fem space created from the basis
    quad_info_t corners;       // Array of 4 corners
    matrix_full_t mass_node;   // Cached can be empty
    matrix_full_t mass_edge;   // Cached can be empty
    matrix_full_t mass_surf;   // Cached can be empty
} element_mass_matrix_cache_t;

MFV2D_INTERNAL
extern PyTypeObject element_mass_matrix_cache_type;

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_node(element_mass_matrix_cache_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_edge(element_mass_matrix_cache_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_surf(element_mass_matrix_cache_t *cache);

#endif // ELEMENT_CACHE_H
