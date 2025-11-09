#ifndef ELEMENT_CACHE_H
#define ELEMENT_CACHE_H

#include "../algebra/matrix.h"
#include "../common/common.h"
#include "../fem_space/fem_space.h"
#include "basis.h"

typedef struct
{
    PyObject_HEAD;               // head of the type
    basis_1d_t *basis_xi;        // Basis functions in the 1st dimension
    basis_1d_t *basis_eta;       // Basis functions in the 2nd dimension
    fem_space_2d_t *fem_space;   // Fem space created from the basis
    quad_info_t corners;         // Array of 4 corners
    matrix_full_t mass_node;     // Cached can be empty
    matrix_full_t mass_edge;     // Cached can be empty
    matrix_full_t mass_surf;     // Cached can be empty
    matrix_full_t mass_node_inv; // Cached can be empty
    matrix_full_t mass_edge_inv; // Cached can be empty
    matrix_full_t mass_surf_inv; // Cached can be empty
} element_fem_space_2d_t;

MFV2D_INTERNAL
extern PyTypeObject element_fem_space_2d_type;

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_node(element_fem_space_2d_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_edge(element_fem_space_2d_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_surf(element_fem_space_2d_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_node_inv(element_fem_space_2d_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_edge_inv(element_fem_space_2d_t *cache);

MFV2D_INTERNAL
const matrix_full_t *element_mass_cache_get_surf_inv(element_fem_space_2d_t *cache);

#endif // ELEMENT_CACHE_H
