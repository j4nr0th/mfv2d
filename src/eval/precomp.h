//
// Created by jan on 20.2.2025.
//

#ifndef PRECOMP_H
#define PRECOMP_H

#include "../common.h"
#include "../common_defines.h"

#include "matrix.h"

#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

typedef enum
{
    MASS_0 = 0,
    MASS_1 = 1,
    MASS_2 = 2,
    MASS_0_I = 3,
    MASS_1_I = 4,
    MASS_2_I = 5,
    MASS_CNT,
} mass_mtx_indices_t;

INTERPLIB_INTERNAL
const char *mass_mtx_indices_str(mass_mtx_indices_t v);

typedef struct
{
    unsigned order;
    unsigned n_int;
    const double *nodes_int;
    const double *mass_nodal;
    const double *mass_edge_00;
    const double *mass_edge_01;
    const double *mass_edge_11;
    const double *mass_surf;
    PyArrayObject *arr_int_nodes;
    PyArrayObject *arr_node;
    PyArrayObject *arr_edge_00;
    PyArrayObject *arr_edge_01;
    PyArrayObject *arr_edge_11;
    PyArrayObject *arr_surf;
} basis_precomp_t;

typedef struct
{
    matrix_full_t mass_matrices[MASS_CNT];
} precompute_t;

/**
 * Create a `precompute_t` object with all matrices. This function expects Python `BasisCache` data as input, so it is
 * equivalent to the `Element.mass_matrix_node` and similar methods.
 *
 * Another notable point about this function is that it actually allocates and deallocates the memory it needs in a
 * FIFO order, with both output and internal memory. This means, a stack-based allocator can be used to make memory
 * allocation overheads trivially small.
 *
 * @param basis Pre-computed basis double products on the reference element. These are filled with contents of Python
 * `BasisCache` object.
 * @param x0 Bottom left corner's x coordinate.
 * @param x1 Bottom right corner's x coordinate.
 * @param x2 Top right corner's x coordinate.
 * @param x3 Top left corner's x coordinate.
 * @param y0 Bottom left corner's y coordinate.
 * @param y1 Bottom right corner's y coordinate.
 * @param y2 Top right corner's y coordinate.
 * @param y3 Top left corner's y coordinate.
 * @param out Pointer which receives the computed mass matrices.
 * @param allocator Allocator to be used in this function for output and intermediate buffers. Can be stack-based
 * @return Non-zero on success.
 */
INTERPLIB_INTERNAL
int precompute_create(const basis_precomp_t *basis, double x0, double x1, double x2, double x3, double y0, double y1,
                      double y2, double y3, precompute_t *out, allocator_callbacks *allocator);

/**
 * Turn Python serialized data into C-friendly form.
 *
 * @param serialized Serialized BasisCache tuple obtained by calling `BasisCache.c_serializaton`.
 * @param out Pointer to the struct which is filled out with arrays.
 * @return Non-zero on success.
 */
INTERPLIB_INTERNAL
int basis_precomp_create(PyObject *serialized, basis_precomp_t *out);

/**
 * Release the memory associated with the C-friendly precomputed data.
 *
 * @param this Basis precomputation to release.
 */
INTERPLIB_INTERNAL
void basis_precomp_destroy(basis_precomp_t *this);

#endif // PRECOMP_H
