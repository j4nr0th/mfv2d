//
// Created by jan on 29.9.2024.

#ifndef LAGRANGE_H
#define LAGRANGE_H
#include "common_defines.h"
#include "error.h"


/**
* @brief Compute weights of lagrange interpolation at the nodes. The interpolation can be computed for any function on
* the same mesh by taking the inner product of the weights with the function values.
*
* @param n_nodes Number of nodes.
* @param v Point where the interpolation is needed.
* @param x Array of x-values of nodes, which must be monotonically increasing.
* @param weights Array which receives the weights for the interpolation.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_init(
    unsigned n_nodes,
    double v,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes)
);

/**
* @brief Evaluate first derivative of lagrange's interpolation.The interpolation can be computed for any function on
* the same mesh by taking the inner product of the weights with the function values.
*
* @param n_nodes Number of nodes.
* @param v Point where the interpolation is needed.
* @param x Array of x-values of nodes, which must be monotonically increasing.
* @param weights Array which receives the weights for the interpolation.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t dlagrange_interpolation(
    unsigned n_nodes,
    double v,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes)
);

#endif //LAGRANGE_H
