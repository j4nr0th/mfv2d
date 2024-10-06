//
// Created by jan on 29.9.2024.

#ifndef LAGRANGE_H
#define LAGRANGE_H
#include "common_defines.h"
#include "error.h"


/**
* @brief Compute weights of lagrange interpolation at the nodes. The interpolation can be computed for any function on
* the same mesh by taking the inner product of the weight matrix with the function values.
*
* @param n_in Number of points where the interpolation will be needed.
* @param pos Array of nodes where interpolation will be computed
* @param n_nodes Number of nodes where the function is known.
* @param x Array of x-values of nodes where the function is known which must be monotonically increasing.
* @param weights Array which receives the weights for the interpolation.
* @param work Array used to store intermediate results.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_init(
    unsigned n_in,
    const double INTERPLIB_ARRAY_ARG(pos, static n_in),
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes * n_in),
    double INTERPLIB_ARRAY_ARG(work, restrict n_nodes)
);

/**
* @brief Evaluate first derivative of lagrange's interpolation.The interpolation can be computed for any function on
* the same mesh by taking the inner product of the weight matrix with the function values.
*
* @param n_in Number of points where the interpolation will be needed.
* @param pos Array of nodes where interpolation will be computed
* @param n_nodes Number of nodes where the function is known.
* @param x Array of x-values of nodes where the function is known which must be monotonically increasing.
* @param weights Array which receives the weights for the interpolation.
* @param work1 Array used to store intermediate results.
* @param work2 Array used to store intermediate results.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t dlagrange_interpolation(
    unsigned n_in,
    const double INTERPLIB_ARRAY_ARG(pos, static n_in),
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes * n_in),
    double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
    double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes)
);

#endif //LAGRANGE_H
