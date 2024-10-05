//
// Created by jan on 29.9.2024.

#ifndef LAGRANGE_H
#define LAGRANGE_H
#include "common_defines.h"
#include "error.h"


/**
* @brief Compute weights of lagrange interpolation at the nodes.
*
* @param n_nodes Number of nodes.
* @param x Array of x-values of nodes, which must be monotonically increasing.
* @param y Array of y-values of nodes.
* @param weights Array which receives the weights for the interpolation.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_init(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(y, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes)
);

/**
* @brief Compute the interpolation at a specific point.
*
* @param n_nodes Number of interpolation nodes.
* @param nodes Interpolation nodes
* @param weights Interpolation weights computed prior.
* @param x Value where to evaluate the interpolation. Must be between the nodes.
* @param p_y Pointer which receives the output of the interpolation.
* @param work Array used by function for intermediate results. Should have space for at least `this->n_nodes` elements.
*/
INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_evaluate(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(nodes, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(weights, static n_nodes),
    double x,
    double* p_y,
    double INTERPLIB_ARRAY_ARG(work, restrict n_nodes)
    );

/**
* @brief Evaluate first derivative of lagrange's interpolation.
*
* @param n_nodes Number of nodes.
* @param x Array of x-values of nodes, which must be monotonically increasing.
* @param y Array of y-values of nodes.
* @param v Where the interpolation should be evaluated at.
* @param p_out Pointer which receives the output.
*
* @return `INTERP_SUCCESS` on success, `INTERP_ERROR_NOT_INCREASING` if `x[i + 1] > x[i]` does not hold for all `i`.
*/
INTERPLIB_INTERNAL
interp_error_t dlagrange_interpolation(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(y, static n_nodes),
    double v,
    double* restrict p_out
);

#endif //LAGRANGE_H
