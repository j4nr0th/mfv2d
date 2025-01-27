//
// Created by jan on 19.10.2024.
//

#ifndef CUBIC_SPLINES_H
#define CUBIC_SPLINES_H
#include "../common.h"
#include "../common_defines.h"
#include "../error.h"

/**
 * Structure used to define boundary conditions for a cubic spline.
 *
 * The boundary condition is defined as:
 *
 * $$
 *   k_1 \cdot \frac{df}{dx} + k_2 * \frac{d^2 f}{dx^2} = v
 * $$
 *
 */
typedef struct
{
    //  Coefficient of the first derivative in the equation.
    double k1;
    //  Coefficient of the second derivative in the equation.
    double k2;
    //  The weighted sum of the first and second derivative.
    double v;
} cubic_spline_bc;

/**
 * Create cubic spline interpolation based on nodal values.
 *
 * The function computes derivatives $\frac{d x}{d t} = k$, such that Hermite
 * polynomials can be used to interpolate the polynomial between the nodes
 * $x(i) = x[i]$. As such, an interpolation between `x[i]` and `x[i + 1]` can
 * be found as $x(t) = x_i * H_1(t) + x_{i+1} * H_2(t) + k_i * H_3(t) + k_{i+1}
 * * H_4(t)$, with $0 \leq t \leq 1$ where $H_1(t)$, $H_2(t)$, $H_3(t)$, and
 * $H_4(t)$ are the Hermite polynomials defined as:
 *
 * - $H_1(t) = 2 t^3 - 3 t^2 + 1$
 * - $H_2(t) = 3 t^2 - 2 t^3$
 * - $H_3(t) = t^3 - 2 t^2 + t$
 * - $H_4(t) = t^3 - t^2$
 *
 * @param n Number of nodes.
 * @param x Array of nodal values.
 * @param k Output array for derivatives at the nodes.
 * @param work Work array of the same size as `x` and `k`, used for
 * intermediate results.
 * @param bc_left Boundary condition to apply on the left boundary of the
 * spline.
 * @param bc_right Boundary condition to apply on the right boundary of the
 * spline.
 *
 * @returns `INTERP_SUCCESS` if successful, `INTERP_ERROR_BAD_SYSTEM` if
 * boundary conditions lead to a system that can't be solved easily.
 */
INTERPLIB_INTERNAL
interp_error_t interp_cubic_spline_init(unsigned n, const double INTERPLIB_ARRAY_ARG(x, restrict static n),
                                        double INTERPLIB_ARRAY_ARG(k, restrict n),
                                        double INTERPLIB_ARRAY_ARG(work, restrict n), const cubic_spline_bc bc_left,
                                        const cubic_spline_bc bc_right);

#endif // CUBIC_SPLINES_H
