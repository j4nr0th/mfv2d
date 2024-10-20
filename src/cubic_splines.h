//
// Created by jan on 19.10.2024.
//

#ifndef CUBIC_SPLINES_H
#define CUBIC_SPLINES_H
#include "common_defines.h"
#include "error.h"
#include "common.h"

typedef struct
{
    //  Number of points
    unsigned n;
    //  Nodal values of the function
    double* f;
    //  First derivatives
    double* df;
} interp_cubic_spline;

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

INTERPLIB_INTERNAL
interp_error_t interp_cubic_spline_init(
    unsigned n,
    const double INTERPLIB_ARRAY_ARG(nodes, static n),
    const cubic_spline_bc bc_left,
    const cubic_spline_bc bc_right,
    interp_cubic_spline** p_out,
    const jmtx_allocator_callbacks* allocator_out
);



#endif //CUBIC_SPLINES_H
