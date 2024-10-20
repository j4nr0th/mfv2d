//
// Created by jan on 19.10.2024.
//

#include "cubic_splines.h"

#include <float.h>
#include <math.h>

static const double MATH_TOL = (double)FLT_EPSILON * 10.0;

INTERPLIB_INTERNAL
interp_error_t interp_cubic_spline_init(
    unsigned n,
    const double INTERPLIB_ARRAY_ARG(x, restrict static n),
    double INTERPLIB_ARRAY_ARG(k, restrict n),
    double INTERPLIB_ARRAY_ARG(work, restrict n),
    const cubic_spline_bc bc_left,
    const cubic_spline_bc bc_right
)
{
    //  TODO: handle special case where the BCs cause some pain with the first/last two rows.

    double first_row[2] = {bc_left.k1 + 4.0 * bc_left.k2, 2.0 * bc_left.k2};
    if ASSERT(fabs(first_row[0]) > MATH_TOL, "Left BC was bad.")
    {
        return INTERP_ERROR_BAD_SYSTEM;
    }
    double last_row[2] = {2.0 * bc_right.k2, bc_right.k1 + 4.0 * bc_right.k2};
    if ASSERT(fabs(last_row[1]) > MATH_TOL, "Right BC was bad.")
    {
        return INTERP_ERROR_BAD_SYSTEM;
    }
    work[0] = first_row[1] / first_row[0];
    k[0] = (6.0 * (x[1] - x[0]) * bc_left.k2 + bc_left.v) / first_row[0];
    for (unsigned i = 1; i < n - 1; ++i)
    {
        const double newb = 1.0 / (8.0 - 2.0 * work[i-1]);
        work[i] = 2.0 * newb;
        k[i] = (6 * (x[i + 1] - x[i - 1]) - 2.0 * k[i - 1]) * newb;
    }
    const double newb = 1.0 / (last_row[1] - last_row[0] * work[n-2]);
    k[n - 1] = (6 * (x[n - 1] - x[n - 2]) * bc_right.k2 + bc_right.v - last_row[0] * k[n - 2]) * newb;
    // Solve by back-substitution
    for (unsigned i = n; i > 1; --i)
    {
        k[i - 2] = k[i - 2] - k[i - 1] * work[i - 2];
    }
    return INTERP_SUCCESS;
}
