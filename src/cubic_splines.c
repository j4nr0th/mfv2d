//
// Created by jan on 19.10.2024.
//

#include "cubic_splines.h"

#include <float.h>
#include <math.h>

static const double MATH_TOL = (double)FLT_EPSILON * 10.0;

INTERPLIB_INTERNAL
interp_error_t interp_cubic_spline_init(
    unsigned n, const double INTERPLIB_ARRAY_ARG(nodes, static n), const cubic_spline_bc bc_left,
    const cubic_spline_bc bc_right, interp_cubic_spline** p_out, const jmtx_allocator_callbacks* allocator
)
{
    //  TODO: handle special case where the BCs cause some pain with the first/last two rows.
    interp_cubic_spline* const this = allocate(allocator, sizeof(*this));
    if (!this)
    {
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    this->n = n;
    this->f = NULL;
    this->df = NULL;

    this->df = allocate(allocator, sizeof(*this->df) * n);
    if (!this->df)
    {
        deallocate(allocator, this);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    this->f = allocate(allocator, sizeof(*this->f) * n);
    if (!this->f)
    {
        deallocate(allocator, this->df);
        deallocate(allocator, this);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    double* restrict y = this->df;
    double* restrict c = this->f;

    double first_row[2] = {bc_left.k1 + 4.0 * bc_left.k2, 2.0 * bc_left.k2};
    double last_row[2] = {2.0 * bc_right.k2, bc_right.k1 + 4.0 * bc_right.k2};
    c[0] = first_row[1] / first_row[0];
    y[0] = 6.0 * (nodes[1] - nodes[0]) * bc_left.k2 + bc_left.v;
    for (unsigned i = 1; i < n - 1; ++i)
    {
        const double newb = 1.0 / (8.0 - 2.0 * c[i-1]);
        c[i] = 2.0 * newb;
        y[i] = (6 * (nodes[i + 1] - nodes[i - 1]) - 2.0 * y[i - 1]) * newb;
    }
    const double newb = 1.0 / (last_row[1] - last_row[0] * c[n-2]);
    y[n - 1] = (6 * (nodes[n - 1] - nodes[n - 2]) * bc_left.k2 + bc_right.v - 2.0 * y[n - 2]) * newb;
    c[n - 1] = nodes[n - 1];
    // Solve by back-substitution
    for (unsigned i = n; i > 0; --i)
    {
        y[i - 2] = y[i - 2] - y[i - 1] * c[i - 2];
        c[i - 2] = nodes[i - 2]; // Reuse the memory of c
    }

    *p_out = this;
    return INTERP_SUCCESS;
}
