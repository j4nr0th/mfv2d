//
// Created by jan on 29.9.2024.
//

#include "lagrange.h"


INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_init(
    unsigned n_in,
    const double INTERPLIB_ARRAY_ARG(pos, static n_in),
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes * n_in),
    double INTERPLIB_ARRAY_ARG(work, restrict n_nodes)
)
{
    for (unsigned i = 0; i < n_in; ++i)
    {
        if ASSERT(x[0] <= pos[i] && pos[i] <= x[n_nodes - 1], "Point out of bounds")
        {
            return INTERP_ERROR_NOT_IN_DOMAIN;
        }
    }

    for (unsigned i = 1; i < n_nodes; ++i)
    {
        if ASSERT(x[i] > x[i-1], "Nodes not monotonically increasing")
        {
            return INTERP_ERROR_NOT_INCREASING;
        }
    }


    work[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n_nodes; ++j)
    {
        const double dif = x[0] - x[j];
        work[0] *= dif;
        work[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n_nodes; ++i)
    {
        for (unsigned j = i + 1; j < n_nodes; ++j)
        {
            const double dif = x[i] - x[j];
            work[i] *= +dif;
            work[j] *= -dif;
        }
    }

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work[i] = 1.0 / work[i];
    }

    //  Compute the numerator now
    for (unsigned k = 0; k < n_in; ++k)
    {
        double* const row = weights + n_nodes * k;
        //  First loop can be used to initialize the row
        {
            const double dif = pos[k] - x[0];
            row[0] = 1.0;
            for (unsigned j = 1; j < n_nodes; ++j)
            {
                row[j] = +dif;
            }
        }
        for (unsigned i = 1; i < n_nodes; ++i)
        {
            const double dif = pos[k] - x[i];
            for (unsigned j = 0; j < i; ++j)
            {
                row[j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_nodes; ++j)
            {
                row[j] *= +dif;
            }
        }
        //  Multiply by 1/denominator
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            row[i] *= work[i];
        }
    }

    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
interp_error_t dlagrange_interpolation(
    unsigned n_nodes,
    double v,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes)
)
{
    if ASSERT(v >= x[0] && v <= x[n_nodes - 1], "Point not in domain")
    {
        return INTERP_ERROR_NOT_IN_DOMAIN;
    }
    for (unsigned i = 1; i < n_nodes; ++i)
    {
        if ASSERT(x[i] > x[i - 1], "Nodes not monotonically increasing.")
        {
            return INTERP_ERROR_NOT_INCREASING;
        }
    }

    for (unsigned i = 0; i < n_nodes; ++i)
    {
        double acc = 0.0;
        for (unsigned j = 0; j < n_nodes; ++j)
        {
            if (j == i)
            {
                continue;
            }
            double acc2 = 1.0 / (x[i] - x[j]);
            for (unsigned k = 0; k < n_nodes; ++k)
            {
                if (k == j || k == i)
                {
                    continue;
                }
                acc2 *= (v - x[k]) / (x[i] - x[k]);
            }
            acc += acc2;
        }
        weights[i] = acc;
    }

    return INTERP_SUCCESS;
}



