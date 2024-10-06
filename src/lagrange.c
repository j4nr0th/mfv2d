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
    unsigned n_in,
    const double INTERPLIB_ARRAY_ARG(pos, static n_in),
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes * n_in),
    /* cache for denominators (once per fn) */
    double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
    /* cache for differences (once per node) */
    double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes)
)
{
    for (unsigned i = 0; i < n_in; ++i)
    {
        if ASSERT(x[0] <= pos[i] && pos[i] <= x[n_nodes - 1], "Point not in domain")
        {
            return INTERP_ERROR_NOT_IN_DOMAIN;
        }
    }

    for (unsigned i = 1; i < n_nodes; ++i)
    {
        if ASSERT(x[i] > x[i - 1], "Nodes not monotonically increasing.")
        {
            return INTERP_ERROR_NOT_INCREASING;
        }
    }

    // compute denominators

    work1[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n_nodes; ++j)
    {
        const double dif = x[0] - x[j];
        work1[0] *= dif;
        work1[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n_nodes; ++i)
    {
        for (unsigned j = i + 1; j < n_nodes; ++j)
        {
            const double dif = x[i] - x[j];
            work1[i] *= +dif;
            work1[j] *= -dif;
        }
    }

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work1[i] = 1.0 / work1[i];
    }

    //  Now loop per node
    for (unsigned ipos = 0; ipos < n_in; ++ipos)
    {
        const double v = pos[ipos];
        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Compute the differences
            work2[j] = v - x[j];
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i, j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = i + 1; k < n_nodes; ++k)
                {
                    dlijdx *= work2[k];
                }
                //  L_i^j and L_j^i have same numerators
                weights[n_nodes * ipos + j] += dlijdx;
                weights[n_nodes * ipos + i] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] *= work1[j];
        }
    }

    return INTERP_SUCCESS;
}



