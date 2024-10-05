//
// Created by jan on 29.9.2024.
//

#include "lagrange.h"


static void lagrange_denominator(unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static restrict n_nodes),
    double INTERPLIB_ARRAY_ARG(out, restrict n_nodes)
)
{
    out[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n_nodes; ++j)
    {
        const double dif = x[0] - x[j];
        out[0] *= dif;
        out[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n_nodes; ++i)
    {
        for (unsigned j = i + 1; j < n_nodes; ++j)
        {
            const double dif = x[i] - x[j];
            out[i] *= +dif;
            out[j] *= -dif;
        }
    }
}

static void lagrange_numerator(unsigned n_nodes,
    double x,
    const double INTERPLIB_ARRAY_ARG(nodes, static restrict n_nodes),
    double INTERPLIB_ARRAY_ARG(out, restrict n_nodes)
)
{
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        out[i] = 1.0;
    }

    for (unsigned i = 0; i < n_nodes; ++i)
    {
        const double dif = x - nodes[i];
        for (unsigned j = 0; j < i; ++j)
        {
            out[j] *= +dif;
        }
        for (unsigned j = i + 1; j < n_nodes; ++j)
        {
            out[j] *= +dif;
        }
    }
}



INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_init(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(y, static n_nodes),
    double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes)
)
{
    lagrange_denominator(n_nodes, x, weights);
    for (unsigned i = 1; i < n_nodes; ++i)
    {
        if ASSERT(x[i] > x[i-1], "Nodes not monotonically increasing")
        {
            return INTERP_ERROR_NOT_INCREASING;
        }
    }

    // pre-divide the weights to save on future divisions
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        weights[i] = y[i] / weights[i];
    }

    return INTERP_SUCCESS;
}


INTERPLIB_INTERNAL
interp_error_t lagrange_interpolation_evaluate(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(nodes, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(weights, static n_nodes),
    double x,
    double* p_y,
    double INTERPLIB_ARRAY_ARG(work, restrict n_nodes)
)
{
    if ASSERT(x >= nodes[0] && x <= nodes[n_nodes - 1], "Value outside of nodes.")
    {
       return INTERP_ERROR_NOT_IN_DOMAIN;
    }
    lagrange_numerator(n_nodes, x, nodes, work);

    double sum = 0.0;
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        sum += work[i] * weights[i];
    }

    *p_y = sum;

    return INTERP_SUCCESS;
}


INTERPLIB_INTERNAL
interp_error_t dlagrange_interpolation(
    unsigned n_nodes,
    const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
    const double INTERPLIB_ARRAY_ARG(y, static n_nodes),
    double v,
    double* restrict p_out
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

    double out = 0.0;
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
        out += y[i] * acc;
    }

    *p_out = out;

    return INTERP_SUCCESS;
}



