//
// Created by jan on 29.9.2024.
//

#include "lagrange.h"

INTERPLIB_INTERNAL
void lagrange_polynomial_denominators(unsigned n, const double INTERPLIB_ARRAY_ARG(nodes, restrict static n),
                                      double INTERPLIB_ARRAY_ARG(denominators, restrict n))
{
    denominators[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n; ++j)
    {
        const double dif = nodes[0] - nodes[j];
        denominators[0] *= dif;
        denominators[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n; ++i)
    {
        for (unsigned j = i + 1; j < n; ++j)
        {
            const double dif = nodes[i] - nodes[j];
            denominators[i] *= +dif;
            denominators[j] *= -dif;
        }
    }
}

INTERPLIB_INTERNAL
void lagrange_polynomial_coefficients(unsigned n, unsigned j,
                                      const double INTERPLIB_ARRAY_ARG(nodes, restrict static n),
                                      double INTERPLIB_ARRAY_ARG(coefficients, restrict n))
{
    coefficients[0] = 1.0;
    for (unsigned i = 0; i < j; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i + 1] = 0.0;
        for (unsigned k = i + 1; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
    for (unsigned i = j + 1; i < n; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i] = 0.0;
        for (unsigned k = i; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
}

INTERPLIB_INTERNAL
void lagrange_polynomial_values(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in), unsigned n_nodes,
                                const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                double INTERPLIB_ARRAY_ARG(work, restrict n_nodes))
{
    lagrange_polynomial_denominators(n_nodes, x, work);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work[i] = 1.0 / work[i];
    }

    //  Compute the numerator now
    for (unsigned k = 0; k < n_in; ++k)
    {
        double *const row = weights + n_nodes * k;
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
}

INTERPLIB_INTERNAL
void lagrange_polynomial_first_derivative(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in),
                                          unsigned n_nodes, const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                          double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                          /* cache for denominators (once per fn) */
                                          double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
                                          /* cache for differences (once per node) */
                                          double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes))
{
    // compute denominators
    lagrange_polynomial_denominators(n_nodes, x, work1);

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
                //  Loop split into three parts to enforce k != {i,
                //  j}
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
}

INTERPLIB_INTERNAL
interp_error_t lagrange_polynomial_second_derivative(unsigned n_in, const double INTERPLIB_ARRAY_ARG(pos, static n_in),
                                                     unsigned n_nodes,
                                                     const double INTERPLIB_ARRAY_ARG(x, static n_nodes),
                                                     double INTERPLIB_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                     double INTERPLIB_ARRAY_ARG(work1, restrict n_nodes),
                                                     double INTERPLIB_ARRAY_ARG(work2, restrict n_nodes))
{
    // compute denominators
    lagrange_polynomial_denominators(n_nodes, x, work1);

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

        for (unsigned i = 0; i < n_nodes; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                for (unsigned k = 0; k < j; ++k)
                {
                    double dlijkdx = 1.0;
                    //  Loop split into four parts to enforce l
                    //  != {i, j, k}
                    for (unsigned l = 0; l < k; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = k + 1; l < j; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = j + 1; l < i; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = i + 1; l < n_nodes; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    //  L_i^j and L_j^i have same numerators
                    weights[n_nodes * ipos + k] += 2 * dlijkdx;
                    weights[n_nodes * ipos + j] += 2 * dlijkdx;
                    weights[n_nodes * ipos + i] += 2 * dlijkdx;
                }
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
