#ifndef GAUSS_LOBATTO_H
#define GAUSS_LOBATTO_H
#include "../common/common.h"
#include "../common/error.h"

/**
 * Evaluates the Legendre polynomial of degree n and its derivative using Bonnet's recursion formula.
 * Stores the results in the provided output array.
 *
 * @param n The degree of the Legendre polynomial. Must be greater than or equal to 2.
 * @param x The point at which the Legendre polynomial is evaluated.
 * @param out A two-element array where the result is stored.
 *            out[0] receives the value of the Legendre polynomial of degree n-1.
 *            out[1] receives the value of the Legendre polynomial of degree n.
 */
MFV2D_INTERNAL
void legendre_eval_bonnet_two(unsigned n, double x, double MFV2D_ARRAY_ARG(out, 2));

/**
 * Computes the nodes and weights for Gauss-Lobatto quadrature using an iterative method.
 * The method calculates `n` nodes stored in `x` and their corresponding weights stored in `w`.
 * For nodes or weights that fail to converge within the specified tolerance and maximum iterations,
 * a non-convergence counter is returned.
 *
 * @param n The number of nodes to compute. Must be greater than or equal to 2.
 * @param tol The tolerance for convergence. Determines the acceptable level of numerical error.
 * @param max_iter The maximum number of iterations allowed for the iterative convergence process.
 * @param x An output array of size `n` where the computed Gauss-Lobatto nodes are stored.
 *          The first and last elements of the array are predefined as -1 and +1, respectively.
 * @param w An output array of size `n` where the computed Gauss-Lobatto weights are stored.
 *          The first and last elements of the array are predefined and set to specific values.
 * @return The number of nodes that failed to converge within the specified tolerance and maximum iterations.
 *         Returns 0 if all nodes converge successfully.
 */
MFV2D_INTERNAL
int gauss_lobatto_nodes_weights(const unsigned n, const double tol, const unsigned max_iter,
                                double MFV2D_ARRAY_ARG(x, restrict n), double MFV2D_ARRAY_ARG(w, restrict n));

MFV2D_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *mod, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

MFV2D_INTERNAL
extern const char compute_gll_docstring[];

typedef struct
{
    unsigned degree;
    double accuracy;
    double nodes_and_weights[];
} gll_entry_t;

typedef struct
{
    PyObject_HEAD;
    unsigned count;
    unsigned capacity;
    gll_entry_t **entries;
} gll_cache_t;

MFV2D_INTERNAL
extern PyType_Spec gll_cache_type_spec;

MFV2D_INTERNAL
const gll_entry_t *gll_cache_get_entry(const gll_cache_t *self, unsigned degree, double min_accuracy);

MFV2D_INTERNAL
mfv2d_result_t gll_cache_write_entry(gll_cache_t *self, unsigned degree, double accuracy,
                                     const double MFV2D_ARRAY_ARG(nodes, degree + 1),
                                     const double MFV2D_ARRAY_ARG(weights, degree + 1));

#endif // GAUSS_LOBATTO_H
