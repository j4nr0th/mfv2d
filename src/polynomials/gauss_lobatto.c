//
// Created by jan on 8/1/25.
//

#include "gauss_lobatto.h"

/**
 * Evaluates the Legendre polynomial of degree n and its derivative using Bonnet's recursion formula.
 * Stores the results in the provided output array.
 *
 * @param n The degree of the Legendre polynomial. Must be greater than or equal to 2.
 * @param x The point at which the Legendre polynomial is evaluated.
 * @param out A two-element array where the result is stored.
 *            `out[0]` receives the value of the Legendre polynomial with degree n-1.
 *            `out[1]` receives the value of the Legendre polynomial with degree n.
 */
MFV2D_INTERNAL
void legendre_eval_bonnet_two(const unsigned n, const double x, double MFV2D_ARRAY_ARG(out, 2))
{
    // n >= 2
    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = i - 1;
        const double new = (k1 * v2 - k2 * v1) / (double)i;
        v1 = v2;
        v2 = new;
    }
    out[0] = v1;
    out[1] = v2;
}

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
                                double MFV2D_ARRAY_ARG(x, restrict n), double MFV2D_ARRAY_ARG(w, restrict n))
{
    int non_converged = 0;
    // n >= 2
    x[0] = -1.0;
    x[n - 1] = +1.0;
    w[n - 1] = w[0] = 2.0 / (double)(n * (n - 1));
    const double kx_1 = 1.0 - 3.0 * (n - 2) / (8.0 * (n - 1) * (n - 1) * (n - 1));
    const double kx_2 = M_PI / (4.0 * (n - 1) + 1);
    for (unsigned i = 2; i < n; ++i)
    {
        double new_x = kx_1 * cos(kx_2 * (4 * i - 3));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]) / denominator;
            const double d2y = (2 * new_x * dy - (n - 1) * n * leg_poly[1]) / denominator;
            const double d3y = (4 * new_x * d2y - ((n - 1) * n - 2) * dy) / denominator;
            const double dx = 2 * dy * d2y / (2 * d2y * d2y - dy * d3y);
            new_x -= dx;
            error = fabs(dx);
        }
        // this is done like this to catch any NaNs
        non_converged += 1 - (error <= tol);
        x[n - i] = new_x;
        legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
        w[n - i] = 2.0 / (n * (n - 1) * leg_poly[1] * leg_poly[1]);
    }
    return non_converged;
}

static int ensure_gll_cache_and_state(PyObject *self, PyTypeObject *defining_class, gll_cache_t **p_cache,
                                      const mfv2d_module_state_t **p_state)
{
    const mfv2d_module_state_t *state;
    if (defining_class)
        state = PyType_GetModuleState(defining_class);
    else
        state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->type_gll_cache))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s instead.", state->type_gll_cache->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_cache = (gll_cache_t *)self;
    *p_state = state;
    return 0;
}

MFV2D_INTERNAL
const char compute_gll_docstring[] =
    "compute_gll(order: int, /, max_iter: int = 10, tol: float = 1e-15, cache: "
    "GLLCache | None = DEFAULT_GLL_CACHE) -> tuple[array, array]\n"
    "Compute Gauss-Legendre-Lobatto integration nodes and weights.\n"
    "\n"
    "If you are often re-using these, consider caching them.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "   Order of the scheme. The number of node-weight pairs is one more.\n"
    "max_iter : int, default: 10\n"
    "   Maximum number of iterations used to further refine the values.\n"
    "tol : float, default: 1e-15\n"
    "   Tolerance for stopping the refinement of the nodes.\n"
    "cache : GLLCache or None, default: DEFAULT_GLL_CACHE\n"
    "        The cache to use for computing the GLLNodes. If none is given, then the results\n"
    "        are not cached.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "   Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.\n"
    "array\n"
    "   Array of integration weights which correspond to the nodes.\n"
    "\n"
    "Examples\n"
    "--------\n"
    "Gauss-Legendre-Lobatto nodes computed using this function, along with\n"
    "the weights.\n"
    "\n"
    ".. jupyter-execute::\n"
    "\n"
    "    >>> import numpy as np\n"
    "    >>> from mfv2d._mfv2d import compute_gll\n"
    "    >>> from matplotlib import pyplot as plt\n"
    "    >>>\n"
    "    >>> n = 5\n"
    "    >>> nodes, weights = compute_gll(n)\n"
    "    >>>\n"
    "    >>> # Plot these\n"
    "    >>> plt.figure()\n"
    "    >>> plt.scatter(nodes, weights)\n"
    "    >>> plt.xlabel(\"$\\\\xi$\")\n"
    "    >>> plt.ylabel(\"$w$\")\n"
    "    >>> plt.grid()\n"
    "    >>> plt.show()\n"
    "\n"
    "Since these are computed in an iterative way, giving a tolerance\n"
    "which is too strict or not allowing for sufficient iterations\n"
    "might cause an exception to be raised to do failure to converge.\n"
    "\n";

MFV2D_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    int order;
    int max_iter = 10;
    double tol = 1e-15;
    gll_cache_t *cache = (gll_cache_t *)state->cache_gll;

    if (parse_arguments_check(
            (argument_t[]){
                {.type = ARG_TYPE_INT, .p_val = &order},
                {.type = ARG_TYPE_INT, .kwname = "max_iter", .p_val = &max_iter, .optional = 1},
                {.type = ARG_TYPE_DOUBLE, .kwname = "tol", .p_val = &tol, .optional = 1},
                {.type = ARG_TYPE_PYTHON, .kwname = "cache", .p_val = &cache, .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (Py_IsNone((PyObject *)cache))
    {
        cache = NULL;
    }
    else if (!PyObject_TypeCheck(cache, state->type_gll_cache))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s instead.", state->type_gll_cache->tp_name,
                     Py_TYPE(cache)->tp_name);
        return NULL;
    }

    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be positive, but was given as %i.", order);
        return NULL;
    }
    if (max_iter < 0)
    {
        PyErr_Format(PyExc_ValueError, "Number of maximum iterations must be positive, but was given as %i.", max_iter);
        return NULL;
    }
    if (tol < 0)
    {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%g", tol);
        PyErr_Format(PyExc_ValueError, "Tolerance must be positive %s", buffer);
        return NULL;
    }

    const npy_intp array_size = order + 1;
    PyArrayObject *const nodes = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!nodes)
    {
        return NULL;
    }
    PyArrayObject *const weights = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!weights)
    {
        Py_DECREF(nodes);
        return NULL;
    }

    const gll_entry_t *e = cache ? gll_cache_get_entry(cache, order, tol) : NULL;
    double *const p_x = PyArray_DATA(nodes);
    double *const p_w = PyArray_DATA(weights);

    if (e)
    {
        // We already cached this, so just copy the data over
        memcpy(p_x, e->nodes_and_weights, sizeof(double) * (order + 1));
        memcpy(p_w, e->nodes_and_weights + order + 1, sizeof(double) * (order + 1));
    }
    else
    {
        // Not cached, compute it
        if (order != 0)
        {
            const int non_converged = gauss_lobatto_nodes_weights(order + 1, tol, max_iter, p_x, p_w);
            if (non_converged != 0)
            {
                PyErr_Format(PyExc_RuntimeWarning,
                             "A total of %i nodes were non-converged. Consider changing"
                             " the tolerance or increase the number of iterations.",
                             non_converged);
            }
        }
        else
        {
            // Corner case
            p_x[0] = 0.0;
            p_w[0] = 2.0;
        }

        // If we have a cache, store the result!
        if (cache)
        {
            const mfv2d_result_t res = gll_cache_write_entry(cache, order, tol, p_x, p_w);
            if (res != MFV2D_SUCCESS)
            {
                // Not critical, but we should print a warning
                PyErr_WarnEx(PyExc_RuntimeWarning, "Failed to cache the result.", 1);
            }
        }
    }

    return PyTuple_Pack(2, nodes, weights);
}

const gll_entry_t *gll_cache_get_entry(const gll_cache_t *const self, const unsigned degree, const double min_accuracy)
{
    // Check if the cache is empty
    if (self->count == 0)
        return NULL;
    // Since entries are sorted, first do a range check
    if (degree < self->entries[0]->degree)
        return NULL;
    if (degree > self->entries[self->count - 1]->degree)
        return NULL;

    // Just do linear search; it will be fast enough, since the cache should not have more than like 16 entries
    for (unsigned i = 0; i < self->count; ++i)
        if (self->entries[i]->degree == degree)
        {
            if (self->entries[i]->accuracy >= min_accuracy)
                return self->entries[i];
            return NULL;
        }

    return NULL;
}

mfv2d_result_t gll_cache_write_entry(gll_cache_t *const self, const unsigned degree, const double accuracy,
                                     const double MFV2D_ARRAY_ARG(nodes, static const degree + 1),
                                     const double MFV2D_ARRAY_ARG(weights, static const degree + 1))
{
    // Check if it is already present
    gll_entry_t *entry = (gll_entry_t *)gll_cache_get_entry(self, degree, accuracy);
    if (entry)
    {
        if (entry->accuracy < accuracy)
        {
            // Update data if accuracy is better
            memcpy(entry->nodes_and_weights, nodes, sizeof(double) * (degree + 1));
            memcpy(entry->nodes_and_weights + degree + 1, weights, sizeof(double) * (degree + 1));
            entry->accuracy = accuracy;
        }
        return MFV2D_SUCCESS;
    }

    // Check if the cache has space
    if (self->count == self->capacity)
    {
        const unsigned new_capacity = self->capacity ? self->capacity * 2 : 8;
        gll_entry_t **const new_ptr = PyMem_Realloc(self->entries, sizeof(gll_entry_t *) * new_capacity);
        if (!new_ptr)
            return MFV2D_FAILED_ALLOC;
        memset(new_ptr + self->capacity, 0, sizeof(gll_entry_t *) * (new_capacity - self->capacity));
        self->entries = new_ptr;
        self->capacity = new_capacity;
    }

    // Prepare the entry to insert
    entry = PyMem_Malloc(sizeof(gll_entry_t) + 2 * (degree + 1) * sizeof(double));
    if (!entry)
        return MFV2D_FAILED_ALLOC;

    entry->degree = degree;
    entry->accuracy = accuracy;
    memcpy(entry->nodes_and_weights, nodes, sizeof(double) * (degree + 1));
    memcpy(entry->nodes_and_weights + degree + 1, weights, sizeof(double) * (degree + 1));

    // Move the entries out of the way
    unsigned i;
    for (i = self->count; i > 0; --i)
    {
        if (self->entries[i - 1]->degree < degree)
            break;
        self->entries[i] = self->entries[i - 1];
    }
    // Append to the end
    self->entries[i] = entry;
    self->count += 1;
    return MFV2D_SUCCESS;
}

static PyObject *gll_cache_new(PyTypeObject *type, const PyObject *args, const PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "GLLCache takes no parameters in its constructor.");
        return NULL;
    }

    gll_cache_t *const self = (gll_cache_t *)type->tp_alloc(type, 0);
    if (!self)
    {
        return NULL;
    }
    self->count = 0;
    self->capacity = 0;
    self->entries = NULL;
    return (PyObject *)self;
}

static void gll_cache_dealloc(gll_cache_t *self)
{
    PyObject_GC_UnTrack(self);
    for (unsigned i = 0; i < self->count; ++i)
    {
        PyMem_Free(self->entries[i]);
        self->entries[i] = NULL;
    }
    PyMem_Free(self->entries);
    self->entries = NULL;
    self->count = 0;
    self->capacity = 0;
    Py_TYPE(self)->tp_free((PyObject *)self);
}
PyDoc_STRVAR(gll_cache_clear_docstring, "clear() -> None\n"
                                        "Remove all entries from the cache.\n");

static PyObject *gll_cache_clear(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                 const Py_ssize_t nargs, PyObject *kwnames)
{
    (void)args;

    const mfv2d_module_state_t *state;
    gll_cache_t *this;
    if (ensure_gll_cache_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "GLLCache.clear() takes no parameters.");
        return NULL;
    }

    for (unsigned i = 0; i < this->count; ++i)
    {
        PyMem_Free(this->entries[i]);
        this->entries[i] = NULL;
    }

    this->count = 0;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(gll_cache_usage_docstring,
             "usage() -> tuple[tuple[int, float], ...]\n"
             "Return info about the current usage of the cache.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "tuple of (int, float)\n"
             "    Entries in the cache, characterized by the number of nodes and the tolerance\n"
             "    they were computed at.\n");

static PyObject *gll_cache_usage(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                 const Py_ssize_t nargs, PyObject *kwnames)
{
    (void)args;
    const mfv2d_module_state_t *state;
    gll_cache_t *this;
    if (ensure_gll_cache_and_state(self, defining_class, &this, &state) < 0)
        return NULL;
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "GLLCache.usage() takes no parameters.");
        return NULL;
    }

    PyTupleObject *const result = (PyTupleObject *)PyTuple_New(this->count);
    if (!result)
        return NULL;

    for (unsigned i = 0; i < this->count; ++i)
    {
        PyObject *const entry = Py_BuildValue("Id", this->entries[i]->degree, this->entries[i]->accuracy);
        if (!entry)
        {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, entry);
    }

    return (PyObject *)result;
}

PyDoc_STRVAR(gll_cache_type_docstring, "Cache for Gauss-Legendre-Lobatto integration nodes and weights.");

PyType_Spec gll_cache_type_spec = {
    .name = "mfv2d._mfv2d.GLLCache",
    .basicsize = sizeof(gll_cache_t),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {.slot = Py_tp_new, .pfunc = gll_cache_new},
            {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
            {.slot = Py_tp_dealloc, .pfunc = gll_cache_dealloc},
            {
                .slot = Py_tp_methods,
                .pfunc =
                    (PyMethodDef[]){
                        {
                            .ml_name = "clear",
                            .ml_meth = (void *)gll_cache_clear,
                            .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                            .ml_doc = gll_cache_clear_docstring,
                        },
                        {
                            .ml_name = "usage",
                            .ml_meth = (void *)gll_cache_usage,
                            .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                            .ml_doc = gll_cache_usage_docstring,
                        },
                        {}, // sentinel
                    },
            },
            {.slot = Py_tp_doc, .pfunc = (void *)gll_cache_type_docstring},
            {}, // sentinel
        },
};
