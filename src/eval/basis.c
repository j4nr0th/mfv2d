#include "basis.h"
#include "../basis/gausslobatto.h"
#include "../basis/lagrange.h"

static mfv2d_result_t compute_nodal_and_edge_values(integration_rule_1d_t *rule, int order, double **nodal_vals,
                                                    double **edge_vals, double **p_roots)
{
    double *roots = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (order + 1));
    if (!roots)
    {
        return MFV2D_FAILED_ALLOC;
    }
    double *weights = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (order + 1) * (rule->order + 1));
    if (!weights)
    {
        PyMem_RawFree(roots);
        return MFV2D_FAILED_ALLOC;
    }
    const int non_converged = gauss_lobatto_nodes_weights(order + 1, 1e-15, 10, roots, weights);
    if (non_converged)
    {
        PyMem_RawFree(roots);
        PyMem_RawFree(weights);
        return MFV2D_NOT_CONVERGED;
    }

    double *const work = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (rule->order + 1));
    if (!work)
    {
        deallocate(&SYSTEM_ALLOCATOR, roots);
        deallocate(&SYSTEM_ALLOCATOR, weights);
        return MFV2D_FAILED_ALLOC;
    }

    lagrange_polynomial_values(rule->order + 1, rule->nodes, order + 1, roots, weights, work);

    *nodal_vals = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (order + 1) * (rule->order + 1));
    if (!*nodal_vals)
    {
        deallocate(&SYSTEM_ALLOCATOR, weights);
        deallocate(&SYSTEM_ALLOCATOR, work);
        deallocate(&SYSTEM_ALLOCATOR, roots);
        return MFV2D_FAILED_ALLOC;
    }
    // Transpose the values
    for (unsigned i = 0; i < order + 1; ++i)
    {
        for (unsigned j = 0; j < rule->order + 1; ++j)
        {
            (*nodal_vals)[i * (rule->order + 1) + j] = weights[j * (order + 1) + i];
        }
    }
    double *work2 = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (rule->order + 1));
    if (!work2)
    {
        deallocate(&SYSTEM_ALLOCATOR, *nodal_vals);
        deallocate(&SYSTEM_ALLOCATOR, weights);
        deallocate(&SYSTEM_ALLOCATOR, work);
        deallocate(&SYSTEM_ALLOCATOR, roots);
        return MFV2D_FAILED_ALLOC;
    }

    lagrange_polynomial_first_derivative(rule->order + 1, rule->nodes, order + 1, roots, weights, work, work2);

    *edge_vals = allocate(&SYSTEM_ALLOCATOR, sizeof(double) * (order + 1) * (rule->order + 1));
    if (!*edge_vals)
    {
        deallocate(&SYSTEM_ALLOCATOR, *nodal_vals);
        deallocate(&SYSTEM_ALLOCATOR, work2);
        deallocate(&SYSTEM_ALLOCATOR, weights);
        deallocate(&SYSTEM_ALLOCATOR, work);
        deallocate(&SYSTEM_ALLOCATOR, roots);
        return MFV2D_FAILED_ALLOC;
    }
    // Transpose and negative cumsum in one
    for (unsigned i = 0; i < rule->order + 1; ++i)
    {
        double v = 0;
        for (unsigned j = 0; j < order; ++j)
        {
            v -= weights[i * (order + 1) + j];
            (*edge_vals)[j * (rule->order + 1) + i] = v;
        }
    }

    *p_roots = roots;

    deallocate(&SYSTEM_ALLOCATOR, weights);
    deallocate(&SYSTEM_ALLOCATOR, work2);
    deallocate(&SYSTEM_ALLOCATOR, work);
    return MFV2D_SUCCESS;
}
static PyObject *basis_1d_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    integration_rule_1d_t *rule;
    int order;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", (char *const[3]){"order", "rule", NULL}, &order,
                                     &integration_rule_1d_type, &rule))
        return NULL;

    if (order <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must greater than zero, got %d.", order);
        return NULL;
    }

    double *nodal_vals, *edge_vals, *root_vals;
    mfv2d_result_t computed = 0;
    Py_BEGIN_ALLOW_THREADS;
    computed = compute_nodal_and_edge_values(rule, order, &nodal_vals, &edge_vals, &root_vals);
    Py_END_ALLOW_THREADS;
    if (computed != MFV2D_SUCCESS)
    {
        if (computed == MFV2D_NOT_CONVERGED)
            PyErr_Format(PyExc_RuntimeError, "Could not compute GLL nodes.");

        return NULL;
    }

    basis_1d_t *self = (basis_1d_t *)type->tp_alloc(type, 0);
    self->order = order;
    self->roots = root_vals;
    self->nodal_basis = nodal_vals;
    self->edge_basis = edge_vals;
    self->integration_rule = rule;
    Py_INCREF(rule);

    return (PyObject *)self;
}

static void basis_1d_dealloc(basis_1d_t *self)
{
    Py_DECREF(self->integration_rule);
    deallocate(&SYSTEM_ALLOCATOR, self->nodal_basis);
    deallocate(&SYSTEM_ALLOCATOR, self->edge_basis);
    deallocate(&SYSTEM_ALLOCATOR, self->roots);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *basis_1d_repr(const basis_1d_t *self)
{
    char buffer[128];
    (void)snprintf(buffer, sizeof(buffer), "Basis1D(order=%u)", self->order);
    return PyUnicode_FromString(buffer);
}

static PyObject *basis_1d_get_order(const basis_1d_t *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->order);
}

static PyObject *basis_1d_get_nodal(const basis_1d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp n[2] = {self->order + 1, self->integration_rule->order + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(2, n, NPY_DOUBLE, self->nodal_basis);
    if (out)
    {
        if (PyArray_SetBaseObject(out, (PyObject *)self) < 0)
        {
            Py_DECREF(out);
            return NULL;
        }
        Py_INCREF(self);
    }
    return (PyObject *)out;
}

static PyObject *basis_1d_get_edge(const basis_1d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp n[2] = {self->order, self->integration_rule->order + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(2, n, NPY_DOUBLE, self->edge_basis);
    if (out)
    {
        if (PyArray_SetBaseObject(out, (PyObject *)self) < 0)
        {
            Py_DECREF(out);
            return NULL;
        }
        Py_INCREF(self);
    }
    return (PyObject *)out;
}

static PyObject *basis_1d_get_roots(const basis_1d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp n[1] = {self->order + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(1, n, NPY_DOUBLE, self->roots);
    if (out)
    {
        if (PyArray_SetBaseObject(out, (PyObject *)self) < 0)
        {
            Py_DECREF(out);
            return NULL;
        }
        Py_INCREF(self);
    }
    return (PyObject *)out;
}

static PyObject *basis_1d_get_integration_rule(const basis_1d_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->integration_rule);
    return (PyObject *)self->integration_rule;
}

static PyGetSetDef basis_1d_getsets[] = {{.name = "order",
                                          .get = (getter)basis_1d_get_order,
                                          .set = NULL,
                                          .doc = "int : Order of the basis.",
                                          .closure = NULL},
                                         {.name = "node",
                                          .get = (getter)basis_1d_get_nodal,
                                          .set = NULL,
                                          .doc = "array : Nodal basis values.",
                                          .closure = NULL},
                                         {.name = "edge",
                                          .get = (getter)basis_1d_get_edge,
                                          .set = NULL,
                                          .doc = "array : Edge basis values.",
                                          .closure = NULL},
                                         {.name = "rule",
                                          .get = (getter)basis_1d_get_integration_rule,
                                          .set = NULL,
                                          .doc = "IntegrationRule1D : integration rule used",
                                          .closure = NULL},
                                         {
                                             .name = "roots",
                                             .get = (getter)basis_1d_get_roots,
                                             .set = NULL,
                                             .doc = "array : Roots of the nodal basis.",
                                         },
                                         {NULL}};

PyDoc_STRVAR(basis_1d_doc, "Basis1D(order: int, rule: IntegrationRule1D)\n"
                           "One-dimensional basis functions collection used for FEM space creation.\n"
                           "\n"
                           "Parameters\n"
                           "----------\n"
                           "order : int\n"
                           "    Order of basis used.\n"
                           "\n"
                           "rule : IntegrationRule1D\n"
                           "    Integration rule for basis creation.\n"
                           "Examples\n"
                           "--------\n"
                           "An example of how these basis might look for a 3-rd order\n"
                           "element is shown bellow.\n"
                           "\n"
                           ".. jupyter-execute::\n"
                           "\n"
                           "    >>> from matplotlib import pyplot\n"
                           "    >>> from mfv2d._mfv2d import Basis1D, IntegrationRule1D\n"
                           "    >>> \n"
                           "    >>> #Make a high order rule to make it easy to visualize\n"
                           "    >>> rule = IntegrationRule1D(order=31)\n"
                           "    >>> basis = Basis1D(3, rule)\n"
                           "\n"
                           "Now the nodal basis can be plotted:\n"
                           "\n"
                           ".. jupyter-execute::\n"
                           "\n"
                           "    >>> plt.figure()\n"
                           "    >>> for i in range(basis.order + 1):\n"
                           "    ...     plt.plot(basis.rule.nodes, basis.node[i, ...], label=f\"$b_{{{i}}}$\")\n"
                           "    >>> plt.grid()\n"
                           "    >>> plt.legend()\n"
                           "    >>> plt.xlabel(\"$\\\\xi$\")\n"
                           "    >>> plt.title(\"Nodal basis\")\n"
                           "    >>> plt.show()\n"
                           "\n"
                           "Edge basis can also be shown:\n"
                           "\n"
                           ".. jupyter-execute::\n"
                           "\n"
                           "    >>> plt.figure()\n"
                           "    >>> for i in range(basis.order):\n"
                           "    ...     plt.plot(basis.rule.nodes, basis.edge[i, ...], label=f\"$e_{{{i}}}$\")\n"
                           "    >>> plt.grid()\n"
                           "    >>> plt.legend()\n"
                           "    >>> plt.xlabel(\"$\\\\xi$\")\n"
                           "    >>> plt.title(\"Edge basis\")\n"
                           "    >>> plt.show()\n"
                           "\n");

PyTypeObject basis_1d_type = {
    .tp_new = basis_1d_new,
    .tp_dealloc = (destructor)basis_1d_dealloc,
    .tp_repr = (reprfunc)basis_1d_repr,
    .tp_getset = basis_1d_getsets,
    .tp_name = "mfv2d._mfv2d.Basis1D",
    .tp_basicsize = sizeof(basis_1d_t),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = basis_1d_doc,
    .tp_itemsize = 0,
};

MFV2D_INTERNAL
basis_2d_t *create_basis_2d_object(PyTypeObject *type, basis_1d_t *basis_xi, basis_1d_t *basis_eta)
{
    basis_2d_t *const self = (basis_2d_t *)type->tp_alloc(type, 0);
    if (!self)
    {
        return NULL;
    }

    Py_INCREF(basis_xi);
    Py_INCREF(basis_eta);
    self->basis_xi = basis_xi;
    self->basis_eta = basis_eta;
    return self;
}
// __new__ method
static PyObject *basis_2d_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    basis_1d_t *basis_xi = NULL, *basis_eta = NULL;
    static char *kwlist[] = {"basis_xi", "basis_eta", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!", kwlist, &basis_1d_type, &basis_xi, &basis_1d_type, &basis_eta))
        return NULL;

    basis_2d_t *const self = create_basis_2d_object(type, basis_xi, basis_eta);

    return (PyObject *)self;
}

// __repr__ method
static PyObject *basis_2d_repr(const basis_2d_t *self)
{
    return PyUnicode_FromFormat("Basis2D(basis_xi=%R, basis_eta=%R)", self->basis_xi, self->basis_eta);
}

static PyObject *basis_2d_get_basis_xi(const basis_2d_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_xi);
    return (PyObject *)self->basis_xi;
}
static PyObject *basis_2d_get_basis_eta(const basis_2d_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_eta);
    return (PyObject *)self->basis_eta;
}

static void basis_2d_dealloc(basis_2d_t *self)
{
    Py_XDECREF(self->basis_xi);
    Py_XDECREF(self->basis_eta);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *basis_2d_get_orders(const basis_2d_t *self, void *Py_UNUSED(closure))
{
    return Py_BuildValue("II", self->basis_xi->order, self->basis_eta->order);
}

static PyObject *basis_2d_get_integration_orders(const basis_2d_t *self, void *Py_UNUSED(closure))
{
    return Py_BuildValue("II", self->basis_xi->integration_rule->order, self->basis_eta->integration_rule->order);
}

static PyGetSetDef Basis2D_getset[] = {
    {"basis_xi", (getter)basis_2d_get_basis_xi, NULL, "Basis1D : Basis used for the Xi direction.", NULL},
    {"basis_eta", (getter)basis_2d_get_basis_eta, NULL, "Basis1D : Basis used for the Eta direction.", NULL},
    {"orders", (getter)basis_2d_get_orders, NULL, "(int, int) : Order of the basis."},
    {"integration_orders", (getter)basis_2d_get_integration_orders, NULL,
     "(int, int) : Order of the integration rules."},
    {NULL} // Sentinel
};

PyDoc_STRVAR(basis_2d_type_docstring,
             "Basis2D(basis_xi: Basis1D, basis_eta: Basis1D)\n"
             "Two dimensional basis resulting from a tensor product of one dimensional basis.\n");

// Type definition
PyTypeObject basis_2d_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.Basis2D",
    .tp_basicsize = sizeof(basis_2d_t),
    .tp_dealloc = (destructor)basis_2d_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = basis_2d_new,
    .tp_repr = (reprfunc)basis_2d_repr,
    .tp_getset = Basis2D_getset,
    .tp_doc = basis_2d_type_docstring,
};
