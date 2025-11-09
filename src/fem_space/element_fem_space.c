#include "../fem_space/element_fem_space.h"
#include "basis.h"

static void element_fem_space_2d_dealloc(element_fem_space_2d_t *this)
{
    Py_DECREF(this->basis_xi);
    Py_DECREF(this->basis_eta);
    deallocate(&SYSTEM_ALLOCATOR, this->fem_space);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_node.data);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_edge.data);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_surf.data);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_node_inv.data);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_edge_inv.data);
    deallocate(&SYSTEM_ALLOCATOR, this->mass_surf_inv.data);
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *element_fem_space_2d_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    basis_2d_t *basis;
    PyArrayObject *corners;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", (char *[3]){"basis", "corners", NULL}, &basis_2d_type,
                                     &basis, &PyArray_Type, &corners))
        return NULL;

    if (check_input_array(corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS,
                          "corners") < 0)
        return NULL;

    element_fem_space_2d_t *const self = (element_fem_space_2d_t *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->basis_xi = basis->basis_xi;
    self->basis_eta = basis->basis_eta;
    Py_INCREF(basis->basis_xi);
    Py_INCREF(basis->basis_eta);
    self->corners = *(quad_info_t *)PyArray_DATA(corners);

    const fem_space_1d_t space_1 = basis_1d_as_fem_space(basis->basis_xi);
    const fem_space_1d_t space_2 = basis_1d_as_fem_space(basis->basis_eta);

    const mfv2d_result_t res =
        fem_space_2d_create(&space_1, &space_2, &self->corners, &self->fem_space, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not create FEM space, reason %s.", mfv2d_result_str(res));
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

static PyObject *element_fem_space_2d_get_mass_node(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_node(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix node.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_mass_edge(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_edge(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix edge.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_mass_surf(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_surf(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix surface.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_basis_2d(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    return (PyObject *)create_basis_2d_object(&basis_2d_type, self->basis_xi, self->basis_eta);
}

static PyObject *element_fem_space_2d_get_basis_xi(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_xi);
    return (PyObject *)self->basis_xi;
}

static PyObject *element_fem_space_2d_get_basis_eta(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_eta);
    return (PyObject *)self->basis_eta;
}

static PyObject *element_fem_space_2d_get_corners(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t corners = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = 4, .cols = 2},
        .data = (double *)&self->corners,
    };
    return (PyObject *)matrix_full_to_array(&corners);
}

static PyObject *element_fem_space_2d_get_node_inv(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_node_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix node inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_edge_inv(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_edge_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix edge inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_surf_inv(element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_surf_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix surf inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_fem_space_2d_get_orders(const element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    return Py_BuildValue("II", self->basis_xi->order, self->basis_eta->order);
}

static PyObject *element_fem_space_2d_get_integration_orders(const element_fem_space_2d_t *self,
                                                             void *Py_UNUSED(closure))
{
    return Py_BuildValue("II", self->basis_xi->integration_rule->order, self->basis_eta->integration_rule->order);
}

static PyObject *element_fem_space_2d_get_order_1(const element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->basis_xi->order);
}

static PyObject *element_fem_space_2d_get_order_2(const element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->basis_eta->order);
}

static PyObject *element_fem_space_2d_get_jacobian(const element_fem_space_2d_t *self, void *Py_UNUSED(closure))
{
    const npy_intp dims[4] = {self->basis_eta->integration_rule->order + 1, self->basis_xi->integration_rule->order + 1,
                              2, 2};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(4, dims, NPY_DOUBLE);
    if (out == NULL)
        return NULL;
    double *const out_data = (double *)PyArray_DATA(out);
    for (npy_intp i = 0; i < dims[0]; ++i)
    {
        for (npy_intp j = 0; j < dims[1]; ++j)
        {
            out_data[4 * (i * dims[0] + j) + 0] = self->fem_space->jacobian[i * dims[0] + j].j00;
            out_data[4 * (i * dims[0] + j) + 1] = self->fem_space->jacobian[i * dims[0] + j].j01;
            out_data[4 * (i * dims[0] + j) + 2] = self->fem_space->jacobian[i * dims[0] + j].j10;
            out_data[4 * (i * dims[0] + j) + 3] = self->fem_space->jacobian[i * dims[0] + j].j11;
        }
    }

    return (PyObject *)out;
}

static PyObject *element_fem_space_2d_get_jacobian_determinant(const element_fem_space_2d_t *self,
                                                               void *Py_UNUSED(closuere))
{
    const npy_intp dims[2] = {self->basis_eta->integration_rule->order + 1,
                              self->basis_xi->integration_rule->order + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == NULL)
        return NULL;
    double *const out_data = (double *)PyArray_DATA(out);
    for (npy_intp i = 0; i < dims[0]; ++i)
    {
        for (npy_intp j = 0; j < dims[1]; ++j)
        {
            out_data[j + i * dims[0]] = self->fem_space->jacobian[j + i * dims[0]].det;
        }
    }

    return (PyObject *)out;
}

static PyGetSetDef element_fem_space_2d_getsets[] = {
    {
        .name = "mass_node",
        .get = (getter)element_fem_space_2d_get_mass_node,
        .doc = "array : Mass matrix for nodal basis.",
        .set = NULL,
    },
    {
        .name = "mass_edge",
        .get = (getter)element_fem_space_2d_get_mass_edge,
        .doc = "array : Mass matrix for edge basis.",
        .set = NULL,
    },
    {
        .name = "mass_surf",
        .get = (getter)element_fem_space_2d_get_mass_surf,
        .doc = "array : Mass matrix for surface basis.",
        .set = NULL,
    },
    {
        .name = "mass_node_inv",
        .get = (getter)element_fem_space_2d_get_node_inv,
        .doc = "array : Inverse mass matrix for nodal basis.",
        .set = NULL,
    },
    {
        .name = "mass_edge_inv",
        .get = (getter)element_fem_space_2d_get_edge_inv,
        .doc = "array : Inverse mass matrix for edge basis.",
        .set = NULL,
    },
    {
        .name = "mass_surf_inv",
        .get = (getter)element_fem_space_2d_get_surf_inv,
        .doc = "array : Inverse mass matrix for surface basis.",
        .set = NULL,
    },
    {
        .name = "basis_2d",
        .get = (getter)element_fem_space_2d_get_basis_2d,
        .doc = "Basis2D : Basis used for the element.",
        .set = NULL,
    },
    {
        .name = "basis_xi",
        .get = (getter)element_fem_space_2d_get_basis_xi,
        .doc = "Basis1D : Basis used for the first dimension.",
        .set = NULL,
    },
    {
        .name = "basis_eta",
        .get = (getter)element_fem_space_2d_get_basis_eta,
        .doc = "Basis1D : Basis used for the second dimension.",
        .set = NULL,
    },
    {
        .name = "corners",
        .get = (getter)element_fem_space_2d_get_corners,
        .doc = "array : Corners of the element.",
        .set = NULL,
    },
    {
        .name = "orders",
        .get = (getter)element_fem_space_2d_get_orders,
        .doc = "tuple[int, int] : Orders of the basis.",
        .set = NULL,
        .closure = NULL,
    },
    {
        .name = "integration_orders",
        .get = (getter)element_fem_space_2d_get_integration_orders,
        .doc = "tuple[int, int] : Orders of integration rules used by the basis.",
        .set = NULL,
        .closure = NULL,
    },
    {
        .name = "order_1",
        .get = (getter)element_fem_space_2d_get_order_1,
        .doc = "int : Order of the basis in the first dimension.",
        .set = NULL,
        .closure = NULL,
    },
    {
        .name = "order_2",
        .get = (getter)element_fem_space_2d_get_order_2,
        .doc = "int : Order of the basis in the second dimension.",
        .set = NULL,
        .closure = NULL,
    },
    {
        .name = "jacobian",
        .get = (void *)element_fem_space_2d_get_jacobian,
        .set = NULL,
        .doc = "(M, N, 2, 2) array : Jacobian components for the element at integration points.\n"
               "\n"
               "These are returned in the following order:\n"
               "\n"
               "- :math:`\\mathbf{J}_{0, 0} = \\frac{\\mathrm{d} x}{\\mathrm{d} \\xi}`\n"
               "- :math:`\\mathbf{J}_{0, 1} = \\frac{\\mathrm{d} y}{\\mathrm{d} \\xi}`\n"
               "- :math:`\\mathbf{J}_{1, 0} = \\frac{\\mathrm{d} x}{\\mathrm{d} \\eta}`\n"
               "- :math:`\\mathbf{J}_{1, 1} = \\frac{\\mathrm{d} y}{\\mathrm{d} \\eta}`\n",
        .closure = NULL,
    },
    {
        .name = "jacobian_determinant",
        .get = (void *)element_fem_space_2d_get_jacobian_determinant,
        .set = NULL,
        .doc = "(M, N) array : Determinant of the Jacobian at the integration points.",
        .closure = NULL,
    },
    {0}, // Sentilel
};

static PyObject *element_fem_space_2d_mass_from_order(element_fem_space_2d_t *this, PyObject *args, PyObject *kwargs)
{
    int i_order;
    int inverse = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", (char *[3]){"order", "inverse", NULL}, &i_order, &inverse))
        return NULL;

    if (i_order <= 0 || i_order > 3)
    {
        PyErr_Format(PyExc_ValueError, "Order must be between 1 and 3, got %d.", i_order);
        return NULL;
    }

    const matrix_full_t *out = NULL;
    switch (i_order)
    {
    case 1:
        out = inverse ? element_mass_cache_get_node_inv(this) : element_mass_cache_get_node(this);
        break;
    case 2:
        out = inverse ? element_mass_cache_get_edge_inv(this) : element_mass_cache_get_edge(this);
        break;
    case 3:
        out = inverse ? element_mass_cache_get_surf_inv(this) : element_mass_cache_get_surf(this);
        break;
    }
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

PyDoc_STRVAR(mass_from_order_docstr, "mass_from_order(order: UnknownFormOrder, inverse: bool=False) -> array\n"
                                     "Compute mass matrix for the given order.\n"
                                     "\n"
                                     "Parameters\n"
                                     "----------\n"
                                     "order : UnknownFormOrder\n"
                                     "    Order of the differential for to get the matrix from.\n"
                                     "\n"
                                     "inverse : bool, default: False\n"
                                     "    Should the matrix be inverted.\n"
                                     "\n"
                                     "Returns\n"
                                     "-------\n"
                                     "array\n"
                                     "    Mass matrix of the specified order (or inverse if specified).\n");

static PyMethodDef element_fem_space_2d_methods[] = {
    {
        "mass_from_order",
        (void *)element_fem_space_2d_mass_from_order,
        METH_VARARGS | METH_KEYWORDS,
        mass_from_order_docstr,
    },
    {0}, // Sentinel
};

MFV2D_INTERNAL
PyTypeObject element_fem_space_2d_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.ElementFemSpace2D",
    .tp_basicsize = sizeof(element_fem_space_2d_t),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = "Caches element mass matrices",
    .tp_new = element_fem_space_2d_new,
    .tp_dealloc = (destructor)element_fem_space_2d_dealloc,
    .tp_getset = element_fem_space_2d_getsets,
    .tp_methods = element_fem_space_2d_methods,
};

const matrix_full_t *element_mass_cache_get_node(element_fem_space_2d_t *cache)
{
    if (cache->mass_node.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_node(cache->fem_space, &cache->mass_node, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_node;
}

const matrix_full_t *element_mass_cache_get_edge(element_fem_space_2d_t *cache)
{
    if (cache->mass_edge.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_edge(cache->fem_space, &cache->mass_edge, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_edge;
}

const matrix_full_t *element_mass_cache_get_surf(element_fem_space_2d_t *cache)
{
    if (cache->mass_surf.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_surf(cache->fem_space, &cache->mass_surf, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_surf;
}

const matrix_full_t *element_mass_cache_get_node_inv(element_fem_space_2d_t *cache)
{
    if (cache->mass_node_inv.data == NULL)
    {
        const matrix_full_t *this = element_mass_cache_get_node(cache);

        if (matrix_full_invert(this, &cache->mass_node_inv, &SYSTEM_ALLOCATOR) != MFV2D_SUCCESS)
        {
            return NULL;
        }
    }

    return &cache->mass_node_inv;
}

const matrix_full_t *element_mass_cache_get_edge_inv(element_fem_space_2d_t *cache)
{
    if (cache->mass_edge_inv.data == NULL)
    {
        const matrix_full_t *this = element_mass_cache_get_edge(cache);

        if (matrix_full_invert(this, &cache->mass_edge_inv, &SYSTEM_ALLOCATOR) != MFV2D_SUCCESS)
        {
            return NULL;
        }
    }

    return &cache->mass_edge_inv;
}

const matrix_full_t *element_mass_cache_get_surf_inv(element_fem_space_2d_t *cache)
{
    if (cache->mass_surf_inv.data == NULL)
    {
        const matrix_full_t *this = element_mass_cache_get_surf(cache);

        if (matrix_full_invert(this, &cache->mass_surf_inv, &SYSTEM_ALLOCATOR) != MFV2D_SUCCESS)
        {
            return NULL;
        }
    }

    return &cache->mass_surf_inv;
}
