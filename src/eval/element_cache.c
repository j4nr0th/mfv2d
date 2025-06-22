#include "element_cache.h"
#include "basis.h"

static void element_mass_matrix_cache_dealloc(element_mass_matrix_cache_t *this)
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

static PyObject *element_mass_matrix_cache_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    basis_2d_t *basis;
    PyArrayObject *corners;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", (char *[3]){"basis", "corners", NULL}, &basis_2d_type,
                                     &basis, &PyArray_Type, &corners))
        return NULL;

    if (check_input_array(corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS,
                          "corners") < 0)
        return NULL;

    element_mass_matrix_cache_t *const self = (element_mass_matrix_cache_t *)type->tp_alloc(type, 0);
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

static PyObject *element_mass_matrix_cache_get_mass_node(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_node(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix node.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_mass_matrix_cache_get_mass_edge(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_edge(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix edge.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_mass_matrix_cache_get_mass_surf(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_surf(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix surface.");
        return NULL;
    }

    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_mass_matrix_cache_get_basis_2d(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    return (PyObject *)create_basis_2d_object(&basis_2d_type, self->basis_xi, self->basis_eta);
}

static PyObject *element_mass_matrix_cache_get_basis_xi(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_xi);
    return (PyObject *)self->basis_xi;
}

static PyObject *element_mass_matrix_cache_get_basis_eta(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->basis_eta);
    return (PyObject *)self->basis_eta;
}

static PyObject *element_mass_matrix_cache_get_corners(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t corners = {
        .base = {.type = MATRIX_TYPE_FULL, .rows = 4, .cols = 2},
        .data = (double *)&self->corners,
    };
    return (PyObject *)matrix_full_to_array(&corners);
}

static PyObject *element_mass_matrix_cache_get_node_inv(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_node_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix node inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_mass_matrix_cache_get_edge_inv(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_edge_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix edge inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyObject *element_mass_matrix_cache_get_surf_inv(element_mass_matrix_cache_t *self, void *Py_UNUSED(closure))
{
    const matrix_full_t *out = element_mass_cache_get_surf_inv(self);
    if (out == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not compute mass matrix surf inverse.");
        return NULL;
    }
    return (PyObject *)matrix_full_to_array(out);
}

static PyGetSetDef element_mass_matrix_cache_getsets[] = {
    {.name = "mass_node",
     .get = (getter)element_mass_matrix_cache_get_mass_node,
     .doc = "array : Mass matrix for nodal basis.",
     .set = NULL},
    {.name = "mass_edge",
     .get = (getter)element_mass_matrix_cache_get_mass_edge,
     .doc = "array : Mass matrix for edge basis.",
     .set = NULL},
    {.name = "mass_surf",
     .get = (getter)element_mass_matrix_cache_get_mass_surf,
     .doc = "array : Mass matrix for surface basis.",
     .set = NULL},
    {.name = "mass_node_inv",
     .get = (getter)element_mass_matrix_cache_get_node_inv,
     .doc = "array : Inverse mass matrix for nodal basis.",
     .set = NULL},
    {.name = "mass_edge_inv",
     .get = (getter)element_mass_matrix_cache_get_edge_inv,
     .doc = "array : Inverse mass matrix for edge basis.",
     .set = NULL},
    {.name = "mass_surf_inv",
     .get = (getter)element_mass_matrix_cache_get_surf_inv,
     .doc = "array : Inverse mass matrix for surface basis.",
     .set = NULL},
    {
        .name = "basis_2d",
        .get = (getter)element_mass_matrix_cache_get_basis_2d,
        .doc = "Basis2D : Basis used for the element.",
        .set = NULL,
    },
    {
        .name = "basis_xi",
        .get = (getter)element_mass_matrix_cache_get_basis_xi,
        .doc = "Basis1D : Basis used for the first dimension.",
        .set = NULL,
    },
    {
        .name = "basis_eta",
        .get = (getter)element_mass_matrix_cache_get_basis_eta,
        .doc = "Basis1D : Basis used for the second dimension.",
        .set = NULL,
    },
    {
        .name = "corners",
        .get = (getter)element_mass_matrix_cache_get_corners,
        .doc = "array : Corners of the element.",
        .set = NULL,
    },

    {0}, // Sentilel
};

static PyObject *element_mass_matrix_cache_mass_from_order(element_mass_matrix_cache_t *this, PyObject *args,
                                                           PyObject *kwargs)
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

static PyMethodDef element_mass_matrix_cache_methods[] = {
    {"mass_from_order", (PyCFunction)element_mass_matrix_cache_mass_from_order, METH_VARARGS | METH_KEYWORDS,
     mass_from_order_docstr},
    {0}, // Sentinel
};

MFV2D_INTERNAL
PyTypeObject element_mass_matrix_cache_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.ElementMassMatrixCache",
    .tp_basicsize = sizeof(element_mass_matrix_cache_t),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = "Caches element mass matrices",
    .tp_new = element_mass_matrix_cache_new,
    .tp_dealloc = (destructor)element_mass_matrix_cache_dealloc,
    .tp_getset = element_mass_matrix_cache_getsets,
    .tp_methods = element_mass_matrix_cache_methods,
};

const matrix_full_t *element_mass_cache_get_node(element_mass_matrix_cache_t *cache)
{
    if (cache->mass_node.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_node(cache->fem_space, &cache->mass_node, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_node;
}

const matrix_full_t *element_mass_cache_get_edge(element_mass_matrix_cache_t *cache)
{
    if (cache->mass_edge.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_edge(cache->fem_space, &cache->mass_edge, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_edge;
}

const matrix_full_t *element_mass_cache_get_surf(element_mass_matrix_cache_t *cache)
{
    if (cache->mass_surf.data == NULL)
    {
        const mfv2d_result_t res = compute_mass_matrix_surf(cache->fem_space, &cache->mass_surf, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
            return NULL;
    }

    return &cache->mass_surf;
}

const matrix_full_t *element_mass_cache_get_node_inv(element_mass_matrix_cache_t *cache)
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

const matrix_full_t *element_mass_cache_get_edge_inv(element_mass_matrix_cache_t *cache)
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

const matrix_full_t *element_mass_cache_get_surf_inv(element_mass_matrix_cache_t *cache)
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
