//
// Created by jan on 27.6.2025.
//

#include "element_equation.h"

// Implementation

static PyObject *element_dof_equation_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "ElementDofEquation.__new__ takes no keyword arguments.");
        return NULL;
    }

    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 2)
    {
        PyErr_SetString(PyExc_TypeError, "At least one (int, float) pair must be provided.");
        return NULL;
    }

    // Parse first arg (element) and the rest as *pairs
    const int element = PyLong_AsInt(PyTuple_GET_ITEM(args, 0));
    if (PyErr_Occurred())
        return NULL;

    if (element < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Element index must be non-negative.");
        return NULL;
    }

    // Allocate raw C arrays
    uint32_t *const dofs_data = allocate(&SYSTEM_ALLOCATOR, (nargs - 1) * sizeof(uint32_t));
    double *const coeffs_data = allocate(&SYSTEM_ALLOCATOR, (nargs - 1) * sizeof(double));
    if (!dofs_data || !coeffs_data)
    {
        deallocate(&SYSTEM_ALLOCATOR, dofs_data);
        deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
        return NULL;
    }
    // Fill arrays
    for (unsigned i = 1; i < nargs; ++i)
    {
        PyObject *const pair = PyTuple_GET_ITEM(args, i);

        if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2)
        {
            PyErr_Format(PyExc_TypeError, "Each pair must be a tuple (int, float), got %R", pair);
            free(dofs_data);
            free(coeffs_data);
            deallocate(&SYSTEM_ALLOCATOR, dofs_data);
            deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
            return NULL;
        }
        PyObject *dof_obj = PyTuple_GET_ITEM(pair, 0);
        PyObject *coeff_obj = PyTuple_GET_ITEM(pair, 1);
        const long dof = PyLong_AsLong(dof_obj);
        const double coeff = PyFloat_AsDouble(coeff_obj);
        if (PyErr_Occurred())
        {
            deallocate(&SYSTEM_ALLOCATOR, dofs_data);
            deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
            return NULL;
        }
        if (dof < 0)
        {
            PyErr_SetString(PyExc_ValueError, "DoF index must be non-negative.");
            deallocate(&SYSTEM_ALLOCATOR, dofs_data);
            deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
            return NULL;
        }
        if (!isfinite(coeff))
        {
            PyErr_SetString(PyExc_ValueError, "Coefficient must be finite.");
            deallocate(&SYSTEM_ALLOCATOR, dofs_data);
            deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
            return NULL;
        }
        dofs_data[i - 1] = (uint32_t)dof;
        coeffs_data[i - 1] = coeff;
    }
    element_dof_equation_t *const self = (element_dof_equation_t *)type->tp_alloc(type, 0);
    if (self == NULL)
    {
        deallocate(&SYSTEM_ALLOCATOR, dofs_data);
        deallocate(&SYSTEM_ALLOCATOR, coeffs_data);
        return NULL;
    }

    // Assign to object
    self->element = element;
    self->n_pairs = nargs - 1;
    self->dofs = dofs_data;
    self->coeffs = coeffs_data;
    return (PyObject *)self;
}

static void element_dof_equation_dealloc(element_dof_equation_t *self)
{
    deallocate(&SYSTEM_ALLOCATOR, self->dofs);
    deallocate(&SYSTEM_ALLOCATOR, self->coeffs);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *element_dof_equation_pairs(const element_dof_equation_t *self, PyObject *Py_UNUSED(ignored))
{
    // Return a generator of (int, float) pairs
    PyObject *tuple = PyTuple_New(self->n_pairs);
    if (!tuple)
        return NULL;

    for (Py_ssize_t i = 0; i < self->n_pairs; ++i)
    {
        PyObject *pair = PyTuple_Pack(2, PyLong_FromUnsignedLong(self->dofs[i]), PyFloat_FromDouble(self->coeffs[i]));
        if (!pair)
        {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, pair);
    }
    return PyObject_GetIter(tuple);
}

static Py_ssize_t element_dof_equation_len(const element_dof_equation_t *self)
{
    return (Py_ssize_t)self->n_pairs;
}

static PyObject *element_dof_equation_get_element(const element_dof_equation_t *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->element);
}

static PyObject *element_dof_equation_get_dofs(const element_dof_equation_t *self, void *Py_UNUSED(closure))
{
    // Return a new NumPy array (copy) from the raw C array
    const npy_intp dims = self->n_pairs;
    PyObject *arr = PyArray_SimpleNew(1, &dims, NPY_UINT32);
    if (!arr)
        return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), self->dofs, self->n_pairs * sizeof(uint32_t));
    return arr;
}

static PyObject *element_dof_equation_get_coeffs(const element_dof_equation_t *self, void *Py_UNUSED(closure))
{
    // Return a new NumPy array (copy) from the raw C array
    const npy_intp dims = self->n_pairs;
    PyObject *arr = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    if (!arr)
        return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), self->coeffs, self->n_pairs * sizeof(double));
    return arr;
}

PyDoc_STRVAR(element_dof_equation_pairs_docstr, "pairs() -> Iterator[tuple[int, float]]\n"
                                                "Get iterator over pairs.\n"
                                                "\n"
                                                "Returns\n"
                                                "-------\n"
                                                "Iterator of (int, float)\n"
                                                "    Iterator over all pairs of DoF index and coefficient pairs\n"
                                                "    for this equation.\n");

static PyMethodDef element_dof_equation_methods[] = {
    {"pairs", (PyCFunction)element_dof_equation_pairs, METH_NOARGS, element_dof_equation_pairs_docstr}, {NULL}};

static PyGetSetDef element_dof_equation_getset[] = {
    {"element", (getter)element_dof_equation_get_element, NULL, "int : Index of the element.", NULL},
    {"dofs", (getter)element_dof_equation_get_dofs, NULL,
     "NDArray[uint32] : Indices of degrees of freedom in the pairs.", NULL},
    {"coeffs", (getter)element_dof_equation_get_coeffs, NULL, "NDArray[double] : Coefficients of degrees of freedom.",
     NULL},
    {NULL}};

PyDoc_STRVAR(element_dof_equation_docstr, "ElementDofEquation(element: int, *pairs: tuple[int, float])\n"
                                          "Describes element DoFs involved in the equation and their coefficients.\n"
                                          "\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "element : int\n"
                                          "    Index of the element the degrees of freedom belong to.\n"
                                          "\n"
                                          "*pairs : (int, float)\n"
                                          "    Pairs of indices of degrees of freedom with their coefficient.\n"
                                          "    At least one must be specified.\n"

);

static PySequenceMethods element_dof_equation_as_sequence = {.sq_length = (lenfunc)element_dof_equation_len};

PyTypeObject element_dof_equation_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.ElementDofEquation",
    .tp_basicsize = sizeof(element_dof_equation_t),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)element_dof_equation_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = element_dof_equation_docstr,
    .tp_methods = element_dof_equation_methods,
    .tp_getset = element_dof_equation_getset,
    .tp_new = element_dof_equation_new,
    .tp_as_sequence = &element_dof_equation_as_sequence,
};
