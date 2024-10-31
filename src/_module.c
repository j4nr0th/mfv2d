//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#include "common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Internal headers
#include "basis1d.h"
#include "cubic_splines.h"
#include "lagrange.h"
#include "polynomial1d.h"
#include "spline1d.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyObject *interp_lagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res = lagrange_polynomial_values(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

static PyObject *interp_dlagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(2 * n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res =
        lagrange_polynomial_first_derivative(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work, work + n_nodes);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

static PyObject *interp_d2lagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(2 * n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res =
        lagrange_polynomial_second_derivative(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work, work + n_nodes);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

static PyObject *interp_hermite_coefficients(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyObject *bc1;
    PyObject *bc2;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x, &PyTuple_Type, &bc1, &PyTuple_Type, &bc2))
    {
        return NULL;
    }
    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");

    cubic_spline_bc bc_left, bc_right;
    if (!PyArg_ParseTuple(bc1, "ddd", &bc_left.k1, &bc_left.k2, &bc_left.v))
    {
        return NULL;
    }
    if (!PyArg_ParseTuple(bc2, "ddd", &bc_right.k1, &bc_right.k2, &bc_right.v))
    {
        return NULL;
    }

    npy_intp n_pts = PyArray_SIZE(x);

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, &n_pts, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(n_pts * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_error_t interp_res = interp_cubic_spline_init(n_pts, p_x, p_out, work, bc_left, bc_right);
    PyMem_Free(work);
    if ASSERT (interp_res == INTERP_SUCCESS, "Interpolation failed")
    {
        Py_DECREF(out);
        PyErr_Format(PyExc_RuntimeError, "Could not compute Hermite coefficients, error code %s (%s)",
                     interp_error_str(interp_res), interp_error_msg(interp_res));
        return NULL;
    }

    return (PyObject *)PyArray_Return(out);
}

PyDoc_STRVAR(interp_lagrange_doc, "lagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");
PyDoc_STRVAR(interp_dlagrange_doc, "dlagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");
PyDoc_STRVAR(interp_d2lagrange_doc, "d2lagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");
PyDoc_STRVAR(interp_hermite_doc, "hermite(x: np.ndarray, bc1: tuple[float, float, float], bc2: "
                                 "tuple[float, float, float]) -> np.ndarray");

static PyObject *test_method(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;

    return PyUnicode_FromString("Test successful!\n");
}

static PyMethodDef module_methods[] = {
    {"test", test_method, METH_NOARGS, "Test method that only returns a string."},
    {"lagrange1d", interp_lagrange, METH_VARARGS, interp_lagrange_doc},
    {"dlagrange1d", interp_dlagrange, METH_VARARGS, interp_dlagrange_doc},
    {"d2lagrange1d", interp_d2lagrange, METH_VARARGS, interp_d2lagrange_doc},
    {"hermite", interp_hermite_coefficients, METH_VARARGS, interp_hermite_doc},
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "_interp",
                             .m_doc = "Internal C-extension implementing interpolation functions",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__interp(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        goto failed;
    }

    if (PyType_Ready(&basis1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&polynomial1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&spline1d_type_object) < 0)
    {
        goto failed;
    }
    if (PyType_Ready(&spline1di_type_object) < 0)
    {
        goto failed;
    }

    PyObject *mod = PyModule_Create(&module);
    if (!mod)
    {
        goto failed;
    }

    int res = PyModule_AddObjectRef(mod, "Basis1D", (PyObject *)&basis1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Polynomial1D", (PyObject *)&polynomial1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Spline1D", (PyObject *)&spline1d_type_object);
    if (res)
    {
        goto failed;
    }
    res = PyModule_AddObjectRef(mod, "Spline1Di", (PyObject *)&spline1di_type_object);
    if (res)
    {
        goto failed;
    }

    return mod;

failed:
    // Py_XDECREF(spline1d_type);
    // Py_XDECREF(poly_basis_type);
    // Py_XDECREF(basis1d_type);
    Py_XDECREF(mod);
    return NULL;
}
