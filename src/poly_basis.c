//
// Created by jan on 21.10.2024.
//

#include "poly_basis.h"

#include <numpy/arrayobject.h>

#include "common.h"

static PyObject *poly_basis_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject* input;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char*[2]){"", NULL}, &input))
    {
        return NULL;
    }

    PyObject* array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return NULL;
    }

    const unsigned n = PyArray_Size(array);
    allocfunc alloc = PyType_GetSlot(type, Py_tp_alloc);
    polynomial_basis_t* this = (polynomial_basis_t*)alloc(type, (Py_ssize_t)(n ? n : 0));
    if (!this)
    {
        goto end;
    }

    this->n = n;
    const double* k = PyArray_DATA((PyArrayObject*)array);
    for (unsigned i = 0; i < n; ++i)
    {
        this->k[i] = k[i];
    }

end:
    Py_DECREF(array);
    return (PyObject*)this;
}

static PyObject *basis_call(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    PyObject* input;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char*[2]){"", NULL}, &input))
    {
        return NULL;
    }
    PyObject* array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ENSURECOPY, NULL);
    if (!array)
    {
        return NULL;
    }

    const npy_intp m = PyArray_SIZE((PyArrayObject*)array);
    const unsigned n = this->n;
    const double* restrict k = this->k;
    double* restrict const pv = PyArray_DATA((PyArrayObject*)array);

    //  Could make this OpenMP?
    for (npy_intp i = 0; i < m; ++i)
    {
        const double v = pv[i];
        double vv = 1.0;
        double sum = k[0];
        for (unsigned j = 1; j < n; ++j)
        {
            vv *= v;
            sum += k[j] * vv;
        }
        pv[i] = sum;
    }

    return array;
}

static PyObject *basis_derivative(PyObject *self, PyObject* Py_UNUSED(args))
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;

    polynomial_basis_t* out;

    if (this->n <= 1)
    {
        out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, 1);
        if (!out)
        {
            return NULL;
        }
        out->n = 1;
        out->k[0] = 0.0;
        return (PyObject*)out;
    }
    out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, this->n - 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n - 1;
    for (unsigned i = 1; i < this->n; ++i)
    {
        out->k[i - 1] = this->k[i] * (double)i;
    }
    return (PyObject*)out;
}

static PyObject *basis_antiderivative(PyObject *self, PyObject* Py_UNUSED(args))
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;

    polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, this->n + 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n + 1;
    out->k[0] = 0.0;
    for (unsigned i = 0; i < this->n; ++i)
    {
        out->k[i + 1] = this->k[i] / (double)(i + 1);
    }
    return (PyObject*)out;
}

static PyObject *poly_basis_get_coefficients(PyObject* self, void* Py_UNUSED(closure))
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    npy_intp n = this->n;
    PyObject* array = PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, (void*)this->k);
    if (!array)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject*)array, self))
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    return array;
}

static int poly_basis_set_coefficients(PyObject* self, PyObject* v, void* Py_UNUSED(closure))
{
    polynomial_basis_t* this = (polynomial_basis_t*)self;
    PyObject* array = PyArray_FromAny(v, PyArray_DescrFromType(NPY_DOUBLE), 0, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return -1;
    }
    const unsigned n = PyArray_SIZE((PyArrayObject*)array);
    if (n != this->n)
    {
        Py_DECREF(array);
        (void)PyErr_Format(PyExc_ValueError, "Polynomial has %u coefficients, but %u were given instead.", n);
        return -1;
    }

    const double* k = PyArray_DATA((PyArrayObject*)array);
    for (unsigned i = 0; i < n; ++i)
    {
        this->k[i] = k[i];
    }

    Py_DECREF(array);
    return 0;
}

static PyObject *poly_basis_str(PyObject* self)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    unsigned char_cnt = 1;

    char_cnt += snprintf(NULL, 0, "%g", this->k[0]);
    if (this->n >= 2)
    {
        char_cnt += snprintf(NULL, 0, "%g * x + ", this->k[1]);
    }
    for (unsigned i = 2; i < this->n; ++i)
    {
        char_cnt += snprintf(NULL, 0, "%g * x^%u + ", this->k[i], i);
    }

    char* buffer = PyMem_Malloc(sizeof(*buffer) * char_cnt);
    if (!buffer)
    {
        return PyErr_NoMemory();
    }
    unsigned written = 0;

    for (unsigned i = this->n; i > 2; --i)
    {
        written += snprintf(buffer + written, char_cnt - written, "%g * x^%u + ", this->k[i], i - 1);
    }
    if (this->n >= 2)
    {
        written += snprintf(buffer + written, char_cnt - written, "%g * x + ", this->k[1]);
    }

    written += snprintf(buffer + written, char_cnt - written, "%g", this->k[0]);
    (void)written;
    buffer[char_cnt - 1] = 0;

    PyObject* out = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    buffer = NULL;

    return out;
}

static PyObject *poly_basis_repr(PyObject* self)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    unsigned char_cnt = 10;

    char_cnt += snprintf(NULL, 0, "Polynomial1D([ ");
    for (unsigned i = 0; i < this->n; ++i)
    {
        char_cnt += snprintf(NULL, 0, "%g, ", this->k[i]);
    }
    char_cnt += snprintf(NULL, 0, " ])");

    char* const buffer = allocate(&PYTHON_ALLOCATOR, sizeof(*buffer) * char_cnt);
    if (!buffer)
    {
        return PyErr_NoMemory();
    }
    unsigned written = 0;
    written += snprintf(buffer + written, char_cnt - written, "Polynomial1D([ ");
    for (unsigned i = 0; i < this->n; ++i)
    {
        written += snprintf(buffer + written, char_cnt - written, "%g, ", this->k[i]);
    }
    written += snprintf(buffer + written, char_cnt - written, " ])");
    (void)written;
    buffer[char_cnt - 1] = 0;

    PyObject* out = PyUnicode_FromString(buffer);
    deallocate(&PYTHON_ALLOCATOR, buffer);

    return out;
}

static PyObject *poly_basis_add(PyObject* self, PyObject* o)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    if (PyFloat_Check(o))
    {
        double k = PyFloat_AsDouble(o);
        polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, this->n);
        if (!out)
        {
            return NULL;
        }
        out->n = this->n;
        for (unsigned i = 0; i < this->n; ++i)
        {
            out->k[i] = this->k[i];
        }
        out->k[0] += k;
        return (PyObject*)out;
    }
    if (!PyObject_TypeCheck(o, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type))
    {
        return PyErr_Format(PyExc_TypeError, "Polynomial1D can only be multiplied by Polynomial1D or float.");
    }
    const polynomial_basis_t* other = (polynomial_basis_t*)o;

    const polynomial_basis_t *longer, *shorter;
    if (this->n > other->n)
    {
        longer = this;
        shorter = other;
    }
    else
    {
        shorter = this;
        longer = other;
    }

    polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, longer->n);
    if (!out)
    {
        return NULL;
    }
    out->n = longer->n;
    unsigned i;
    for (i = 0; i < shorter->n; ++i)
    {
        out->k[i] = longer->k[i] + shorter->k[i];
    }
    for (;i < longer->n; ++i)
    {
        out->k[i] = longer->k[i];
    }
    return (PyObject*)out;
}

static PyObject *poly_basis_mul(PyObject* self, PyObject* o)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;
    if (PyFloat_Check(o))
    {
        double k = PyFloat_AsDouble(o);
        polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, this->n);
        if (!out)
        {
            return NULL;
        }
        out->n = this->n;
        for (unsigned i = 0; i < this->n; ++i)
        {
            out->k[i] = this->k[i] * k;
        }
        return (PyObject*)out;
    }
    if (!PyObject_TypeCheck(o, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type))
    {
        return PyErr_Format(PyExc_TypeError, "Polynomial1D can only be multiplied by Polynomial1D or float.");
    }
    const polynomial_basis_t* other = (polynomial_basis_t*)o;
    polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, (this->n - 1) + (other->n - 1) + 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n + other->n - 1;
    for (unsigned i = 0; i < out->n; ++i)
    {
        out->k[i] = 0.0;
    }
    for (unsigned i = 0; i < this->n; ++i)
    {
        for (unsigned j = 0; j < other->n; ++j)
        {
            out->k[i + j] += this->k[i] * other->k[j];
        }
    }
    return (PyObject*)out;
}

static PyObject *poly_basis_neg(PyObject* self)
{
    const polynomial_basis_t* this = (polynomial_basis_t*)self;

    polynomial_basis_t* out = PyObject_NewVar(polynomial_basis_t, (PyTypeObject*)INTERPLIB_PYTHON_API.poly1d_type, this->n);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n;
    for (unsigned i = 0; i < out->n; ++i)
    {
        out->k[i] = -this->k[i];
    }
    return (PyObject*)out;
}

static PyGetSetDef poly_basis_getset[] =
    {
        {.name = "coefficients", .get = poly_basis_get_coefficients, .set = poly_basis_set_coefficients, .doc = "Coefficients of the polynomial", .closure = NULL},
        {NULL, NULL, NULL, NULL, NULL} // sentinel
    };

static PyMethodDef poly_basis_methods[] =
    {
        {"derivative", basis_derivative, METH_NOARGS},
        {"antiderivative", basis_antiderivative, METH_NOARGS},
        {NULL, NULL, 0, NULL}, // sentinel
    };


static PyType_Slot poly_basis_slots[] =
    {
    {.slot = Py_tp_base, .pfunc = NULL},
    {.slot = Py_tp_new, .pfunc = poly_basis_new},
    {.slot = Py_tp_call, .pfunc = basis_call},
    {.slot = Py_tp_methods, .pfunc = poly_basis_methods},
    {.slot = Py_tp_getset, .pfunc = poly_basis_getset},
    {.slot = Py_tp_str, .pfunc = poly_basis_str},
    {.slot = Py_tp_repr, .pfunc = poly_basis_repr},
    {.slot = Py_nb_add, .pfunc = poly_basis_add},
    {.slot = Py_nb_multiply, .pfunc = poly_basis_mul},
    {.slot = Py_nb_negative, .pfunc = poly_basis_neg},
    {.slot = 0, .pfunc = NULL}, // sentinel
    };

PyType_Spec poly_basis_type_spec =
    {
        .name = "_interp.Polynomial1D",
        .basicsize = sizeof(polynomial_basis_t),
        .itemsize = sizeof(double),
        .flags = Py_TPFLAGS_BASETYPE|Py_TPFLAGS_DEFAULT|Py_TPFLAGS_IMMUTABLETYPE,
        .slots = poly_basis_slots,
    };
