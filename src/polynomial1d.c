//
// Created by jan on 21.10.2024.
//

#include "polynomial1d.h"

#include <numpy/arrayobject.h>

#include "basis1d.h"
#include "common.h"
#include "lagrange.h"
#include <stddef.h>

static PyObject *polynomial1d_vectorcall(PyObject *self, PyObject *const *args, size_t nargsf,
                                         PyObject *Py_UNUSED(kwnames))
{
    if (PyVectorcall_NARGS(nargsf) != 1)
    {
        PyErr_Format(PyExc_ValueError,
                     "Polynomial1D can be called with only one argument, instead "
                     "%u were given",
                     (unsigned)nargsf);
        return NULL;
    }
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    PyObject *input = *args;
    PyObject *array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY, NULL);
    if (!array)
    {
        return NULL;
    }

    const npy_intp m = PyArray_SIZE((PyArrayObject *)array);
    const unsigned n = this->n;
    const double *restrict k = this->k;
    double *restrict const pv = PyArray_DATA((PyArrayObject *)array);

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

static PyObject *polynomial1d_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *input;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char *[2]){"", NULL}, &input))
    {
        return NULL;
    }

    PyObject *array = PyArray_FromAny(input, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return NULL;
    }

    const unsigned n = PyArray_Size(array);
    allocfunc alloc = PyType_GetSlot(type, Py_tp_alloc);
    polynomial_basis_t *this = (polynomial_basis_t *)alloc(type, (Py_ssize_t)(n ? n : 0));
    if (!this)
    {
        goto end;
    }

    this->n = n;
    this->call_poly = polynomial1d_vectorcall;
    const double *k = PyArray_DATA((PyArrayObject *)array);
    for (unsigned i = 0; i < n; ++i)
    {
        this->k[i] = k[i];
    }

end:
    Py_DECREF(array);
    return (PyObject *)this;
}

static PyObject *polynomial1d_derivative(PyObject *self, void *Py_UNUSED(closure))
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;

    polynomial_basis_t *out;

    if (this->n <= 1)
    {
        out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, 1);
        if (!out)
        {
            return NULL;
        }
        out->n = 1;
        out->call_poly = polynomial1d_vectorcall;
        out->k[0] = 0.0;
        return (PyObject *)out;
    }
    out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n - 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n - 1;
    out->call_poly = polynomial1d_vectorcall;
    for (unsigned i = 1; i < this->n; ++i)
    {
        out->k[i - 1] = this->k[i] * (double)i;
    }
    return (PyObject *)out;
}

static PyObject *polynomial1d_antiderivative(PyObject *self, void *Py_UNUSED(closure))
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;

    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n + 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n + 1;
    out->call_poly = polynomial1d_vectorcall;
    out->k[0] = 0.0;
    for (unsigned i = 0; i < this->n; ++i)
    {
        out->k[i + 1] = this->k[i] / (double)(i + 1);
    }
    return (PyObject *)out;
}

static PyObject *polynomial1d_get_coefficients(PyObject *self, void *Py_UNUSED(closure))
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    npy_intp n = this->n;
    PyObject *array = PyArray_SimpleNewFromData(1, &n, NPY_DOUBLE, (void *)this->k);
    if (!array)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)array, self))
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    return array;
}

static int polynomial1d_set_coefficients(PyObject *self, PyObject *v, void *Py_UNUSED(closure))
{
    polynomial_basis_t *this = (polynomial_basis_t *)self;
    PyObject *array = PyArray_FromAny(v, PyArray_DescrFromType(NPY_DOUBLE), 0, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        return -1;
    }
    const unsigned n = PyArray_SIZE((PyArrayObject *)array);
    if (n != this->n)
    {
        Py_DECREF(array);
        (void)PyErr_Format(PyExc_ValueError, "Polynomial has %u coefficients, but %u were given instead.", n);
        return -1;
    }

    const double *k = PyArray_DATA((PyArrayObject *)array);
    for (unsigned i = 0; i < n; ++i)
    {
        this->k[i] = k[i];
    }

    Py_DECREF(array);
    return 0;
}

static PyObject *polynomial1d_str(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
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

    char *buffer = PyMem_Malloc(sizeof(*buffer) * char_cnt);
    if (!buffer)
    {
        return PyErr_NoMemory();
    }
    unsigned written = 0;

    for (unsigned i = this->n; i > 2; --i)
    {
        written += snprintf(buffer + written, char_cnt - written, "%g * x^%u + ", this->k[i - 1], i - 1);
    }
    if (this->n >= 2)
    {
        written += snprintf(buffer + written, char_cnt - written, "%g * x + ", this->k[1]);
    }

    written += snprintf(buffer + written, char_cnt - written, "%g", this->k[0]);
    (void)written;
    buffer[char_cnt - 1] = 0;

    PyObject *out = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    buffer = NULL;

    return out;
}

static PyObject *polynomial1d_repr(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    unsigned char_cnt = 10;

    char_cnt += snprintf(NULL, 0, "Polynomial1D([ ");
    for (unsigned i = 0; i < this->n; ++i)
    {
        char_cnt += snprintf(NULL, 0, "%g, ", this->k[i]);
    }
    char_cnt += snprintf(NULL, 0, " ])");

    char *const buffer = allocate(&PYTHON_ALLOCATOR, sizeof(*buffer) * char_cnt);
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
    written += snprintf(buffer + written, char_cnt - written, "])");
    (void)written;
    buffer[char_cnt - 1] = 0;

    PyObject *out = PyUnicode_FromString(buffer);
    deallocate(&PYTHON_ALLOCATOR, buffer);

    return out;
}

static PyObject *polynomial1d_add(PyObject *o1, PyObject *o2)
{
    //  Check which is the polynomial
    if (!PyObject_TypeCheck(o1, &polynomial1d_type_object))
    {
        //  Make o1 always be the polynomial
        PyObject *tmp = o2;
        o2 = o1;
        o1 = tmp;
    }
    const polynomial_basis_t *this = (polynomial_basis_t *)o1;
    double k = PyFloat_AsDouble(o2);
    if (!PyErr_Occurred())
    {
        polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n);
        if (!out)
        {
            return NULL;
        }
        out->n = this->n;
        out->call_poly = polynomial1d_vectorcall;
        for (unsigned i = 0; i < this->n; ++i)
        {
            out->k[i] = this->k[i];
        }
        out->k[0] += k;
        return (PyObject *)out;
    }
    // conversion to double failed, so clear exception and check for polynomial
    // instead PyErr_SetHandledException(NULL);
    PyErr_Clear();
    if (!PyObject_TypeCheck(o2, &polynomial1d_type_object))
    {
        // Couldn't get a float, nor is it a polynomial.
        PyErr_Format(PyExc_TypeError, "Polynomial1D can only be added to Polynomial1D or float.");
        return NULL;
    }
    const polynomial_basis_t *other = (polynomial_basis_t *)o2;

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

    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, longer->n);
    if (!out)
    {
        return NULL;
    }
    out->n = longer->n;
    out->call_poly = polynomial1d_vectorcall;
    unsigned i;
    for (i = 0; i < shorter->n; ++i)
    {
        out->k[i] = longer->k[i] + shorter->k[i];
    }
    for (; i < longer->n; ++i)
    {
        out->k[i] = longer->k[i];
    }
    return (PyObject *)out;
}

INTERPLIB_INTERNAL
void multiply_polynomials(unsigned n1, const double INTERPLIB_ARRAY_ARG(k1, restrict static n1), unsigned n2,
                          const double INTERPLIB_ARRAY_ARG(k2, static n2),
                          double INTERPLIB_ARRAY_ARG(out, ((n1 - 1) + (n2 - 1) + 1)))
{
    //  must manually set the highest term to zero, since it will be a special case
    for (unsigned i = n1; i > 0; --i)
    {
        out[(i - 1) + (n2 - 1)] = k1[(i - 1)] * k2[(n2 - 1)];
    }
    for (unsigned j = n2 - 1; j > 0; --j)
    {
        for (unsigned i = n1; i > 1; --i)
        {
            out[(i - 1) + (j - 1)] += k1[(i - 1)] * k2[(j - 1)];
        }
        out[(j - 1)] = k1[0] * k2[(j - 1)];
    }
}

static PyObject *polynomial1d_mul(PyObject *o1, PyObject *o2)
{
    //  Check which is the polynomial
    if (!PyObject_TypeCheck(o1, &polynomial1d_type_object))
    {
        //  Make o1 always be the polynomial
        PyObject *tmp = o2;
        o2 = o1;
        o1 = tmp;
    }

    const polynomial_basis_t *this = (polynomial_basis_t *)o1;
    double k = PyFloat_AsDouble(o2);
    if (!PyErr_Occurred())
    {
        polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n);
        if (!out)
        {
            return NULL;
        }
        out->n = this->n;
        out->call_poly = polynomial1d_vectorcall;
        for (unsigned i = 0; i < this->n; ++i)
        {
            out->k[i] = this->k[i] * k;
        }
        return (PyObject *)out;
    }
    // conversion to double failed, so clear exception and check for polynomial
    // instead PyErr_SetHandledException(NULL);
    PyErr_Clear();
    if (!PyObject_TypeCheck(o2, &polynomial1d_type_object))
    {
        // Couldn't get a float, nor is it a polynomial.
        PyErr_Format(PyExc_TypeError, "Polynomial1D can only be multiplied by "
                                      "Polynomial1D or float.");
        return NULL;
    }
    const polynomial_basis_t *other = (polynomial_basis_t *)o2;
    polynomial_basis_t *out =
        PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, (this->n - 1) + (other->n - 1) + 1);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n + other->n - 1;
    out->call_poly = polynomial1d_vectorcall;
    multiply_polynomials(this->n, this->k, other->n, other->k, out->k);
    return (PyObject *)out;
}

static PyObject *polynomial1d_neg(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;

    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n;
    out->call_poly = polynomial1d_vectorcall;
    for (unsigned i = 0; i < out->n; ++i)
    {
        out->k[i] = -this->k[i];
    }
    return (PyObject *)out;
}

static PyObject *polynomial1d_copy(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;

    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n;
    out->call_poly = polynomial1d_vectorcall;
    for (unsigned i = 0; i < out->n; ++i)
    {
        out->k[i] = this->k[i];
    }
    return (PyObject *)out;
}

static PyObject *polynomial1d_as_pyfloat(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    if (this->n > 1)
    {
        PyErr_Format(PyExc_TypeError,
                     "Polynomials with anything more than a single (constant) term can not be converted to floats.");
        return NULL;
    }
    return PyFloat_FromDouble(this->k[0]);
}

static PyObject *polynomial1d_pow(PyObject *self, PyObject *o, PyObject *modulo)
{
    if (!Py_IsNone(modulo))
    {
        PyErr_Format(PyExc_TypeError, "Polynomial1D can't be given a modulo");
        return NULL;
    }
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    unsigned long long p = PyLong_AsUnsignedLongLong(o);
    if (PyErr_Occurred())
    {
        // Couldn't get an int
        PyErr_Format(PyExc_TypeError, "Polynomial1D can only be raised to a power of an integer. ");
        return NULL;
    }

    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, p * (this->n - 1) + 1);
    if (!out)
    {
        return NULL;
    }
    out->call_poly = polynomial1d_vectorcall;
    out->n = 1;
    for (unsigned i = 1; i < p * (this->n - 1) + 1; ++i)
    {
        out->k[i] = 0.0;
    }
    out->k[0] = 1.0;
    for (unsigned power = 0; power < p; ++power)
    {
        // for (unsigned j = out->n; j > 0; --j)
        // {
        //     for (unsigned i = this->n; i > 1; --i)
        //     {
        //         out->k[(i - 1) + (j - 1)] += this->k[(i - 1)] * out->k[(j - 1)];
        //     }
        //     out->k[(j - 1)] = this->k[0] * out->k[(j - 1)];
        // }
        multiply_polynomials(this->n, this->k, out->n, out->k, out->k);
        out->n += this->n - 1;
    }
    return (PyObject *)out;
}

static Py_ssize_t polynomial1d_length(PyObject *self)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    return (Py_ssize_t)this->n;
}

static PyObject *polynomial1d_get_coefficient(PyObject *self, Py_ssize_t idx)
{
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    if ((unsigned)idx >= this->n)
    {
        PyErr_Format(PyExc_IndexError, "Index %u is out of bounds for a polynomial with %u terms.", (unsigned)idx,
                     this->n);
        return NULL;
    }
    return PyArray_Scalar((void *)(this->k + idx), PyArray_DescrFromType(NPY_DOUBLE), NULL);
}

static int polynomial1d_set_coefficient(PyObject *self, Py_ssize_t idx, PyObject *arg)
{
    polynomial_basis_t *this = (polynomial_basis_t *)self;
    if ((unsigned)idx >= this->n)
    {
        PyErr_Format(PyExc_IndexError, "Index %u is out of bounds for a polynomial with %u terms.", (unsigned)idx,
                     this->n);
        return -1;
    }
    const double v = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
    {
        return -1;
    }
    this->k[idx] = v;
    return 0;
}

PyDoc_STRVAR(lagrange_nodal_basis_docstring,
             "lagrange_nodal_basis(self, nodes) -> tuple[Polynomial1D, ...]\n"
             "Return Lagrange nodal polynomial basis on the given set of nodes.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "nodes : (N,) array_like\n"
             "    Nodes used for Lagrange polynomials.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "tuple of ``N`` :class:`Polynomial1D`\n"
             "    Lagrange basis polynomials, which are one at the node of their index\n"
             " and zero at all other.\n");
static PyObject *lagrange_nodal_basis(PyObject *cls, PyObject *arg)
{
    PyArrayObject *const nodes =
        (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!nodes)
    {
        return NULL;
    }
    PyTupleObject *out = NULL;
    double *restrict denominators = NULL;
    const unsigned n = (unsigned)PyArray_SIZE(nodes);
    const double *restrict x = PyArray_DATA(nodes);
    int failed = 0;

    if (n == 0)
    {
        PyErr_Format(PyExc_ValueError, "Zero nodes were provided.");
        failed = 1;
        goto end;
    }

    denominators = PyMem_Malloc(sizeof(*denominators) * n);
    if (!denominators)
    {
        failed = 1;
        goto end;
    }
    // Pre-compute denominators
    lagrange_polynomial_denominators(n, x, denominators);

    //  Invert the denominator
    for (unsigned i = 0; i < n; ++i)
    {
        denominators[i] = 1.0 / denominators[i];
    }

    out = (PyTupleObject *)PyTuple_New(n);
    if (!out)
    {
        failed = 1;
        goto end;
    }

    for (unsigned j = 0; j < n; ++j)
    {
        polynomial_basis_t *this = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, n);
        if (!this)
        {
            failed = 1;
            goto end;
        }
        this->n = n;
        this->call_poly = polynomial1d_vectorcall;
        lagrange_polynomial_coefficients(n, j, x, this->k);
        for (unsigned i = 0; i < n; ++i)
        {
            this->k[i] *= denominators[j];
        }
        PyTuple_SET_ITEM((PyObject *)out, j, (PyObject *)this);
    }

end:

    if (failed && out)
    {
        for (unsigned i = 0; i < PyTuple_GET_SIZE(out); ++i)
        {
            polynomial_basis_t *this = (polynomial_basis_t *)PyTuple_GET_ITEM(out, i);
            Py_XDECREF(this);
            PyTuple_SET_ITEM(out, i, NULL);
        }
        Py_DECREF(out);
        out = NULL;
    }
    PyMem_Free(denominators);
    Py_DECREF(nodes);
    return (PyObject *)out;
}

PyDoc_STRVAR(lagrange_nodal_fit_docstring,
             "lagrange_nodal_fit(self, nodes: npt.ArrayLike, values: npt.ArrayLike) -> Polynomial1D\n"
             "Use Lagrange nodal polynomial basis to fit a function.\n"
             "\n"
             "Equivalent to calling ``sum(b * y for (b, y) in\n"
             " zip(Polynomial1D.lagrange_nodal_basis(x), f(x))``\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "nodes : (N,) array_like\n"
             "    Nodes used for Lagrange polynomials.\n"
             "values : (N,) array_like\n"
             "    Values of the function at the nodes.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Polynomial1D\n"
             "    Polynomial of order ``N`` which matches function exactly at the nodes.\n");
static PyObject *lagrange_nodal_fit(PyObject *cls, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function always takes exactly two parameters, but was called with %z", nargs);
        return NULL;
    }
    PyArrayObject *nodes = NULL, *values = NULL;
    nodes = (PyArrayObject *)PyArray_FromAny(args[0], PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS,
                                             NULL);
    if (!nodes)
    {
        return NULL;
    }
    values = (PyArrayObject *)PyArray_FromAny(args[1], PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS,
                                              NULL);
    if (!values)
    {
        Py_DECREF(nodes);
        return NULL;
    }
    double *restrict denominators = NULL;
    double *restrict current_basis = NULL;
    const unsigned n = (unsigned)PyArray_SIZE(nodes);
    const double *restrict x = PyArray_DATA(nodes);
    const double *restrict y = PyArray_DATA(values);
    int failed = 0;
    polynomial_basis_t *this = NULL;

    if (n != (unsigned)PyArray_SIZE(values))
    {
        PyErr_Format(PyExc_ValueError,
                     "Nodes and values arrays must have the same length, instead %u nodes were given and %u values.", n,
                     (unsigned)PyArray_SIZE(values));
        failed = 1;
        goto end;
    }
    if (n == 0)
    {
        PyErr_Format(PyExc_ValueError, "Zero nodes were provided.");
        failed = 1;
        goto end;
    }
    this = PyObject_NewVar(polynomial_basis_t, (PyTypeObject *)cls, n);
    if (!this)
    {
        failed = 1;
        goto end;
    }
    this->n = n;
    this->call_poly = polynomial1d_vectorcall;

    denominators = PyMem_Malloc(sizeof(*denominators) * n);
    if (!denominators)
    {
        failed = 1;
        goto end;
    }
    // Pre-compute denominators
    lagrange_polynomial_denominators(n, x, denominators);

    //  Invert the denominator and multiply with values
    for (unsigned i = 0; i < n; ++i)
    {
        denominators[i] = y[i] / denominators[i];
    }
    current_basis = PyMem_Malloc(sizeof(*current_basis) * n);
    if (!current_basis)
    {
        failed = 1;
        goto end;
    }

    //  First polynomial can be done in-place
    //  get coefficients
    lagrange_polynomial_coefficients(n, 0, x, this->k);
    //  multiply by value over the denominator
    for (unsigned i = 0; i < n; ++i)
    {
        this->k[i] *= denominators[0];
    }

    //  Other polynomials are done in steps
    for (unsigned j = 1; j < n; ++j)
    {
        //  Compute base value
        lagrange_polynomial_coefficients(n, j, x, current_basis);
        //  Add to final one
        for (unsigned i = 0; i < n; ++i)
        {
            this->k[i] += denominators[j] * current_basis[i];
        }
    }

end:
    if (failed && this)
    {
        Py_DECREF(this);
        this = NULL;
    }
    PyMem_Free(current_basis);
    PyMem_Free(denominators);
    Py_DECREF(nodes);
    Py_DECREF(values);
    return (PyObject *)this;
}

PyDoc_STRVAR(polynomial1d_offset_by_docstring,
             "offset_by(self, x: float | np.floating) -> Polynomial1D\n"
             "Compute polynomial offset by specified amount.\n"
             "\n"
             "The offset polynomial :math:`p^\\prime(t)` is such that:\n"
             "\n"
             ".. math:\n"
             "\n"
             "   p^\\prime(t) = p(t + t_0),\n"
             "\n"
             "where :math:`p(t)` is the original polynomial and :math:`t_0` is the offset.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "t : float\n"
             "   Amount by which to offset the polynomial by.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Polynomial1D\n"
             "   Polynomial which has been offset by the specified amount.\n");
static PyObject *polynomial1d_offset_by(PyObject *self, PyObject *arg)
{
    const double t0 = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    const polynomial_basis_t *this = (polynomial_basis_t *)self;
    polynomial_basis_t *out = PyObject_NewVar(polynomial_basis_t, &polynomial1d_type_object, this->n);
    if (!out)
    {
        return NULL;
    }
    out->n = this->n;
    out->call_poly = polynomial1d_vectorcall;

    //  Expand out the terms
    for (unsigned n = 0; n < this->n; ++n)
    {
        double t = t0;
        const double a = this->k[n];
        //  Assign the newly added term
        out->k[n] = a;
        // Add to previously already computed ones
        unsigned binomial_coefficient = 1;
        for (unsigned k = 0; k < n; ++k)
        {
            const unsigned kp = n - k - 1;
            binomial_coefficient = binomial_coefficient * (n - k) / (k + 1);
            out->k[kp] += a * (double)binomial_coefficient * t;
            t *= t0;
        }
    }

    return (PyObject *)out;
}

static PyGetSetDef polynomial1d_getset[] = {
    {.name = "coefficients",
     .get = polynomial1d_get_coefficients,
     .set = polynomial1d_set_coefficients,
     .doc = "coefficients(self) -> npt.NDArray[np.float64]\n"
            "Coefficients of the polynomial.\n",
     .closure = NULL},
    {.name = "derivative",
     .get = polynomial1d_derivative,
     .set = NULL,
     .doc = "derivative(self) -> npt.NDArray[np.float64]\n"
            "Return the derivative of the polynomial.\n",
     .closure = NULL},
    {.name = "antiderivative",
     .get = polynomial1d_antiderivative,
     .set = NULL,
     .doc = "antiderivative(self) -> npt.NDArray[np.float64]\n"
            "Return the antiderivative of the polynomial.\n",
     .closure = NULL},
    {NULL, NULL, NULL, NULL, NULL} // sentinel
};

static PyNumberMethods polynomial1d_number_methods = {
    .nb_add = polynomial1d_add,
    .nb_multiply = polynomial1d_mul,
    .nb_negative = polynomial1d_neg,
    .nb_float = polynomial1d_as_pyfloat,
    .nb_positive = polynomial1d_copy,
    .nb_power = polynomial1d_pow,
};

static PySequenceMethods polynomial1d_sequence_methods = {
    .sq_length = polynomial1d_length,
    .sq_item = polynomial1d_get_coefficient,
    .sq_ass_item = polynomial1d_set_coefficient,
};

static PyMethodDef polynomial1d_methods[] = {
    {.ml_name = "lagrange_nodal_basis",
     .ml_meth = lagrange_nodal_basis,
     .ml_flags = METH_O | METH_CLASS,
     .ml_doc = lagrange_nodal_basis_docstring},
    {.ml_name = "lagrange_nodal_fit",
     .ml_meth = (PyCFunction)lagrange_nodal_fit,
     .ml_flags = METH_FASTCALL | METH_CLASS,
     .ml_doc = lagrange_nodal_fit_docstring},
    {.ml_name = "offset_by",
     .ml_meth = (PyCFunction)polynomial1d_offset_by,
     .ml_flags = METH_O,
     .ml_doc = polynomial1d_offset_by_docstring},
    {NULL, NULL, 0, NULL}, // sentinel
};

PyDoc_STRVAR(polynomial1d_docstring, "Polynomial1D(coefficients: npt.ArrayLike)\n"
                                     "Function with increasing integer power basis.\n"
                                     "\n"
                                     "Parameters\n"
                                     "----------\n"
                                     "coefficients : (N,) array_like\n"
                                     "    Coefficients of the polynomial, starting at the constant term, up to the\n"
                                     "    term with the highest power.\n");

INTERPLIB_INTERNAL
PyTypeObject polynomial1d_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_interp.Polynomial1D",
    .tp_basicsize = sizeof(polynomial_basis_t),
    .tp_itemsize = sizeof(double),
    .tp_vectorcall_offset = offsetof(polynomial_basis_t, call_poly),
    // .tp_repr = ,
    .tp_call = PyVectorcall_Call, // polynomial1d_call,
    // .tp_vectorcall = polynomial1d_vectorcall,
    .tp_iter = PySeqIter_New,
    .tp_str = polynomial1d_str,
    .tp_repr = polynomial1d_repr,
    .tp_doc = polynomial1d_docstring,
    .tp_methods = polynomial1d_methods,
    .tp_getset = polynomial1d_getset,
    .tp_base = &basis1d_type_object,
    .tp_new = polynomial1d_new,
    .tp_as_number = &polynomial1d_number_methods,
    .tp_as_sequence = &polynomial1d_sequence_methods,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_VECTORCALL,
};
