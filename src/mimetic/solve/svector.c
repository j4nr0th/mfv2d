//
// Created by jan on 19.3.2025.
//

#include "svector.h"
#include "numpy/ndarrayobject.h"

int sparse_vector_new(svector_t *this, uint64_t n, uint64_t capacity, const allocator_callbacks *allocator)
{
    entry_t *const entries = allocate(allocator, sizeof *entries * capacity);
    if (!entries)
    {
        return -1;
    }

    this->count = 0;
    this->n = n;
    this->capacity = capacity;
    this->entries = entries;

    return 0;
}

void sparse_vec_del(svector_t *this, const allocator_callbacks *allocator)
{
    deallocate(allocator, this->entries);
    *this = (svector_t){};
}

int sparse_vec_resize(svector_t *this, uint64_t capacity, const allocator_callbacks *allocator)
{
    if (this->capacity >= capacity)
        return 0;

    entry_t *const new_ptr = reallocate(allocator, this->entries, sizeof(*this->entries) * capacity);
    if (!new_ptr)
    {
        return -1;
    }
    this->entries = new_ptr;
    return 0;
}

svec_object_t *sparse_vec_to_python(const svector_t *this)
{
    svec_object_t *const self = (svec_object_t *)svec_type_object.tp_alloc(&svec_type_object, (Py_ssize_t)this->count);
    if (!self)
        return NULL;

    for (uint64_t i = 0; i < this->count; ++i)
    {
        self->entries[i] = this->entries[i];
    }

    self->capacity = this->count;
    self->count = this->count;
    self->n = this->n;
    return self;
}

static PyObject *svec_repr(const svec_object_t *this)
{
    size_t capacity = 8 * this->count + 64;
    size_t count = 0;
    char *buffer = PyMem_RawMalloc(capacity * sizeof *buffer);
    if (!buffer)
        return NULL;

    count += snprintf(buffer + count, capacity - count, "SparseVector(%zu,", this->n);
    for (uint64_t i = 0; i < this->count; ++i)
    {
        if (capacity - count <= 64)
        {
            const size_t new_capacity = capacity << 1;
            char *const new_ptr = PyMem_RawRealloc(buffer, new_capacity);
            if (!new_ptr)
            {
                PyMem_RawFree(buffer);
                return NULL;
            }
            buffer = new_ptr;
            capacity = new_capacity;
        }
        count += snprintf(buffer + count, capacity - count, " (%" PRIu64 ", %g),", this->entries[i].index,
                          this->entries[i].value);
    }
    buffer[count - 1] = ')';
    PyObject *const str_out = PyUnicode_FromString(buffer);
    PyMem_RawFree(buffer);
    return str_out;
}

static PyObject *svec_from_entries(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    // Check if indice/values are given as two arrays
    Py_ssize_t n;
    PyObject *py_indices, *py_values;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "nOO", (char *[4]){"n", "indices", "values", NULL}, &n, &py_indices,
                                     &py_values))
    {
        return NULL;
    }

    if (n <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "The dimension of the vector must be strictly positive.");
        return NULL;
    }
    // These are as two arrays
    PyArrayObject *const array_indices = (PyArrayObject *)PyArray_FromAny(
        py_indices, PyArray_DescrFromType(NPY_UINT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!array_indices)
        return NULL;
    PyArrayObject *const array_values = (PyArrayObject *)PyArray_FromAny(
        py_values, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!array_values)
    {
        Py_DECREF(array_indices);
        return NULL;
    }
    const npy_double *const pv = PyArray_DATA(array_values);
    const npy_uint64 *const pi = PyArray_DATA(array_indices);

    const ssize_t count = PyArray_DIM(array_indices, 0);
    if (count != PyArray_DIM(array_values, 0))
    {
        PyErr_Format(PyExc_ValueError,
                     "Indices and values must be arrays of equal length. Instead indices had %u elements and values "
                     "had %u elements.",
                     (unsigned)count, (unsigned)PyArray_DIM(array_values, 0));
        Py_DECREF(array_values);
        Py_DECREF(array_indices);
        return NULL;
    }

    for (uint64_t i = 1; i < count; ++i)
    {
        if (pi[i - 1] >= pi[i])
        {
            PyErr_Format(PyExc_ValueError, "Entry indices %" PRIu64 " and %" PRIu64 " are not sorted.", i, i + 1);
            Py_DECREF(array_values);
            Py_DECREF(array_indices);
            return NULL;
        }
    }
    for (uint64_t i = 0; i < count; ++i)
    {
        if (pi[i] >= n)
        {
            PyErr_Format(PyExc_ValueError, "Entry index %" PRIu64 " is outside the allowed range [0, %" PRIu64 ").", i,
                         pi[i], (uint64_t)n);
            Py_DECREF(array_values);
            Py_DECREF(array_indices);
            return NULL;
        }
    }

    svec_object_t *const this = (svec_object_t *)type->tp_alloc(type, count);
    if (!this)
    {
        Py_DECREF(array_values);
        Py_DECREF(array_indices);
        return NULL;
    }

    this->capacity = count;
    this->count = count;
    this->n = (uint64_t)n;

    for (uint64_t i = 0; i < count; ++i)
    {
        this->entries[i] = (entry_t){.index = pi[i], .value = pv[i]};
    }
    Py_DECREF(array_values);
    Py_DECREF(array_indices);

    return (PyObject *)this;
}

static PyObject *svec_array(const svec_object_t *this, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *dtype = NULL;
    int b_copy = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Op", (char *[3]){"dtype", "copy", NULL}, &dtype, &b_copy))
    {
        return NULL;
    }

    if (!b_copy)
    {
        PyErr_SetString(PyExc_ValueError, "A copy is always created when converting to NDArray.");
        return NULL;
    }

    const npy_intp size = (npy_intp)this->n;

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_FLOAT64);
    if (!out)
        return NULL;

    npy_float64 *const ptr = PyArray_DATA(out);
    memset(ptr, 0, sizeof *ptr * size);

    for (unsigned i = 0; i < this->count; ++i)
    {
        ptr[this->entries[i].index] = this->entries[i].value;
    }

    if (dtype)
    {
        PyObject *const new_out = PyArray_CastToType(out, dtype, 0);
        Py_DECREF(out);
        return new_out;
    }

    return (PyObject *)out;
}

static PyMethodDef svec_methods[] = {
    {.ml_name = "__array__",
     .ml_meth = (void *)svec_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray[int]\n"
               "Convert to numpy array.\n"},
    {.ml_name = "from_entries",
     .ml_meth = (void *)svec_from_entries,
     .ml_flags = METH_CLASS | METH_KEYWORDS | METH_VARARGS,
     .ml_doc = "from_entries(n: int, indices: array_like, values: array_like) -> SparseVector:\n"
               "Create sparse vector from an array of indices and values.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n : int\n"
               "    Dimension of the vector.\n"
               "\n"
               "indices : array_like\n"
               "    Indices of the entries. Must be sorted.\n"
               "\n"
               "values : array_like\n"
               "    Values of the entries.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "SparseVector\n"
               "    New vector with indices and values as given.\n"},
    {}, // Sentinel
};

static PyObject *svec_get_n(const svec_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->n);
}

static PyObject *svec_get_values(const svec_object_t *this, void *Py_UNUSED(closure))
{
    const npy_intp size = (npy_intp)this->count;
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_FLOAT64);
    if (!array)
        return NULL;
    npy_float64 *const ptr = PyArray_DATA(array);
    for (uint64_t i = 0; i < this->count; ++i)
    {
        ptr[i] = this->entries[i].value;
    }

    return (PyObject *)array;
}

static PyObject *svec_get_indices(const svec_object_t *this, void *Py_UNUSED(closure))
{
    const npy_intp size = (npy_intp)this->count;
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_UINT64);
    if (!array)
        return NULL;
    npy_uint64 *const ptr = PyArray_DATA(array);
    for (uint64_t i = 0; i < this->count; ++i)
    {
        ptr[i] = this->entries[i].index;
    }

    return (PyObject *)array;
}

static PyGetSetDef svec_get_set[] = {
    {.name = "n", .get = (void *)svec_get_n, .set = NULL, .doc = "int : Dimension of the vector.", .closure = NULL},
    {.name = "values",
     .get = (void *)svec_get_values,
     .set = NULL,
     .doc = "array : Values of non-zero entries of the vector.",
     .closure = NULL},
    {.name = "indices",
     .get = (void *)svec_get_indices,
     .set = NULL,
     .doc = "array : Indices of non-zero entries of the vector.",
     .closure = NULL},
    {}, // Sentinel
};

PyTypeObject svec_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.SparseVector",
    .tp_basicsize = sizeof(svec_object_t),
    .tp_itemsize = sizeof(entry_t),
    .tp_repr = (reprfunc)svec_repr,
    // .tp_as_mapping = ,
    // .tp_hash = ,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    // .tp_doc = ,
    // .tp_richcompare = ,
    // .tp_iter = ,
    // .tp_iternext = ,
    .tp_methods = svec_methods,
    .tp_getset = svec_get_set,
};
