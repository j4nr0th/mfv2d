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

int sparse_vec_resize(svector_t *this, const uint64_t new_capacity, const allocator_callbacks *allocator)
{
    if (this->capacity >= new_capacity)
        return 0;

    entry_t *const new_ptr = reallocate(allocator, this->entries, sizeof(*this->entries) * new_capacity);
    if (!new_ptr)
    {
        return -1;
    }
    this->entries = new_ptr;
    this->capacity = new_capacity;
    return 0;
}

int sparse_vector_append(svector_t *this, const entry_t e, const allocator_callbacks *allocator)
{
    enum
    {
        MINIMUM_INCREMENT = 8
    };
    if (this->count >= this->capacity && sparse_vec_resize(this, this->count + MINIMUM_INCREMENT, allocator))
    {
        return -1;
    }
    ASSERT(this->count == 0 || e.index > this->entries[this->count - 1].index,
           "Must have higher index than last in array (index comparison %u vs %u).", (unsigned)e.index,
           (unsigned)this->entries[this->count - 1].index);

    this->entries[this->count] = e;
    this->count += 1;
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

    // self->capacity = this->count;
    self->count = this->count;
    self->n = this->n;
    return self;
}

int sparse_vector_copy(const svector_t *src, svector_t *dst, const allocator_callbacks *allocator)
{
    dst->n = src->n;
    if (sparse_vec_resize(dst, src->count, allocator))
        return -1;

    memcpy(dst->entries, src->entries, sizeof *dst->entries * src->count);
    dst->count = src->count;

    return 0;
}

uint64_t sparse_vector_find_first_geq(const svector_t *this, const uint64_t v, const uint64_t start)
{
    // Quick check if too small
    if (this->entries[0].index >= v)
        return 0;

    // Quick check if too large
    if (this->entries[this->count - 1].index <= v)
    {
        // Actually hit it
        if (this->entries[this->count - 1].index == v)
            return this->count - 1;

        // Nvm, too large
        return this->count;
    }

    enum
    {
        LINEAR_SEARCH_LIMIT = 8
    };
    uint64_t begin = start;
    // Since entries are sorted and unique, entry with value v can not be further ahead than the
    // difference from the first entry.
    const uint64_t first_diff = v - this->entries[0].index + 1;
    uint64_t len = (this->count < first_diff ? this->count : first_diff) - start;
    uint64_t d_pivot = len / 2;

    // Binary search until length is small enough to do the linear search
    while (len > LINEAR_SEARCH_LIMIT)
    {
        const uint64_t pv = this->entries[begin + d_pivot].index;
        if (pv == v)
        {
            return begin + d_pivot;
        }

        if (pv < v)
        {
            begin += d_pivot;
            len -= d_pivot;
        }
        else // if (pv > v)
        {
            len = d_pivot;
        }
        d_pivot = len / 2;
    }

    // Linear search is fine for small sections
    for (uint64_t i = 0; i < len; ++i)
    {
        if (this->entries[i + begin].index >= v)
            return i + begin;
    }

    return begin + len;
}

int sparse_vector_add_inplace(svector_t *first, const svector_t *second, const allocator_callbacks *allocator)
{
    if (second->count == 0)
        return 0;

    unsigned pos_1 = 0, pos_2 = 0;
    unsigned unique = 0;
    while (pos_1 < first->count && pos_2 < second->count)
    {
        if (first->entries[pos_1].index < second->entries[pos_2].index)
        {
            pos_1 += 1;
            unique += 1;
        }
        else if (first->entries[pos_1].index > second->entries[pos_2].index)
        {
            pos_2 += 1;
            unique += 1;
        }
        else
        {
            pos_1 += 1;
            pos_2 += 1;
            unique += 1;
        }
    }
    if (pos_1 < first->count)
    {
        ASSERT(pos_2 == second->count,
               "If the first vector ends before the second one, then all the unique entries must be from the second");
        unique += first->count - pos_1;
    }
    else if (pos_2 < second->count)
    {
        ASSERT(pos_1 == first->count,
               "If the first vector ends before the second one, then all the unique entries must be from the second");
        unique += second->count - pos_2;
    }

    const unsigned new_size = unique;
    if (first->capacity < new_size)
    {
        const unsigned new_capacity = new_size;
        if (sparse_vec_resize(first, new_capacity, allocator))
            return -1;
    }
    // Loop over backwards
    pos_1 = first->count;
    pos_2 = second->count;
    while (pos_1 > 0 && pos_2 > 0)
    {
        ASSERT(pos_1 <= unique,
               "There must never be less unique entries left than there are entries left in the first vector.");
        ASSERT(unique > 0, "There must always be some unique entries left.");
        if (first->entries[pos_1 - 1].index > second->entries[pos_2 - 1].index)
        {
            first->entries[unique - 1] = first->entries[pos_1 - 1];
            pos_1 -= 1;
            unique -= 1;
        }
        else if (first->entries[pos_1 - 1].index < second->entries[pos_2 - 1].index)
        {
            first->entries[unique - 1] = second->entries[pos_2 - 1];
            pos_2 -= 1;
            unique -= 1;
        }
        else // first->entries[pos_1 - 1].index == second->entries[pos_2 - 1].index
        {
            first->entries[unique - 1] =
                (entry_t){.value = first->entries[pos_1 - 1].value + second->entries[pos_2 - 1].value,
                          .index = second->entries[pos_2 - 1].index};
            pos_1 -= 1;
            pos_2 -= 1;
            unique -= 1;
        }
    }
    if (pos_1 != 0)
        ASSERT(pos_1 == unique, "If the second vector ends before the first one, then all the unique entries left "
                                "after the loop should be from the first vector.");

    if (pos_2 != 0)
    {
        ASSERT(pos_2 == unique, "If the first vector ends before the second one, then all the unique entries left "
                                "after the loop should be from the second vector.");
        memcpy(first->entries, second->entries, sizeof *second->entries * pos_2);
    }
    first->count = new_size;

    // TODO: remove
    for (unsigned i = 1; i < first->count; ++i)
    {
        ASSERT(first->entries[i - 1].index < first->entries[i].index,
               "The entries after merge must be sorted, but %u and %u were not", i - 1, i);
    }
    return 0;
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

    // this->capacity = count;
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

static PyObject *svec_get(const svec_object_t *this, PyObject *py_idx)
{
    if (PySlice_Check(py_idx))
    {
        // Quick check for an empty vector
        if (this->count == 0)
        {
            const svector_t v = {.n = this->n, .count = 0, .capacity = 0, .entries = NULL};
            return (PyObject *)sparse_vec_to_python(&v);
        }
        Py_ssize_t start, stop, step;
        // Slice
        if (PySlice_Unpack(py_idx, &start, &stop, &step))
        {
            return NULL;
        }
        if (step != 1)
        {
            PyErr_Format(PyExc_KeyError,
                         "Sparse vectors do not support slices with steps other than 1 (%lld was given).",
                         (long long int)step);
            return NULL;
        }
        const Py_ssize_t seq_len = PySlice_AdjustIndices((Py_ssize_t)this->n, &start, &stop, step);
        const svector_t self = {.n = this->n, .count = this->count, .capacity = 0, .entries = (entry_t *)this->entries};
        const uint64_t begin = sparse_vector_find_first_geq(&self, start, 0);
        svector_t fake;
        if (begin == this->count)
        {
            // Nothing in the range
            fake = (svector_t){.n = stop - start, .count = 0, .capacity = 0, .entries = NULL};
            return (PyObject *)sparse_vec_to_python(&fake);
        }
        const uint64_t end = sparse_vector_find_first_geq(&self, stop, begin);
        fake = (svector_t){
            .n = seq_len, .count = end - begin, .capacity = 0, .entries = (entry_t *)(this->entries + begin)};
        svec_object_t *const vec = sparse_vec_to_python(&fake);
        for (uint64_t i = 0; i < vec->count; ++i)
        {
            vec->entries[i].index -= start;
        }
        return (PyObject *)vec;
    }

    const Py_ssize_t idx = PyLong_AsSsize_t(py_idx);
    if (PyErr_Occurred())
        return NULL;

    if (idx >= this->n || idx < -this->n)
    {
        PyErr_Format(PyExc_KeyError,
                     "Index %" PRIi64 " is outside the allowed range for a vector of dimension %" PRIu64 ".",
                     (int64_t)idx, this->n);
        return NULL;
    }
    // Quick check for an empty vector
    if (this->count == 0)
        return PyLong_FromDouble(0.0);

    uint64_t adjusted_idx;
    if (idx < 0)
    {
        adjusted_idx = (uint64_t)(this->n - idx);
    }
    else
    {
        adjusted_idx = (uint64_t)idx;
    }

    const svector_t self = {.n = this->n, .count = this->count, .capacity = 0, .entries = (entry_t *)this->entries};
    const uint64_t pos = sparse_vector_find_first_geq(&self, adjusted_idx, 0);
    if (pos != this->count || this->entries[pos].index != adjusted_idx)
        return PyFloat_FromDouble(0.0);

    return PyFloat_FromDouble(this->entries[pos].value);
}

static PyObject *svec_concatenate(PyTypeObject *type, PyObject *const *args, const Py_ssize_t nargs)
{
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(args[i], &svec_type_object))
        {
            PyErr_Format(PyExc_TypeError, "Argument %u was not a sparse vector but was instead %R.", i,
                         Py_TYPE(args[i]));
            return NULL;
        }
    }

    uint64_t n_sum = 0, cnt_sum = 0;
    const svec_object_t *const *vecs = (const svec_object_t *const *)args;
    for (unsigned i = 0; i < nargs; ++i)
    {
        n_sum += vecs[i]->n;
        cnt_sum += vecs[i]->count;
    }

    svec_object_t *const this = (svec_object_t *)type->tp_alloc(type, (Py_ssize_t)cnt_sum);
    if (!this)
        return NULL;

    this->n = n_sum;
    this->count = cnt_sum;
    uint64_t offset_n = 0, pos = 0;
    for (unsigned i = 0; i < nargs; ++i)
    {
        const svec_object_t *v = vecs[i];
        for (unsigned j = 0; j < v->count; ++j, ++pos)
        {
            this->entries[pos] = (entry_t){.index = offset_n + v->entries[j].index, .value = v->entries[j].value};
        }
        offset_n += v->n;
    }

    return (PyObject *)this;
}

static PyObject *svec_from_pairs(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "SparseVector.from_pairs takes no keyword arguments.");
        return NULL;
    }

    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 1)
    {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be provided.");
        return NULL;
    }

    // Parse first arg (element) and the rest as *pairs
    const Py_ssize_t dim_size = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, 0));
    if (PyErr_Occurred())
        return NULL;

    if (dim_size < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Vector dimension count must be positive.");
        return NULL;
    }

    svec_object_t *const self = (svec_object_t *)type->tp_alloc(type, nargs - 1);
    if (!self)
        return NULL;

    self->n = (uint64_t)dim_size;
    self->count = nargs - 1;

    // Fill arrays
    for (unsigned i = 1; i < nargs; ++i)
    {
        PyObject *const pair = PyTuple_GET_ITEM(args, i);

        if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2)
        {
            PyErr_Format(PyExc_TypeError, "Each pair must be a tuple (int, float), got %R", pair);
            Py_DECREF(self);
            return NULL;
        }
        PyObject *dof_obj = PyTuple_GET_ITEM(pair, 0);
        PyObject *coeff_obj = PyTuple_GET_ITEM(pair, 1);
        const long dof = PyLong_AsLong(dof_obj);
        const double coeff = PyFloat_AsDouble(coeff_obj);
        if (PyErr_Occurred())
        {
            Py_DECREF(self);
            return NULL;
        }
        if (dof < 0)
        {
            PyErr_SetString(PyExc_ValueError, "DoF index must be non-negative.");
            Py_DECREF(self);
            return NULL;
        }
        if (!isfinite(coeff))
        {
            PyErr_SetString(PyExc_ValueError, "Coefficient must be finite.");
            Py_DECREF(self);
            return NULL;
        }
        if (i > 1 && dof <= self->entries[i - 2].index)
        {
            PyErr_Format(PyExc_ValueError,
                         "DoF indices must be sorted in ascending order. Got %" PRId64 " after %" PRId64, dof,
                         self->entries[i - 2].index);
            Py_DECREF(self);
            return NULL;
        }
        self->entries[i - 1] = (entry_t){.index = (uint64_t)dof, .value = coeff};
    }

    return (PyObject *)self;
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
    {.ml_name = "from_pairs",
     .ml_meth = (void *)svec_from_pairs,
     .ml_flags = METH_CLASS | METH_KEYWORDS | METH_VARARGS,
     .ml_doc = "from_pairs(n: int, *pairs: tuple[int, float], /) -> SparseVector:\n"
               "Create sparse vector from an index-coefficient pairs.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n : int\n"
               "    Dimension of the vector.\n"
               "\n"
               "*pairs : tuple[int, float]\n"
               "    Pairs of values and indices for the vector.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "SparseVector\n"
               "    New vector with indices and values as given.\n"},
    {.ml_name = "concatenate",
     .ml_meth = (void *)svec_concatenate,
     .ml_flags = METH_FASTCALL | METH_CLASS,
     .ml_doc = "concatenate(*vectors: SparseVector) -> SparseVector:\n"
               "Merge sparse vectors together into a single vector.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "*vectors : SparseVector\n"
               "    Sparse vectors that should be concatenated.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Newly created sparse vector.\n"},
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

static PyObject *svec_get_count(const svec_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->count);
}

static int svec_set_n(svec_object_t *this, PyObject *val, void *Py_UNUSED(closure))
{
    const uint64_t n = PyLong_AsSize_t(val);
    if (PyErr_Occurred())
        return -1;

    if (this->count != 0 && this->entries[this->count - 1].index >= n)
    {
        PyErr_Format(PyExc_ValueError,
                     "Can not set the dimension of the vector to %" PRIu64
                     ", because the vector's last element is at index %" PRIu64 ".",
                     n, this->entries[this->count - 1].index);
        return -1;
    }
    this->n = n;
    return 0;
}

static PyGetSetDef svec_get_set[] = {
    {.name = "n",
     .get = (getter)svec_get_n,
     .set = (setter)svec_set_n,
     .doc = "int : Dimension of the vector.",
     .closure = NULL},
    {.name = "values",
     .get = (getter)svec_get_values,
     .set = NULL,
     .doc = "array : Values of non-zero entries of the vector.",
     .closure = NULL},
    {.name = "indices",
     .get = (getter)svec_get_indices,
     .set = NULL,
     .doc = "array : Indices of non-zero entries of the vector.",
     .closure = NULL},
    {.name = "count",
     .get = (getter)svec_get_count,
     .set = NULL,
     .doc = "int : Number of entries in the vector.",
     .closure = NULL},
    {}, // Sentinel
};

#ifndef Py_RETURN_BOOL
static PyObject *return_py_bool(const int v)
{
    if (v)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}
#define Py_RETURN_BOOL(v) return return_py_bool(v)
#endif // Py_RETURN_BOOL

static PyObject *svec_richcompare(const svec_object_t *self, PyObject *other, int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const int want_equal = (op == Py_EQ);
    if (Py_IS_TYPE(other, &svec_type_object))
    {
        const svec_object_t *const other_vec = (const svec_object_t *)other;

        if (self->n != other_vec->n)
        {
            Py_RETURN_BOOL(!want_equal);
        }

        uint64_t i = 0, j = 0;
        for (;;)
        {
            if (i < self->count)
            {
                if (j < other_vec->count)
                {
                    const unsigned idx1 = self->entries[i].index;
                    const unsigned idx2 = other_vec->entries[j].index;
                    if (idx1 == idx2)
                    {
                        if (self->entries[i].value != other_vec->entries[j].value)
                        {
                            Py_RETURN_BOOL(!want_equal);
                        }
                        i += 1;
                        j += 1;
                    }
                    else if (idx1 < idx2)
                    {
                        if (self->entries[i].value != 0.0)
                        {
                            Py_RETURN_BOOL(!want_equal);
                        }
                        i += 1;
                    }
                    else // idx1 > idx2
                    {
                        if (other_vec->entries[j].value != 0.0)
                        {
                            Py_RETURN_BOOL(!want_equal);
                        }
                        j += 1;
                    }
                }
                else // i < self->count && j == other_vec->count
                {
                    for (unsigned k = i; k < self->count; ++k)
                    {
                        if (self->entries[k].value != 0.0)
                        {
                            Py_RETURN_BOOL(!want_equal);
                        }
                    }
                    Py_RETURN_BOOL(want_equal);
                }
            }
            else // i == self->count && j < other_vec->count
            {
                for (unsigned k = j; k < other_vec->count; ++k)
                {
                    if (self->entries[k].value != 0.0)
                    {
                        Py_RETURN_BOOL(!want_equal);
                    }
                }
                Py_RETURN_BOOL(want_equal);
            }
        }

        ASSERT(0, "This should never be reached.");
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(other, NULL, 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
    {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (PyArray_DIM(array, 0) != self->n)
    {
        Py_DECREF(array);
        Py_RETURN_BOOL(!want_equal);
    }

    PyArrayObject *double_array = (PyArrayObject *)PyArray_Cast(array, NPY_FLOAT64);
    Py_DECREF(array);
    if (!double_array)
    {
        return NULL;
    }

    const double *const ptr = PyArray_DATA(double_array);
    uint64_t vec_pos = 0;
    for (uint64_t i = 0; i < self->n; ++i)
    {
        const double array_val = ptr[i];
        double vec_val = 0.0;
        if (vec_pos < self->count && self->entries[vec_pos].index < i)
        {
            vec_pos += 1;
        }
        if (self->entries[vec_pos].index == i)
        {
            vec_val = self->entries[vec_pos].value;
            vec_pos += 1;
        }
        if (array_val != vec_val)
        {
            Py_DECREF(double_array);
            Py_RETURN_BOOL(!want_equal);
        }
    }

    Py_DECREF(double_array);
    Py_RETURN_BOOL(want_equal);
}

static PyMappingMethods svec_mapping = {
    .mp_subscript = (binaryfunc)svec_get,
};

PyTypeObject svec_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.SparseVector",
    .tp_basicsize = sizeof(svec_object_t),
    .tp_itemsize = sizeof(entry_t),
    .tp_repr = (reprfunc)svec_repr,
    .tp_as_mapping = &svec_mapping,
    // .tp_hash = ,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    // .tp_doc = ,
    .tp_richcompare = (richcmpfunc)svec_richcompare,
    // .tp_iter = ,
    // .tp_iternext = ,
    .tp_methods = svec_methods,
    .tp_getset = svec_get_set,
};
