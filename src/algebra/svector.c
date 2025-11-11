#include "svector.h"
#include "numpy/ndarrayobject.h"

mfv2d_result_t sparse_vector_new(svector_t *this, uint64_t n, uint64_t capacity, const allocator_callbacks *allocator)
{
    entry_t *const entries = allocate(allocator, sizeof *entries * capacity);
    if (!entries)
    {
        return MFV2D_FAILED_ALLOC;
    }

    this->count = 0;
    this->n = n;
    this->capacity = capacity;
    this->entries = entries;

    return MFV2D_SUCCESS;
}

void sparse_vector_del(svector_t *this, const allocator_callbacks *allocator)
{
    deallocate(allocator, this->entries);
    *this = (svector_t){};
}

mfv2d_result_t sparse_vec_resize(svector_t *this, const uint64_t new_capacity, const allocator_callbacks *allocator)
{
    if (this->capacity >= new_capacity)
        return MFV2D_SUCCESS;

    entry_t *const new_ptr = reallocate(allocator, this->entries, sizeof(*this->entries) * new_capacity);
    if (!new_ptr)
    {
        return MFV2D_FAILED_ALLOC;
    }
    this->entries = new_ptr;
    this->capacity = new_capacity;
    return MFV2D_SUCCESS;
}

mfv2d_result_t sparse_vector_append(svector_t *this, const entry_t e, const allocator_callbacks *allocator)
{
    enum
    {
        MINIMUM_INCREMENT = 8
    };
    mfv2d_result_t res;
    if (this->count >= this->capacity &&
        (res = sparse_vec_resize(this, this->count + MINIMUM_INCREMENT, allocator)) != MFV2D_SUCCESS)
    {
        return res;
    }
    ASSERT(this->count == 0 || e.index > this->entries[this->count - 1].index,
           "Must have higher index than last in array (index comparison %u vs %u).", (unsigned)e.index,
           (unsigned)this->entries[this->count - 1].index);

    this->entries[this->count] = e;
    this->count += 1;
    return MFV2D_SUCCESS;
}

svec_object_t *sparse_vector_to_python(PyTypeObject *svec_type_object, const svector_t *this)
{
    svec_object_t *const self = (svec_object_t *)svec_type_object->tp_alloc(svec_type_object, (Py_ssize_t)this->count);
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

mfv2d_result_t sparse_vector_copy(const svector_t *src, svector_t *dst, const allocator_callbacks *allocator)
{
    dst->n = src->n;
    const mfv2d_result_t res = sparse_vec_resize(dst, src->count, allocator);
    if (res != MFV2D_SUCCESS)
        return res;

    memcpy(dst->entries, src->entries, sizeof *dst->entries * src->count);
    dst->count = src->count;

    return MFV2D_SUCCESS;
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

mfv2d_result_t sparse_vector_add_inplace(svector_t *first, const svector_t *second,
                                         const allocator_callbacks *allocator)
{
    if (second->count == 0)
        return MFV2D_SUCCESS;

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
        const mfv2d_result_t res = sparse_vec_resize(first, new_capacity, allocator);
        if (res)
            return res;
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
    return MFV2D_SUCCESS;
}

static PyObject *svec_repr(const svec_object_t *this)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(this));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(this, state->type_svec))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s but got %R.", state->type_svec->tp_name, Py_TYPE(this));
        return NULL;
    }

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

static PyObject *svec_from_entries(PyTypeObject *type, PyTypeObject *defining_class, PyObject *const *args,
                                   const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "SparseVector.from_entries() takes no keyword arguments.");
        return NULL;
    }
    const Py_ssize_t kwcnt = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (nargs + kwcnt != 3)
    {
        PyErr_Format(PyExc_TypeError, "SparseVector.from_entries() takes exactly 3 arguments, got %u.",
                     (unsigned)(nargs + kwcnt));
        return NULL;
    }
    Py_ssize_t n = -1;
    PyObject *py_indices = NULL, *py_values = NULL;
    if (nargs > 0)
    {
        n = PyLong_AsSsize_t(args[0]);
        if (PyErr_Occurred())
            return NULL;
        if (nargs > 1)
        {
            py_indices = args[1];
            if (nargs > 2)
            {
                py_values = args[2];
            }
        }
    }
    for (unsigned i = 0; i < kwcnt; ++i)
    {
        PyObject *const kwarg = PyTuple_GET_ITEM(kwnames, i);
        PyObject *const value = args[nargs + i];
        const char *keyword = PyUnicode_AsUTF8(kwarg);
        if (!keyword)
        {
            return NULL;
        }
        if (strcmp(keyword, "indices") == 0)
        {
            py_indices = value;
        }
        else if (strcmp(keyword, "values") == 0)
        {
            py_values = value;
        }
        else if (strcmp(keyword, "n") == 0)
        {
            n = PyLong_AsSsize_t(value);
            if (PyErr_Occurred())
                return NULL;
        }
        else
        {
            PyErr_Format(PyExc_TypeError, "SparseVector.from_pairs() got an unexpected keyword argument '%s'.",
                         keyword);
            return NULL;
        }
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

    for (uint64_t i = 1; i < (uint64_t)count; ++i)
    {
        if (pi[i - 1] >= pi[i])
        {
            PyErr_Format(PyExc_ValueError, "Entry indices at %" PRIu64 " and at %" PRIu64 " are not sorted.", i, i + 1);
            Py_DECREF(array_values);
            Py_DECREF(array_indices);
            return NULL;
        }
    }
    for (uint64_t i = 0; i < (uint64_t)count; ++i)
    {
        if (pi[i] >= (uint64_t)n)
        {
            PyErr_Format(PyExc_ValueError,
                         "Entry index at %" PRIu64 " with value %" PRIu64 " is outside the allowed range [0, %" PRIu64
                         ").",
                         i, pi[i], (uint64_t)n);
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

    for (uint64_t i = 0; i < (uint64_t)count; ++i)
    {
        this->entries[i] = (entry_t){.index = pi[i], .value = pv[i]};
    }
    Py_DECREF(array_values);
    Py_DECREF(array_indices);

    return (PyObject *)this;
}

static PyObject *svec_array(PyObject *self, PyTypeObject *defining_class, PyObject *const *args, const Py_ssize_t nargs,
                            PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
    {
        return NULL;
    }

    if (!PyObject_TypeCheck(self, state->type_svec))
    {
        PyErr_Format(PyExc_TypeError, "SparseVector.__array__ must be called on a %s but got %R.",
                     state->type_svec->tp_name, self);
        return NULL;
    }

    const Py_ssize_t nkwds = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;

    if (nargs + nkwds > 2)
    {
        PyErr_Format(PyExc_TypeError, "SparseVector.__array__ takes at most two arguments, got %u.",
                     (unsigned)(nargs + nkwds));
        return NULL;
    }
    PyArray_Descr *dtype = NULL;
    int b_copy = 0;

    if (nargs > 0)
    {
        dtype = (PyArray_Descr *)args[0];
        if (nargs > 1)
        {
            b_copy = PyObject_IsTrue(args[1]);
            if (PyErr_Occurred())
            {
                return NULL;
            }
        }
    }

    for (unsigned i = 0; i < nkwds; ++i)
    {
        PyObject *const kwarg = PyTuple_GET_ITEM(kwnames, i);
        PyObject *const value = args[nargs + i];
        const char *keyword = PyUnicode_AsUTF8(kwarg);
        if (!keyword)
        {
            return NULL;
        }
        if (strcmp(keyword, "dtype") == 0)
        {
            dtype = (PyArray_Descr *)value;
        }
        else if (strcmp(keyword, "copy") == 0)
        {
            b_copy = PyObject_IsTrue(value);
            if (PyErr_Occurred())
            {
                return NULL;
            }
        }
        else
        {
            PyErr_Format(PyExc_TypeError, "SparseVector.__array__ got an unexpected keyword argument '%s'.", keyword);
            return NULL;
        }
    }

    if (!b_copy)
    {
        PyErr_SetString(PyExc_ValueError, "A copy is always created when converting to NDArray.");
        return NULL;
    }

    const svec_object_t *const this = (svec_object_t *)self;
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
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(this));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(this, state->type_svec))
    {
        PyErr_Format(PyExc_TypeError, "SparseVector.__get__ must be called on a %s but got %R.",
                     state->type_svec->tp_name, this);
        return NULL;
    }

    if (PySlice_Check(py_idx))
    {
        // Quick check for an empty vector
        if (this->count == 0)
        {
            const svector_t v = {.n = this->n, .count = 0, .capacity = 0, .entries = NULL};
            return (PyObject *)sparse_vector_to_python(state->type_svec, &v);
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
            return (PyObject *)sparse_vector_to_python(state->type_svec, &fake);
        }
        const uint64_t end = sparse_vector_find_first_geq(&self, stop, begin);
        fake = (svector_t){
            .n = seq_len, .count = end - begin, .capacity = 0, .entries = (entry_t *)(this->entries + begin)};
        svec_object_t *const vec = sparse_vector_to_python(state->type_svec, &fake);
        for (uint64_t i = 0; i < vec->count; ++i)
        {
            vec->entries[i].index -= start;
        }
        return (PyObject *)vec;
    }

    const Py_ssize_t idx = PyNumber_AsSsize_t(py_idx, PyExc_IndexError);
    if (PyErr_Occurred())
        return NULL;

    uint64_t adjusted_idx;
    if (idx < 0)
    {
        adjusted_idx = (uint64_t)(this->n - idx);
    }
    else
    {
        adjusted_idx = (uint64_t)idx;
    }

    if (adjusted_idx >= this->n)
    {
        PyErr_Format(PyExc_KeyError,
                     "Index %" PRIi64 " is outside the allowed range for a vector of dimension %" PRIu64 ".",
                     (int64_t)idx, this->n);
        return NULL;
    }

    // Quick check for an empty vector
    if (this->count == 0)
        return PyLong_FromDouble(0.0);

    const svector_t self = {.n = this->n, .count = this->count, .capacity = 0, .entries = (entry_t *)this->entries};
    const uint64_t pos = sparse_vector_find_first_geq(&self, adjusted_idx, 0);
    if (pos != this->count || this->entries[pos].index != adjusted_idx)
        return PyFloat_FromDouble(0.0);

    return PyFloat_FromDouble(this->entries[pos].value);
}

static PyObject *svec_concatenate(PyTypeObject *type, PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "SparseVector.concatenate() takes no keyword arguments.");
        return NULL;
    }

    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(args[i], state->type_svec))
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

static PyObject *svec_from_pairs(PyTypeObject *type, PyTypeObject *defining_class, PyObject *const *args,
                                 const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "SparseVector.from_pairs() takes no keyword arguments.");
        return NULL;
    }

    if (nargs < 1)
    {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be provided.");
        return NULL;
    }

    // Parse first arg (element) and the rest as *pairs
    const Py_ssize_t dim_size = PyLong_AsSsize_t(args[0]);
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
        PyObject *const pair = args[i];

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
        if (i > 1 && (uint64_t)dof <= self->entries[i - 2].index)
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

static PyObject *svec_dot(PyObject *self, PyTypeObject *defining_class, PyObject *const *args, const Py_ssize_t nargs,
                          PyObject *kwnames)
{
    mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (nargs != 1)
    {
        PyErr_Format(PyExc_TypeError, "Method requires exactly one argument, got %u.", (unsigned)nargs);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Method takes no keyword arguments.");
        return NULL;
    }
    svec_object_t *const this = (svec_object_t *)self;
    svec_object_t *const that = (svec_object_t *)args[0];

    if (!PyObject_TypeCheck(that, state->type_svec) || !PyObject_TypeCheck(this, state->type_svec))
    {
        PyErr_Format(PyExc_TypeError, "Method requires two sparse vectors, but got a %R and %R.", Py_TYPE(this),
                     Py_TYPE(that));
        return NULL;
    }
    if (this->n != that->n)
    {
        PyErr_Format(PyExc_ValueError, "Can not compute dot product of vectors of different dimensions (%u and %u).",
                     (unsigned)this->n, (unsigned)that->n);
        return NULL;
    }

    double dot_product = 0.0;
    unsigned p1 = 0, p2 = 0;
    while (p1 < this->count && p2 < that->count)
    {
        if (this->entries[p1].index < that->entries[p2].index)
        {
            p1 += 1;
        }
        else if (this->entries[p1].index > that->entries[p2].index)
        {
            p2 += 1;
        }
        else // (this->entries[p1].index == that->entries[p2].index)
        {
            dot_product += this->entries[p1].value * that->entries[p2].value;
            p1 += 1;
            p2 += 1;
        }
    }

    return PyFloat_FromDouble(dot_product);
}

typedef enum
{
    MERGE_MODE_FIRST,
    MERGE_MODE_LAST,
    MERGE_MODE_SUM,
    MERGE_MODE_ERROR,
} svec_merge_mode_t;

static int svec_parse_merge_mode(const char name[], svec_merge_mode_t *out)
{
    if (strcmp(name, "first") == 0)
    {
        *out = MERGE_MODE_FIRST;
        return 0;
    }
    if (strcmp(name, "last") == 0)
    {
        *out = MERGE_MODE_LAST;
        return 0;
    }
    if (strcmp(name, "sum") == 0)
    {
        *out = MERGE_MODE_SUM;
        return 0;
    }
    if (strcmp(name, "error") == 0)
    {
        *out = MERGE_MODE_ERROR;
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "Invalid merge mode '%s'.", name);
    return -1;
}

static PyObject *svec_merge_to_dense(PyTypeObject *Py_UNUSED(type), PyTypeObject *defining_class, PyObject *const *args,
                                     const Py_ssize_t nargs, PyObject *kwnames)
{
    svec_merge_mode_t merge_mode = MERGE_MODE_LAST;
    if (kwnames != NULL)
    {
        const Py_ssize_t kwnames_len = PyTuple_GET_SIZE(kwnames);
        if (kwnames_len != 1)
        {
            PyErr_Format(PyExc_TypeError, "Expected exactly one keyword argument, got %u.", kwnames_len);
            return NULL;
        }

        PyObject *const kwarg = PyTuple_GET_ITEM(kwnames, 0);
        if (!PyUnicode_Check(kwarg))
        {
            PyErr_Format(PyExc_TypeError, "Expected keyword argument to be a string, got %R.", kwarg);
            return NULL;
        }

        const char *const kwarg_str = PyUnicode_AsUTF8(kwarg);
        if (!kwarg_str)
        {
            return NULL;
        }
        if (strcmp(kwarg_str, "duplicates") != 0)
        {
            PyErr_Format(PyExc_TypeError, "Only valid keyword is \"duplicates\", but \"%s\" was given.", kwarg_str);
            return NULL;
        }

        if (!PyUnicode_Check(args[nargs]))
        {
            PyErr_Format(PyExc_TypeError, "Expected keyword value string, got %R.", args[nargs]);
            return NULL;
        }

        const char *const kwarg_value = PyUnicode_AsUTF8(args[nargs]);
        if (!kwarg_value)
        {
            return NULL;
        }
        if (svec_parse_merge_mode(kwarg_value, &merge_mode) < 0)
        {
            PyErr_Format(PyExc_ValueError, "Invalid merge mode '%s'.", kwarg_value);
            return NULL;
        }
    }

    if (nargs == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at least one sparse vector.");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!PyObject_TypeCheck(args[i], state->type_svec))
        {
            PyErr_Format(PyExc_TypeError, "Argument %u was not a sparse vector but was instead %R.", i,
                         Py_TYPE(args[i]));
            return NULL;
        }
    }
    const svec_object_t *const *vecs = (const svec_object_t *const *)args;
    const unsigned n = vecs[0]->n;
    for (unsigned i = 1; i < nargs; ++i)
    {
        if (vecs[i]->n != n)
        {
            PyErr_Format(
                PyExc_ValueError,
                "All sparse vectors must have the same shape (first had %u), but vector %u did not match (had %u).", n,
                i, (unsigned)vecs[i]->n);
            return NULL;
        }
    }

    if (nargs == 1)
    {
        return PyObject_CallMethod(args[0], "__array__", NULL);
    }

    const npy_intp size = n;
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_FLOAT64);

    if (!array)
        return NULL;

    npy_float64 *const ptr = PyArray_DATA(array);
    memset(ptr, 0, size * sizeof(npy_float64));
    switch (merge_mode)
    {
    case MERGE_MODE_LAST:
        for (unsigned i = 0; i < nargs; ++i)
        {
            const svec_object_t *const v = vecs[i];
            for (unsigned j = 0; j < v->count; ++j)
            {
                if (v->entries[j].value == 0.0)
                    continue;

                ptr[v->entries[j].index] = v->entries[j].value;
            }
        }
        break;

    case MERGE_MODE_FIRST:
        for (unsigned i = nargs; i > 0; --i)
        {
            const svec_object_t *const v = vecs[i - 1];
            for (unsigned j = 0; j < v->count; ++j)
            {
                if (v->entries[j].value == 0.0)
                    continue;

                ptr[v->entries[j].index] = v->entries[j].value;
            }
        }
        break;

    case MERGE_MODE_SUM:
        for (unsigned i = 0; i < nargs; ++i)
        {
            const svec_object_t *const v = vecs[i];
            for (unsigned j = 0; j < v->count; ++j)
            {
                if (v->entries[j].value == 0.0)
                    continue;

                ptr[v->entries[j].index] += v->entries[j].value;
            }
        }
        break;

    case MERGE_MODE_ERROR:
        for (unsigned i = 0; i < nargs; ++i)
        {
            const svec_object_t *const v = vecs[i];
            for (unsigned j = 0; j < v->count; ++j)
            {
                if (v->entries[j].value == 0.0)
                    continue;
                if (ptr[v->entries[j].index] != 0.0)
                {
                    PyErr_Format(PyExc_ValueError, "Duplicate entry at index %u for vector %u", j, i);
                    Py_DECREF(array);
                    return NULL;
                }
                ptr[v->entries[j].index] += v->entries[j].value;
            }
        }
        break;
    }

    return (PyObject *)array;
}

static PyMethodDef svec_methods[] = {
    {
        .ml_name = "__array__",
        .ml_meth = (void *)svec_array,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray[int]\n"
                  "Convert to numpy array.\n",
    },
    {
        .ml_name = "from_entries",
        .ml_meth = (void *)svec_from_entries,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
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
                  "    New vector with indices and values as given.\n",
    },
    {
        .ml_name = "from_pairs",
        .ml_meth = (void *)svec_from_pairs,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
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
                  "    New vector with indices and values as given.\n",
    },
    {
        .ml_name = "concatenate",
        .ml_meth = (void *)svec_concatenate,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD | METH_CLASS,
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
                  "    Newly created sparse vector.\n",
    },
    {
        .ml_name = "dot",
        .ml_meth = (void *)svec_dot,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "dot(other: SparseVector) -> float\n"
                  "Compute dot product of two sparse vectors.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "other : SparseVector\n"
                  "    The sparse vector with which to take the dot product with. Its dimension\n"
                  "    must match exactly.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "float\n"
                  "    Dot product of the two sparse vectors.\n",
    },
    {
        .ml_name = "merge_to_dense",
        .ml_meth = (void *)svec_merge_to_dense,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD | METH_CLASS,
        .ml_doc =
            "merge_to_dense(*args: SparseVector, duplicates: typing.Literal[\"first\", \"last\", \"sum\", \"error\"] = "
            "\"first\")\n"
            "Merge sparse vectors into a single dense vector.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "vecs : SparseVector\n"
            "    Sparse vectors that should be merged together. All must have the exact same\n"
            "    size.\n"
            "\n"
            "duplicates : \"first\", \"last\", \"sum\", or \"error\", default: \"last\"\n"
            "    What value to use when encountering duplicates.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "array\n"
            "    Full array with all the entries of vectors combined.\n",
    },
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

static PyObject *svec_l2norm2(const svec_object_t *this, void *Py_UNUSED(closure))
{
    double norm = 0.0;
    for (uint64_t i = 0; i < this->count; ++i)
    {
        norm += this->entries[i].value * this->entries[i].value;
    }
    return PyFloat_FromDouble(norm);
}

static PyGetSetDef svec_get_set[] = {
    {
        .name = "n",
        .get = (getter)svec_get_n,
        .set = (setter)svec_set_n,
        .doc = "int : Dimension of the vector.",
        .closure = NULL,
    },
    {
        .name = "values",
        .get = (getter)svec_get_values,
        .set = NULL,
        .doc = "array : Values of non-zero entries of the vector.",
        .closure = NULL,
    },
    {
        .name = "indices",
        .get = (getter)svec_get_indices,
        .set = NULL,
        .doc = "array : Indices of non-zero entries of the vector.",
        .closure = NULL,
    },
    {
        .name = "count",
        .get = (getter)svec_get_count,
        .set = NULL,
        .doc = "int : Number of entries in the vector.",
        .closure = NULL,
    },
    {
        .name = "norm2",
        .get = (getter)svec_l2norm2,
        .set = NULL,
        .doc = "float : Square of the L2 norm.",
        .closure = NULL,
    },
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
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(other, state->type_svec) || !PyObject_TypeCheck(self, state->type_svec))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const int want_equal = (op == Py_EQ);
    if (Py_IS_TYPE(other, state->type_svec))
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

    if (PyArray_DIM(array, 0) != (npy_intp)self->n)
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

// static PyMappingMethods svec_mapping = {
//     .mp_subscript = (binaryfunc)svec_get,
// };

static PyObject *svec_add(const svec_object_t *self, const svec_object_t *other)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyObject_TypeCheck(self, state->type_svec) || !PyObject_TypeCheck(other, state->type_svec))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const svector_t this_vector = {.n = self->n, .entries = (entry_t *)self->entries, .count = self->count};
    const svector_t that_vector = {.n = other->n, .entries = (entry_t *)other->entries, .count = other->count};

    svector_t sum_vector;
    mfv2d_result_t res = sparse_vector_copy(&this_vector, &sum_vector, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
        return NULL;

    svec_object_t *out = NULL;
    res = sparse_vector_add_inplace(&sum_vector, &that_vector, &SYSTEM_ALLOCATOR);
    if (res == MFV2D_SUCCESS)
    {
        out = sparse_vector_to_python(state->type_svec, &sum_vector);
    }
    sparse_vector_del(&sum_vector, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

static PyObject *svec_sub(const svec_object_t *self, const svec_object_t *other)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (!PyObject_TypeCheck(self, state->type_svec) || !PyObject_TypeCheck(other, state->type_svec))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const svector_t this_vector = {.n = self->n, .entries = (entry_t *)self->entries, .count = self->count};
    const svector_t that_vector = {.n = other->n, .entries = (entry_t *)other->entries, .count = other->count};

    svector_t sum_vector;
    mfv2d_result_t res = sparse_vector_copy(&that_vector, &sum_vector, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
    {
        return NULL;
    }

    // Negate this vector
    for (unsigned i = 0; i < sum_vector.count; ++i)
    {
        sum_vector.entries[i].value = -sum_vector.entries[i].value;
    }

    svec_object_t *out = NULL;
    res = sparse_vector_add_inplace(&sum_vector, &this_vector, &SYSTEM_ALLOCATOR);
    if (res == MFV2D_SUCCESS)
    {
        out = sparse_vector_to_python(state->type_svec, &sum_vector);
    }
    sparse_vector_del(&sum_vector, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

static PyObject *svec_mul(const svec_object_t *self, PyObject *other)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }
    const double v = PyFloat_AsDouble(other);
    if (PyErr_Occurred())
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const svector_t this_vector = {.n = self->n, .entries = (entry_t *)self->entries, .count = self->count};

    svector_t prod_vector = {};

    const mfv2d_result_t res = sparse_vector_copy(&this_vector, &prod_vector, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
        return NULL;

    for (unsigned i = 0; i < prod_vector.count; ++i)
    {
        prod_vector.entries[i].value *= v;
    }

    svec_object_t *const out = sparse_vector_to_python(state->type_svec, &prod_vector);
    sparse_vector_del(&prod_vector, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

static PyType_Slot svec_type_slots[] = {
    {.slot = Py_tp_repr, .pfunc = svec_repr},
    {.slot = Py_mp_subscript, .pfunc = svec_get},
    {.slot = Py_tp_richcompare, .pfunc = svec_richcompare},
    {.slot = Py_tp_methods, .pfunc = svec_methods},
    {.slot = Py_tp_getset, .pfunc = svec_get_set},
    {.slot = Py_nb_add, .pfunc = svec_add},
    {.slot = Py_nb_subtract, .pfunc = svec_sub},
    {.slot = Py_nb_multiply, .pfunc = svec_mul},
    {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
    {}, // sentinel
};

PyType_Spec svec_type_spec = {
    .name = "mfv2d._mfv2d.SparseVector",
    .basicsize = sizeof(svec_object_t),
    .itemsize = sizeof(entry_t),
    .flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots = svec_type_slots,
};
