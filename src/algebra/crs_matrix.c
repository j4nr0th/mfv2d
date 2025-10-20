#include "crs_matrix.h"
#include "../common/error.h"
#include "svector.h"

static void *alloc_wrap(void *state, uint64_t size)
{
    return allocate(state, size);
}

static void dealloc_wrap(void *state, void *ptr)
{
    deallocate(state, ptr);
}

static void *realloc_wrap(void *state, void *ptr, size_t new_size)
{
    return reallocate(state, ptr, new_size);
}

const static jmtx_allocator_callbacks JMTX_ALLOCATOR = {
    .alloc = alloc_wrap,
    .realloc = realloc_wrap,
    .free = dealloc_wrap,
    .state = &SYSTEM_ALLOCATOR,
};

static int crs_matrix_check_build(const crs_matrix_t *this)
{
    if (this->built_rows != this->matrix->base.rows)
    {
        PyErr_Format(PyExc_RuntimeError, "Matrix has not been built for all rows");
        return 0;
    }
    return 1;
}

static PyObject *crs_matrix_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned rows, cols;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "II", (char *[]){"rows", "cols", NULL}, &rows, &cols))
    {
        return NULL;
    }
    crs_matrix_t *const this = (crs_matrix_t *)type->tp_alloc(type, 0);
    if (!this)
    {
        return NULL;
    }

    if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_new(&this->matrix, rows, cols, 32, &JMTX_ALLOCATOR)))
    {
        Py_DECREF(this);
        return NULL;
    }
    this->built_rows = 0;
    return (PyObject *)this;
}

static void crs_matrix_del(crs_matrix_t *this)
{
    jmtxd_matrix_crs_destroy(this->matrix);
    this->matrix = NULL;
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *crs_matrix_build_row(crs_matrix_t *this, PyObject *args, PyObject *kwargs)
{
    unsigned row;
    svec_object_t *entries = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|O!", (char *[]){"row", "entries", NULL}, &row, &svec_type_object,
                                     &entries))
    {
        return NULL;
    }
    if (row >= this->matrix->base.rows)
    {
        PyErr_Format(PyExc_ValueError, "Row index %u out of bounds for matrix of dimensions (%u, %u)", row,
                     this->matrix->base.rows, this->matrix->base.cols);
        return NULL;
    }

    if (row > this->built_rows)
    {
        PyErr_Format(PyExc_ValueError, "Row index %u is greater than the number of rows built (%u)", row,
                     this->built_rows);
        return NULL;
    }
    this->built_rows = row + 1;

    if (entries == NULL)
    {
        (void)jmtxd_matrix_crs_build_row(this->matrix, row, 0, (uint32_t[1]){0}, (double[1]){0});
        Py_RETURN_NONE;
    }
    if (entries->n != this->matrix->base.cols)
    {
        PyErr_Format(PyExc_ValueError, "Dimension of the sparse vector (%u) does not match number of columns (%u)",
                     entries->n, this->matrix->base.cols);
        return NULL;
    }

    uint32_t *indices = allocate(&SYSTEM_ALLOCATOR, entries->count * sizeof *indices);
    if (!indices)
    {
        return NULL;
    }
    double *values = allocate(&SYSTEM_ALLOCATOR, entries->count * sizeof *values);
    if (!values)
    {
        deallocate(&SYSTEM_ALLOCATOR, indices);
        return NULL;
    }

    for (unsigned i = 0; i < entries->count; ++i)
    {
        indices[i] = entries->entries[i].index;
        values[i] = entries->entries[i].value;
    }

    const int success = JMTX_SUCCEEDED(jmtxd_matrix_crs_build_row(this->matrix, row, entries->count, indices, values));
    deallocate(&SYSTEM_ALLOCATOR, indices);
    deallocate(&SYSTEM_ALLOCATOR, values);
    if (!success)
    {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *crs_matrix_toarray(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;
    const unsigned cols = this->matrix->base.cols;
    const unsigned rows = this->matrix->base.rows;
    const npy_intp dims[2] = {rows, cols};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array)
        return NULL;
    npy_double *const ptr = PyArray_DATA(array);
    memset(ptr, 0, sizeof *ptr * rows * this->matrix->base.cols);
    for (unsigned r = 0; r < rows; ++r)
    {
        uint32_t *indices = NULL;
        double *values = NULL;
        const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, r, &indices, &values);
        for (unsigned i = 0; i < n; ++i)
        {
            ptr[r * cols + indices[i]] = values[i];
        }
    }

    return (PyObject *)array;
}

static PyObject *crs_matrix_matmul_left_svec(const crs_matrix_t *this, const svec_object_t *other)
{
    // if (!crs_matrix_check_build(this))
    //     return NULL;
    //
    if (this->matrix->base.cols != other->n)
    {
        PyErr_Format(PyExc_ValueError, "Dimension of the sparse vector does not match number of columns");
        return NULL;
    }

    svector_t svec;
    const unsigned rows = this->matrix->base.rows;
    if (sparse_vector_new(&svec, rows, other->count > rows ? rows : other->count, &SYSTEM_ALLOCATOR))
    {
        return NULL;
    }

    for (unsigned r = 0; r < rows; ++r)
    {
        uint32_t *indices = NULL;
        double *values = NULL;
        const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, r, &indices, &values);
        double v = 0;
        unsigned i1 = 0, i2 = 0;
        while (i1 < n && i2 < other->count)
        {
            while (indices[i1] > other->entries[i2].index && i2 < other->count)
            {
                i2 += 1;
            }
            if (i2 == other->count)
            {
                break;
            }
            while (indices[i1] < other->entries[i2].index && i1 < n)
            {
                i1 += 1;
            }
            if (i1 == n)
            {
                break;
            }
            if (indices[i1] == other->entries[i2].index)
            {
                v += values[i1] * other->entries[i2].value;
                i1 += 1;
                i2 += 1;
            }
        }
        if (v != 0.0 && sparse_vector_append(&svec, (entry_t){.index = r, .value = v}, &SYSTEM_ALLOCATOR))
        {
            sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
            raise_exception_from_current(PyExc_RuntimeError, "Failed to append to sparse vector");
            return NULL;
        }
    }

    svec_object_t *out = sparse_vector_to_python(&svec);
    sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

static PyObject *crs_matrix_matmul_right_svec(const crs_matrix_t *this, const svec_object_t *other)
{
    // if (!crs_matrix_check_build(this))
    //     return NULL;
    //
    if (this->matrix->base.rows != other->n)
    {
        PyErr_Format(PyExc_ValueError, "Dimension of the sparse vector does not match number of rows");
        return NULL;
    }

    svector_t svec = {};
    svector_t temporary = {};
    const unsigned cols = this->matrix->base.cols;
    if (sparse_vector_new(&svec, cols, other->count > cols ? cols : other->count, &SYSTEM_ALLOCATOR))
    {
        return NULL;
    }
    if (sparse_vector_new(&temporary, cols, other->count > cols ? cols : other->count, &SYSTEM_ALLOCATOR))
    {
        sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
        return NULL;
    }

    for (unsigned i = 0; i < other->count; ++i)
    {
        const entry_t *const e = other->entries + i;
        if (e->value == 0.0)
            continue;
        temporary.count = 0;
        uint32_t *indices = NULL;
        double *values = NULL;
        const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, e->index, &indices, &values);
        for (unsigned j = 0; j < n; ++j)
        {
            const double v = values[j] * e->value;
            if (v != 0.0 &&
                sparse_vector_append(&temporary, (entry_t){.index = indices[j], .value = v}, &SYSTEM_ALLOCATOR))
            {
                sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
                sparse_vector_del(&temporary, &SYSTEM_ALLOCATOR);
                raise_exception_from_current(PyExc_RuntimeError, "Failed to append to sparse vector");
                return NULL;
            }
        }

        if (temporary.count > 0 && sparse_vector_add_inplace(&svec, &temporary, &SYSTEM_ALLOCATOR))
        {
            sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
            sparse_vector_del(&temporary, &SYSTEM_ALLOCATOR);
            raise_exception_from_current(PyExc_RuntimeError, "Failed to add sparse vectors");
            return NULL;
        }
    }

    sparse_vector_del(&temporary, &SYSTEM_ALLOCATOR);

    svec_object_t *out = sparse_vector_to_python(&svec);
    sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

static crs_matrix_t *crs_matrix_matmul_double(const crs_matrix_t *this, const crs_matrix_t *that)
{
    if (!crs_matrix_check_build(this))
        return NULL;
    if (!crs_matrix_check_build(that))
        return NULL;
    if (this->matrix->base.cols != that->matrix->base.rows)
    {
        PyErr_Format(PyExc_ValueError,
                     "Matrix multiplication requires that the number of columns of the left matrix "
                     "(%u) is equal to the number of rows of the right matrix (%u).",
                     this->matrix->base.cols, that->matrix->base.rows);
        return NULL;
    }

    crs_matrix_t *const new = (crs_matrix_t *)crs_matrix_type_object.tp_alloc(&crs_matrix_type_object, 0);
    if (!new)
        return NULL;

    const unsigned out_rows = this->matrix->base.rows;
    const unsigned out_cols = that->matrix->base.cols;

    if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_new(
            &new->matrix, out_rows, out_cols,
            this->matrix->n_entries > that->matrix->n_entries ? this->matrix->n_entries : that->matrix->n_entries,
            &JMTX_ALLOCATOR)))
    {
        Py_DECREF(new);
        return NULL;
    }

    unsigned out_buffer_capacity = 1 + new->matrix->capacity / out_rows;
    uint32_t *out_indices = allocate(&SYSTEM_ALLOCATOR, out_buffer_capacity * sizeof(*out_indices));
    double *out_values = allocate(&SYSTEM_ALLOCATOR, out_buffer_capacity * sizeof(*out_values));
    if (!out_indices || !out_values)
    {
        if (out_indices)
            deallocate(&SYSTEM_ALLOCATOR, out_indices);
        if (out_values)
            deallocate(&SYSTEM_ALLOCATOR, out_values);
        Py_DECREF(new);
        return NULL;
    }
    svector_t row_built = {}, temp = {};
    if (sparse_vector_new(&row_built, out_cols, out_buffer_capacity, &SYSTEM_ALLOCATOR) ||
        sparse_vector_new(&temp, out_cols, out_buffer_capacity, &SYSTEM_ALLOCATOR))
    {
        sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
        sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
        deallocate(&SYSTEM_ALLOCATOR, out_indices);
        deallocate(&SYSTEM_ALLOCATOR, out_values);
        Py_DECREF(new);
        return NULL;
    }

    // Build rows one at a time
    for (unsigned r = 0; r < out_rows; ++r)
    {
        row_built.count = 0;
        uint32_t *indices_2 = NULL;
        double *values_2 = NULL;
        const unsigned n_2 = jmtxd_matrix_crs_get_row(this->matrix, r, &indices_2, &values_2);
        for (unsigned i = 0; i < n_2; ++i)
        {
            const uint32_t row_index = indices_2[i];
            const double row_value = values_2[i];

            uint32_t *indices_1 = NULL;
            double *values_1 = NULL;
            temp.count = 0;
            const unsigned n_1 = jmtxd_matrix_crs_get_row(that->matrix, row_index, &indices_1, &values_1);
            for (unsigned j = 0; j < n_1; ++j)
            {
                const double v = values_1[j] * row_value;
                if (v != 0 &&
                    sparse_vector_append(&temp, (entry_t){.index = indices_1[j], .value = v}, &SYSTEM_ALLOCATOR))
                {
                    PyErr_Format(PyExc_RuntimeError, "Failed to append to sparse vector.");
                    sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
                    sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
                    deallocate(&SYSTEM_ALLOCATOR, out_indices);
                    deallocate(&SYSTEM_ALLOCATOR, out_values);
                    Py_DECREF(new);
                    return NULL;
                }
            }

            if (sparse_vector_add_inplace(&row_built, &temp, &SYSTEM_ALLOCATOR))
            {
                PyErr_Format(PyExc_RuntimeError, "Failed to add sparse vectors.");
                sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
                sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
                deallocate(&SYSTEM_ALLOCATOR, out_indices);
                deallocate(&SYSTEM_ALLOCATOR, out_values);
                Py_DECREF(new);
                return NULL;
            }
        }

        if (row_built.count > out_buffer_capacity)
        {
            out_buffer_capacity = row_built.count;
            uint32_t *const new_indices =
                reallocate(&SYSTEM_ALLOCATOR, out_indices, out_buffer_capacity * sizeof(*out_indices));
            if (!new_indices)
            {
                sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
                sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
                deallocate(&SYSTEM_ALLOCATOR, out_indices);
                deallocate(&SYSTEM_ALLOCATOR, out_values);
                Py_DECREF(new);
                return NULL;
            }
            out_indices = new_indices;
            double *const new_values =
                reallocate(&SYSTEM_ALLOCATOR, out_values, out_buffer_capacity * sizeof(*out_values));
            if (!new_values)
            {
                sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
                sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
                deallocate(&SYSTEM_ALLOCATOR, out_indices);
                deallocate(&SYSTEM_ALLOCATOR, out_values);
                Py_DECREF(new);
                return NULL;
            }
            out_values = new_values;
        }

        for (unsigned i = 0; i < row_built.count; ++i)
        {
            out_values[i] = row_built.entries[i].value;
            out_indices[i] = row_built.entries[i].index;
        }

        if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_build_row(new->matrix, r, row_built.count, out_indices, out_values)))
        {
            sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
            sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
            deallocate(&SYSTEM_ALLOCATOR, out_indices);
            deallocate(&SYSTEM_ALLOCATOR, out_values);
            Py_DECREF(new);
            return NULL;
        }
    }

    new->built_rows = out_rows;
    sparse_vector_del(&temp, &SYSTEM_ALLOCATOR);
    sparse_vector_del(&row_built, &SYSTEM_ALLOCATOR);
    deallocate(&SYSTEM_ALLOCATOR, out_indices);
    deallocate(&SYSTEM_ALLOCATOR, out_values);
    return new;
}

static PyObject *crs_matrix_matmul_left(const crs_matrix_t *this, PyObject *other)
{
    if (!crs_matrix_check_build(this))
        return NULL;
    if (PyObject_TypeCheck(other, &svec_type_object))
    {
        return crs_matrix_matmul_left_svec(this, (svec_object_t *)other);
    }
    if (PyObject_TypeCheck(other, &crs_matrix_type_object))
    {
        return (PyObject *)crs_matrix_matmul_double(this, (crs_matrix_t *)other);
    }
    PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(other, PyArray_DescrFromType(NPY_DOUBLE), 1, 2,
                                                                  NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!array)
        return NULL;
    npy_intp in_dims[2];
    in_dims[0] = PyArray_DIM(array, 0);
    if (PyArray_NDIM(array) == 2)
    {
        in_dims[1] = PyArray_DIM(array, 1);
    }
    else
    {
        in_dims[1] = 1;
    }
    const npy_intp rows = (npy_intp)this->matrix->base.rows;
    const npy_intp cols = (npy_intp)this->matrix->base.cols;
    if (in_dims[0] != this->matrix->base.cols)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Matrix has the shape of (%u, %u), so it can only be left multiplied by arrays with %u rows, instead "
            "the argument had the shape of (%u, %u).",
            rows, cols, rows, in_dims[0], in_dims[1]);
        Py_DECREF(array);
        return NULL;
    }

    const npy_intp out_dims[2] = {rows, in_dims[1]};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(array), out_dims, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS;
    npy_double *const out_ptr = PyArray_DATA(out);
    memset(out_ptr, 0, sizeof *out_ptr * out_dims[0] * out_dims[1]);
    const npy_double *in_ptr = PyArray_DATA(array);

    for (unsigned c = 0; c < out_dims[1]; ++c)
    {
        for (unsigned r = 0; r < out_dims[0]; ++r)
        {
            uint32_t *indices = NULL;
            double *values = NULL;
            const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, r, &indices, &values);
            double v = 0;
            for (unsigned i = 0; i < n; ++i)
            {
                v += values[i] * in_ptr[in_dims[1] * indices[i] + c];
            }
            out_ptr[r * out_dims[1] + c] = v;
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(array);
    return (PyObject *)out;
}

static PyObject *crs_matrix_matmul_right(const crs_matrix_t *this, PyObject *other)
{
    if (!crs_matrix_check_build(this))
        return NULL;

    if (PyObject_TypeCheck(other, &svec_type_object))
    {
        return crs_matrix_matmul_right_svec(this, (svec_object_t *)other);
    }
    if (PyObject_TypeCheck(other, &crs_matrix_type_object))
    {
        return (PyObject *)crs_matrix_matmul_double((crs_matrix_t *)other, this);
    }

    PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(other, PyArray_DescrFromType(NPY_DOUBLE), 1, 2,
                                                                  NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!array)
        return NULL;
    npy_intp in_dims[2];
    const int nd = PyArray_NDIM(array);
    in_dims[1] = PyArray_DIM(array, nd - 1);
    if (nd == 2)
    {
        in_dims[0] = PyArray_DIM(array, 0);
    }
    else
    {
        in_dims[0] = 1;
    }
    const npy_intp rows = (npy_intp)this->matrix->base.rows;
    const npy_intp cols = (npy_intp)this->matrix->base.cols;
    if (in_dims[1] != this->matrix->base.rows)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Matrix has the shape of (%u, %u), so it can only be right multiplied by arrays with %u columns, instead "
            "the argument had the shape of (%u, %u).",
            rows, cols, rows, in_dims[0], in_dims[1]);
        Py_DECREF(array);
        return NULL;
    }
    const npy_intp out_dims[2] = {in_dims[0], cols};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(nd, out_dims + (nd == 1), NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    npy_double *const out_ptr = PyArray_DATA(out);
    memset(out_ptr, 0, sizeof *out_ptr * out_dims[0] * out_dims[1]);

    const npy_double *in_ptr = PyArray_DATA(array);

    for (unsigned row = 0; row < out_dims[0]; ++row)
    {
        for (unsigned col = 0; col < in_dims[1]; ++col)
        {
            const double v = in_ptr[row * in_dims[1] + col];
            if (v == 0)
                continue;

            uint32_t *indices = NULL;
            double *values = NULL;
            const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, col, &indices, &values);

            for (unsigned i = 0; i < n; ++i)
            {
                out_ptr[row * out_dims[1] + indices[i]] += values[i] * v;
            }
        }
    }

    Py_END_ALLOW_THREADS;
    Py_DECREF(array);
    return (PyObject *)out;
}

static PyObject *crs_matrix_matmul(PyObject *self, PyObject *other)
{
    if (PyObject_TypeCheck(self, &crs_matrix_type_object))
    {
        return crs_matrix_matmul_left((const crs_matrix_t *)self, other);
    }

    if (PyObject_TypeCheck(other, &crs_matrix_type_object))
    {
        return crs_matrix_matmul_right((const crs_matrix_t *)other, self);
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *crs_matrix_str(const crs_matrix_t *this)
{
    return PyUnicode_FromFormat("MatrixCRS(%u, %u)", this->matrix->base.rows, this->matrix->base.cols);
}

static PyObject *crs_matrix_from_data(crs_matrix_t *const this, PyObject *args, PyObject *kwargs)
{
    PyObject *py_values, *py_column_indices, *py_row_lengths;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", (char *[]){"values", "column_indices", "row_lengths", NULL},
                                     &py_values, &py_column_indices, &py_row_lengths))
    {
        return NULL;
    }

    PyArrayObject *const row_lengths_array = (PyArrayObject *)PyArray_FromAny(
        py_row_lengths, PyArray_DescrFromType(NPY_INT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!row_lengths_array)
    {
        return NULL;
    }
    if (PyArray_DIM(row_lengths_array, 0) != this->matrix->base.rows)
    {
        PyErr_Format(PyExc_ValueError, "Row lengths were specified for %u rows, but the matrix has %u rows.",
                     PyArray_DIM(row_lengths_array, 0), this->matrix->base.rows);
        Py_DECREF(row_lengths_array);
        return NULL;
    }

    PyArrayObject *const column_indices_array = (PyArrayObject *)PyArray_FromAny(
        py_column_indices, PyArray_DescrFromType(NPY_UINT32), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!column_indices_array)
    {
        Py_DECREF(row_lengths_array);
        return NULL;
    }

    PyArrayObject *const values_array = (PyArrayObject *)PyArray_FromAny(
        py_values, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!values_array)
    {
        Py_DECREF(row_lengths_array);
        Py_DECREF(column_indices_array);
        return NULL;
    }

    if (PyArray_DIM(values_array, 0) != PyArray_DIM(column_indices_array, 0))
    {
        PyErr_Format(PyExc_ValueError, "Number of values (%u) does not match number of column indices (%u).",
                     PyArray_DIM(values_array, 0), PyArray_DIM(column_indices_array, 0));
        Py_DECREF(row_lengths_array);
        Py_DECREF(column_indices_array);
        Py_DECREF(values_array);
        return NULL;
    }

    const npy_int64 *const row_lengths_ptr = PyArray_DATA(row_lengths_array);
    const npy_uint32 *const column_indices_ptr = PyArray_DATA(column_indices_array);
    const npy_double *const values_ptr = PyArray_DATA(values_array);

    for (unsigned r = 0, offset = 0; r < this->matrix->base.rows; ++r)
    {
        const unsigned n = row_lengths_ptr[r];
        const uint32_t *const indices = column_indices_ptr + offset;
        const npy_double *const values = values_ptr + offset;
        if (n > this->matrix->base.cols)
        {
            PyErr_Format(PyExc_ValueError, "Row %u has %u entries, but the matrix has %u columns.", r, n,
                         this->matrix->base.cols);
            Py_DECREF(row_lengths_array);
            Py_DECREF(column_indices_array);
            Py_DECREF(values_array);
            return NULL;
        }

        for (unsigned i = 1; i < n; ++i)
        {
            if (indices[i] <= indices[i - 1])
            {
                PyErr_Format(PyExc_ValueError, "Column indices %u and %u are not sorted in row %u.", i - 1, i, r);
                Py_DECREF(row_lengths_array);
                Py_DECREF(column_indices_array);
                Py_DECREF(values_array);
                return NULL;
            }
        }

        if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_build_row(this->matrix, r, n, indices, values)))
        {
            Py_DECREF(row_lengths_array);
            Py_DECREF(column_indices_array);
            Py_DECREF(values_array);
            return NULL;
        }

        offset += n;
    }
    Py_DECREF(row_lengths_array);
    Py_DECREF(column_indices_array);
    Py_DECREF(values_array);
    this->built_rows = this->matrix->base.rows;

    Py_RETURN_NONE;
}

static PyObject *crs_matrix_array_ufunc(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    //
    PyObject *ufunc_method;
    ASSERT(PyTuple_Check(args), "Arguments must be a python tuple.");
    ASSERT(kwds == NULL || PyDict_CheckExact(kwds), "Keyworkds must be passed as a dict.");

    if (PyTuple_GET_SIZE(args) < 2)
    {
        PyErr_Format(PyExc_ValueError, "Expected at least two arguments, got %u.", PyTuple_GET_SIZE(args));
        return NULL;
    }

    {
        PyObject *const ufunc = PyTuple_GET_ITEM(args, 0);
        PyObject *const ufunc_name = PyObject_GetAttrString(ufunc, "__name__");
        if (!ufunc_name)
            return NULL;
        const char *const ufunc_name_str = PyUnicode_AsUTF8(ufunc_name);
        if (!ufunc_name_str)
        {
            Py_DECREF(ufunc_name);
            return NULL;
        }
        if (strcmp(ufunc_name_str, "matmul") != 0)
        {
            Py_DECREF(ufunc_name);
            Py_RETURN_NOTIMPLEMENTED;
        }
        Py_DECREF(ufunc_name);
    }

    {
        PyObject *const method_name = PyTuple_GET_ITEM(args, 1);
        const char *const method = PyUnicode_AsUTF8(method_name);
        if (!method)
            return NULL;

        if (strcmp(method, "__call__") != 0)
        {
            Py_RETURN_NOTIMPLEMENTED;
        }
    }

    PyObject *const normal_args = PyTuple_GetSlice(args, 2, PyTuple_GET_SIZE(args));
    if (!normal_args)
        return NULL;

    // Extract operands from inputs tuple
    PyObject *left, *right;
    const int unpacked_res = PyArg_UnpackTuple(normal_args, "matmul", 2, 2, &left, &right);
    Py_DECREF(normal_args);
    if (!unpacked_res)
    {
        return NULL;
    }
    return crs_matrix_matmul(left, right);
}

static PyObject *crs_matrix_transpose(const crs_matrix_t *this, PyObject *Py_UNUSED(args))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    crs_matrix_t *const new = (crs_matrix_t *)crs_matrix_type_object.tp_alloc(&crs_matrix_type_object, 0);
    if (!new)
        return NULL;

    if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_transpose(this->matrix, &new->matrix, &JMTX_ALLOCATOR)))
    {
        Py_DECREF(new);
        return NULL;
    }
    new->built_rows = new->matrix->base.rows;
    return (PyObject *)new;
}

static PyObject *crs_matrix_shrink(const crs_matrix_t *this, PyObject *Py_UNUSED(args))
{
    if (!crs_matrix_check_build(this))
        return NULL;
    if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_shrink(this->matrix)))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *crs_matrix_remove_entries_bellow(const crs_matrix_t *this, PyObject *arg)
{
    if (!crs_matrix_check_build(this))
        return NULL;
    const double v = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (v < 0)
    {
        PyErr_Format(PyExc_ValueError, "Value must be non-negative, got %f.", v);
        return NULL;
    }
    const unsigned initial = this->matrix->n_entries;
    jmtxd_matrix_crs_remove_bellow_magnitude(this->matrix, v);

    return PyLong_FromUnsignedLong(initial - this->matrix->n_entries);
}

static PyObject *crs_matrix_add_to_dense(const crs_matrix_t *this, PyObject *arg)
{
    if (!crs_matrix_check_build(this))
        return NULL;

    const PyArrayObject *const array = (PyArrayObject *)arg;
    if (check_input_array(array, 2, (const npy_intp[2]){this->matrix->base.rows, this->matrix->base.cols}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out") < 0)
        return NULL;

    npy_double *const in_ptr = PyArray_DATA(array);
    for (unsigned ie = 0, row = 0; ie < this->matrix->n_entries; ++ie)
    {
        while (ie >= this->matrix->end_of_row_offsets[row])
        {
            row += 1;
            ASSERT(row < this->matrix->base.rows, "Internal error - matrix rows could not be deduced for entry %u", ie);
        }
        in_ptr[this->matrix->base.cols * row + this->matrix->indices[ie]] += this->matrix->values[ie];
    }

    Py_RETURN_NONE;
}

static PyObject *crs_matrix_from_dense(PyTypeObject *type, PyObject *arg)
{
    PyArrayObject *const array =
        (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!array)
        return NULL;
    const npy_intp *dims = PyArray_DIMS(array);
    crs_matrix_t *const this = (crs_matrix_t *)type->tp_alloc(type, 0);
    if (!this)
    {
        Py_DECREF(array);
        return NULL;
    }

    if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_new(&this->matrix, dims[0], dims[1], dims[0] * dims[1], &JMTX_ALLOCATOR)))
    {
        Py_DECREF(this);
        Py_DECREF(array);
        return NULL;
    }
    this->built_rows = this->matrix->base.rows;
    this->matrix->n_entries = dims[0] * dims[1];
    const npy_double *const in_ptr = PyArray_DATA(array);
    memcpy(this->matrix->values, in_ptr, sizeof *this->matrix->values * this->matrix->n_entries);
    for (unsigned i = 0; i < this->matrix->base.rows; ++i)
    {
        this->matrix->end_of_row_offsets[i] = (i + 1) * dims[1];
        for (unsigned j = 0; j < this->matrix->base.cols; ++j)
        {
            this->matrix->indices[i * dims[1] + j] = j;
        }
    }
    Py_DECREF(array);
    return (PyObject *)this;
}

static unsigned count_non_empty_rows(const crs_matrix_t *this)
{
    unsigned n = 0;
    for (unsigned r = 0, offset = 0; r < this->matrix->base.rows; ++r)
    {
        if (this->matrix->end_of_row_offsets[r] > offset)
        {
            n += 1;
        }
        offset = this->matrix->end_of_row_offsets[r];
    }
    return n;
}

static PyObject *crs_matrix_multiply_to_sparse(const crs_matrix_t *this, PyObject *arg)
{
    if (!crs_matrix_check_build(this))
        return NULL;
    const PyArrayObject *const array =
        (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!array)
        return NULL;

    const unsigned v_size = PyArray_DIM(array, 0);
    if (v_size != this->matrix->base.cols)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Number of columns of the right vector (%u) does not match the number of columns in the matrix (%u).",
            v_size, this->matrix->base.cols);
        Py_DECREF(array);
        return NULL;
    }
    svector_t svector;
    if (sparse_vector_new(&svector, this->matrix->base.rows, this->matrix->base.rows, &SYSTEM_ALLOCATOR))
    {
        Py_DECREF(array);
        return NULL;
    }

    const npy_double *const in_ptr = PyArray_DATA(array);
    for (unsigned r = 0; r < this->matrix->base.rows; ++r)
    {
        uint32_t *indices = NULL;
        double *values = NULL;
        const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, r, &indices, &values);
        double v = 0.0;
        for (unsigned i = 0; i < n; ++i)
        {
            v += in_ptr[indices[i]] * values[i];
        }
        if (v != 0.0)
        {
            // Memory was pre-allocated for enough entries, so no need to check this.
            (void)sparse_vector_append(&svector, (entry_t){.index = r, .value = v}, &SYSTEM_ALLOCATOR);
        }
    }
    Py_DECREF(array);
    svec_object_t *const out = sparse_vector_to_python(&svector);
    sparse_vector_del(&svector, &SYSTEM_ALLOCATOR);
    return (PyObject *)out;
}

PyDoc_STRVAR(crs_matrix_to_array_docstring, "toarray() -> array\n"
                                            "Convert the sparse matrix to a dense NumPy array.\n"
                                            "\n"
                                            "Returns\n"
                                            "-------\n"
                                            "array\n"
                                            "    Representation of the matrix as a NumPy array.\n");

PyDoc_STRVAR(crs_matrix_build_row_docstring,
             "build_row(row: int, entries: SparseVector | None = None) -> None\n"
             "Add a row to the matrix.\n"
             "\n"
             "This should be called for each row in order as the matrix is constructed,\n"
             "since this updates the end or row offset values. It allows for very quick\n"
             "matrix construction.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "  Index of the row that is being set.\n"
             "\n"
             "entries : SparseVector, optional\n"
             "  Entries of the row that is to be set. If not given, the row is set to empty.\n");

PyDoc_STRVAR(crs_matrix_from_data_docstring,
             "set_from_data(values: numpy.typing.ArrayLike, column_indices: numpy.typing.ArrayLike, row_lengths: "
             "numpy.typing.ArrayLike) -> None\n"
             "Populate the matrix with data.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "values : array\n"
             "    Array of values of entries in the matrix.\n"
             "\n"
             "column_indices : array\n"
             "    Indices of the column indices of the matrix. Must have the exact same lenght\n"
             "    as ``values``. For each row, this should also be sorted and not exceed the\n"
             "    column count of the matrix.\n"
             "\n"
             "row_lengths : array\n"
             "    Length for each row in the matrix.\n"
             "\n"
             "Examples\n"
             "--------\n"
             "This method is primarely intended to allow for conversion to and from\n"
             ":mod:`scipy.sparse.csr_array`, which stores data based on these values.\n"
             "\n"
             ">>> from scipy import sparse as sp\n"
             ">>> from mfv2d._mfv2d import MatrixCRS\n"
             ">>> import numpy as np\n"
             ">>>\n"
             ">>> np.random.seed(0) # Consistant seed for the test\n"
             ">>> np.set_printoptions(precision=2)\n"
             ">>>\n"
             ">>> m = np.random.random_sample(6, 6)\n"
             ">>> mask = m < 0.5\n"
             ">>> m[mask] = 0\n"
             ">>> m[~mask] = 2 * m[~mask] - 1.5\n"
             ">>> m\n"
             "array([[-0.4 , -0.07, -0.29, -0.41,  0.  , -0.21],\n"
             "       [ 0.  ,  0.28,  0.43,  0.  ,  0.08, -0.44],\n"
             "       [-0.36,  0.35,  0.  ,  0.  ,  0.  ,  0.17],\n"
             "       [ 0.06,  0.24,  0.46,  0.1 ,  0.  ,  0.06],\n"
             "       [ 0.  , -0.22,  0.  ,  0.39, -0.46,  0.  ],\n"
             "       [ 0.  ,  0.05,  0.  , -0.36,  0.  , -0.26]])\n"
             "\n"
             "If this array is now converted into a scipy CRS array, it\n"
             "\n"
             ">>> m1 = sp.csr_array(m)\n"
             ">>> m2 = MatrixCRS(*m1.shape)\n"
             ">>> m2.set_from_data(\n"
             "...     m1.data, np.astype(m1.indices, np.uint32), m1.indptr[1:] - m1.indptr[:-1]\n"
             "... )\n"
             ">>> m2.toarray() == m1.toarray()\n"
             "array([[ True,  True,  True,  True,  True,  True],\n"
             "       [ True,  True,  True,  True,  True,  True],\n"
             "       [ True,  True,  True,  True,  True,  True],\n"
             "       [ True,  True,  True,  True,  True,  True],\n"
             "       [ True,  True,  True,  True,  True,  True],\n"
             "       [ True,  True,  True,  True,  True,  True]])\n");

PyDoc_STRVAR(crs_matrix_transpose_docstring,
             "transpose()\n"
             "Transpose of the matrix.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "MatrixCRS\n"
             "    Transposed matrix, where entry :math:`(i, j)` in the original is equal\n"
             "    to entriy :math:`(j, i)` in the transpose.\n");

PyDoc_STRVAR(crs_matrix_shrink_docstring, "shrink()\n"
                                          "Shrink the matrix to the minimum size.\n");

PyDoc_STRVAR(crs_matrix_remove_entries_bellow_docstring,
             "remove_entries_bellow(v: float = 0.0, /)\n"
             "Remove entries with the magnitude bellow specified value.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "v : float, default: 0.0\n"
             "    Magnitude bellow which values should be removed. Can not be\n"
             "    negative.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Number of entries that have been removed.\n");

PyDoc_STRVAR(crs_matrix_add_to_dense_docstring, "add_to_dense(out: numpy.typing.NDArray[numpy.float64], /) -> None\n"
                                                "Add nonzero entries of the matrix to the NumPy array.\n"
                                                "\n"
                                                "This is useful when trying to merge multiple sparse arrays\n"
                                                "into a single dense NumPy array.\n"
                                                "\n"
                                                "Parameters\n"
                                                "----------\n"
                                                "out : array\n"
                                                "    Array to which the output is written. The shape must match\n"
                                                "    exactly, and the type must also be exactly the same.\n");

PyDoc_STRVAR(crs_matrix_from_dense_docstring, "from_dense(x: numpy.typing.ArrayLike, /) -> typing.Self\n"
                                              "Create a new sparse matrix from a dense array.\n"
                                              "\n"
                                              "Parameters\n"
                                              "----------\n"
                                              "x : array_like\n"
                                              "    Dense array the matrix is created from. Must be\n"
                                              "    two dimensional.\n"
                                              "\n"
                                              "Returns\n"
                                              "-------\n"
                                              "MatrixCRS\n"
                                              "    Matrix that is initialized from the data of the full\n"
                                              "    matrix. This includes zeros.\n");

PyDoc_STRVAR(crs_matrix_multiply_to_sparse_docstring,
             "multiply_to_sparse(x: numpy.typing.ArrayLike, /) -> SparseVector\n"
             "Multiply with a dense array, but return the result as a sparse vector.\n"
             "\n"
             "This is useful when the sparse matrix has many empty rows, which is common\n"
             "for element constraint matrices.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "SparseVector\n"
             "    Result of multiplying the dense vector by the sparse matrix as a\n"
             "    sparse vector.\n");

static PyMethodDef crs_matrix_methods[] = {
    {
        .ml_name = "toarray",
        .ml_meth = (void *)crs_matrix_toarray,
        .ml_flags = METH_NOARGS,
        .ml_doc = crs_matrix_to_array_docstring,
    },
    {
        .ml_name = "build_row",
        .ml_meth = (void *)crs_matrix_build_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = crs_matrix_build_row_docstring,
    },
    {
        .ml_name = "set_from_data",
        .ml_meth = (void *)crs_matrix_from_data,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = crs_matrix_from_data_docstring,
    },
    {
        .ml_name = "__array_ufunc__",
        .ml_meth = (void *)crs_matrix_array_ufunc,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Handle numpy ufuncs including matmul",
    },
    {
        .ml_name = "transpose",
        .ml_meth = (void *)crs_matrix_transpose,
        .ml_flags = METH_NOARGS,
        .ml_doc = crs_matrix_transpose_docstring,
    },
    {
        .ml_name = "shrink",
        .ml_meth = (void *)crs_matrix_shrink,
        .ml_flags = METH_NOARGS,
        .ml_doc = crs_matrix_shrink_docstring,
    },
    {
        .ml_name = "remove_entries_bellow",
        .ml_meth = (void *)crs_matrix_remove_entries_bellow,
        .ml_flags = METH_O,
        .ml_doc = crs_matrix_remove_entries_bellow_docstring,
    },
    {
        .ml_name = "add_to_dense",
        .ml_meth = (void *)crs_matrix_add_to_dense,
        .ml_flags = METH_O,
        .ml_doc = crs_matrix_add_to_dense_docstring,
    },
    {
        .ml_name = "from_dense",
        .ml_meth = (void *)crs_matrix_from_dense,
        .ml_flags = METH_O | METH_CLASS,
        .ml_doc = crs_matrix_from_dense_docstring,
    },
    {
        .ml_name = "multiply_to_sparse",
        .ml_meth = (void *)crs_matrix_multiply_to_sparse,
        .ml_flags = METH_O,
        .ml_doc = crs_matrix_multiply_to_sparse_docstring,
    },
    {},
};

static PyObject *crs_matrix_get_row(const crs_matrix_t *this, PyObject *key)
{
    if (!crs_matrix_check_build(this))
        return NULL;

    if (PyObject_TypeCheck(key, &PyTuple_Type))
    {
        unsigned row, col;
        if (!PyArg_ParseTuple(key, "II", &row, &col))
        {
            return NULL;
        }
        if (row >= this->matrix->base.rows || col >= this->matrix->base.cols)
        {
            PyErr_Format(PyExc_KeyError, "Element at (%u, %u) is out of bounds for matrix of dimensions (%u, %u)", row,
                         col, this->matrix->base.rows, this->matrix->base.cols);
            return NULL;
        }
        return PyFloat_FromDouble(jmtxd_matrix_crs_get_entry(this->matrix, row, col));
    }

    const long long_row = PyLong_AsLong(key);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (long_row >= this->matrix->base.rows)
    {
        PyErr_Format(PyExc_KeyError, "Row %ld is out of bounds for matrix of dimensions (%u, %u)", long_row,
                     this->matrix->base.rows, this->matrix->base.cols);
        return NULL;
    }

    uint32_t *indices = NULL;
    double *values = NULL;
    const unsigned n = jmtxd_matrix_crs_get_row(this->matrix, long_row, &indices, &values);
    svector_t svec;
    if (sparse_vector_new(&svec, this->matrix->base.cols, n, &SYSTEM_ALLOCATOR))
    {
        return NULL;
    }
    for (unsigned i = 0; i < n; ++i)
    {
        svec.entries[i] = (entry_t){.index = indices[i], .value = values[i]};
    }
    svec.count = n;
    svec_object_t *const out = sparse_vector_to_python(&svec);
    sparse_vector_del(&svec, &SYSTEM_ALLOCATOR);

    return (PyObject *)out;
}

static PyObject *crs_matrix_get_column_indices(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    npy_intp dims[1] = {this->matrix->n_entries};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
    if (!array)
        return NULL;
    npy_uint32 *const ptr = PyArray_DATA(array);
    for (unsigned i = 0; i < this->matrix->n_entries; ++i)
    {
        ptr[i] = this->matrix->indices[i];
    }
    return (PyObject *)array;
}

static PyObject *crs_matrix_get_row_indices(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    npy_intp dims[1] = {this->matrix->n_entries};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
    if (!array)
        return NULL;
    npy_uint32 *const ptr = PyArray_DATA(array);
    for (unsigned i = 0, r = 0; i < this->matrix->n_entries; ++i)
    {
        while (r < this->matrix->base.rows && i >= this->matrix->end_of_row_offsets[r])
        {
            r += 1;
        }
        ptr[i] = r;
    }
    return (PyObject *)array;
}

static PyObject *crs_matrix_get_values(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    npy_intp dims[1] = {this->matrix->n_entries};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!array)
        return NULL;
    npy_double *const ptr = PyArray_DATA(array);
    for (unsigned i = 0; i < this->matrix->n_entries; ++i)
    {
        ptr[i] = this->matrix->values[i];
    }
    return (PyObject *)array;
}

static PyObject *crs_matrix_get_row_offsets(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    npy_intp dims[1] = {this->matrix->base.rows + 1};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
    if (!array)
        return NULL;
    npy_uint32 *const ptr = PyArray_DATA(array);
    ptr[0] = 0;
    for (unsigned r = 0; r < this->matrix->base.rows; ++r)
    {
        ptr[r + 1] = this->matrix->end_of_row_offsets[r];
    }
    return (PyObject *)array;
}

static PyObject *crs_matrix_get_position_pairs(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    npy_intp dims[2] = {this->matrix->n_entries, 2};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    if (!array)
        return NULL;
    npy_uint32 *const ptr = PyArray_DATA(array);
    for (unsigned i = 0, r = 0; i < this->matrix->n_entries; ++i)
    {
        if (r < this->matrix->base.rows && i >= this->matrix->end_of_row_offsets[r])
        {
            r += 1;
        }
        ptr[2 * i + 0] = r;
        ptr[2 * i + 1] = this->matrix->indices[i];
    }
    return (PyObject *)array;
}

static PyObject *crs_matrix_get_built_rows(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromUnsignedLong(this->built_rows);
}

static PyObject *crs_matrix_get_shape(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    return PyTuple_Pack(2, PyLong_FromUnsignedLong(this->matrix->base.rows),
                        PyLong_FromUnsignedLong(this->matrix->base.cols));
}

static PyObject *crs_matrix_get_nonempty_rows(const crs_matrix_t *this, PyObject *Py_UNUSED(ignored))
{
    if (!crs_matrix_check_build(this))
        return NULL;

    const npy_intp n = count_non_empty_rows(this);
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_UINT32);
    if (!array)
        return NULL;

    npy_uint32 *const ptr = PyArray_DATA(array);
    for (unsigned r = 0, i = 0, offset = 0; i < n; ++r)
    {
        if (this->matrix->end_of_row_offsets[r] > offset)
        {
            ptr[i] = r;
            i += 1;
        }
        offset = this->matrix->end_of_row_offsets[r];
    }

    return (PyObject *)array;
}

PyDoc_STRVAR(crs_matrix_docstring, "MatrixCRS(rows: int, cols: int)\n"
                                   "Compressed-row sparse matrix.\n"
                                   "\n"
                                   "    Type used to store sparse matrices and allow for building them in\n"
                                   "    an efficient way.\n");

static PyNumberMethods crs_matrix_as_number = {
    .nb_matrix_multiply = (binaryfunc)crs_matrix_matmul,
};

static PyMappingMethods crs_matrix_as_mapping = {
    .mp_subscript = (binaryfunc)crs_matrix_get_row,
};

static PyGetSetDef crs_matrix_getset[] = {
    {
        .name = "column_indices",
        .get = (getter)crs_matrix_get_column_indices,
        .set = NULL,
        .doc = "array : Column indices of non-zero values.",
        .closure = NULL,
    },
    {
        .name = "row_indices",
        .get = (getter)crs_matrix_get_row_indices,
        .set = NULL,
        .doc = "array : Row indices of non-zero values.",
        .closure = NULL,
    },
    {
        .name = "values",
        .get = (getter)crs_matrix_get_values,
        .set = NULL,
        .doc = "array : Values of non-zero values.",
        .closure = NULL,
    },
    {
        .name = "position_pairs",
        .get = (getter)crs_matrix_get_position_pairs,
        .set = NULL,
        .doc = "(N, 2) array : Array of position pairs of non-zero values.",
        .closure = NULL,
    },
    {
        .name = "row_offsets",
        .get = (getter)crs_matrix_get_row_offsets,
        .set = NULL,
        .doc = "array : Array with number of elements before each row begins.",
        .closure = NULL,
    },
    {
        .name = "built_rows",
        .get = (getter)crs_matrix_get_built_rows,
        .set = NULL,
        .doc = "int : Number of rows that have been built.",
        .closure = NULL,
    },
    {
        .name = "shape",
        .get = (getter)crs_matrix_get_shape,
        .set = NULL,
        .doc = "(int, int) : Shape of the matrix.",
        .closure = NULL,
    },
    {
        .name = "nonempty_rows",
        .get = (getter)crs_matrix_get_nonempty_rows,
        .set = NULL,
        .doc = "array : Indices of rows with at least one entry.",
        .closure = NULL,
    },
    {},
};

MFV2D_INTERNAL
PyTypeObject crs_matrix_type_object = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.MatrixCRS",
    .tp_new = (newfunc)crs_matrix_new,
    .tp_str = (reprfunc)crs_matrix_str,
    .tp_basicsize = sizeof(crs_matrix_t),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = crs_matrix_docstring,
    .tp_methods = crs_matrix_methods,
    .tp_as_number = &crs_matrix_as_number,
    .tp_as_mapping = &crs_matrix_as_mapping,
    .tp_getset = crs_matrix_getset,
    .tp_del = (destructor)crs_matrix_del,
};
