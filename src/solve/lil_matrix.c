//
// Created by jan on 19.3.2025.
//

#include "lil_matrix.h"
#include "qr_solve.h"
#include <numpy/ndarrayobject.h>

static PyObject *lil_mat_repr(const lil_mat_object_t *this)
{
    return PyUnicode_FromFormat("LiLMatrix(%" PRIu64 ", %" PRIu64 ")", this->rows, this->cols);
}

static PyObject *lil_mat_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    unsigned long long nr = 0, nc = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KK", (char *[3]){"rows", "cols", NULL}, &nr, &nc))
    {
        return NULL;
    }

    lil_mat_object_t *const this = (lil_mat_object_t *)type->tp_alloc(type, (Py_ssize_t)nr);
    if (!this)
    {
        return NULL;
    }
    this->rows = nr;
    this->cols = nc;
    for (uint64_t i = 0; i < nr; ++i)
    {
        this->row_data[i] = (svector_t){.n = nc};
    }

    return (PyObject *)this;
}

static Py_ssize_t lil_mat_length(const lil_mat_object_t *this)
{
    return (Py_ssize_t)this->rows;
}

static PyObject *lil_mat_get_row(const lil_mat_object_t *this, PyObject *arg)
{
    const uint64_t i = PyLong_AsUnsignedLongLong(arg);

    if (PyErr_Occurred())
        return NULL;

    if (i >= this->rows)
    {
        PyErr_Format(PyExc_KeyError,
                     "Row index %" PRIu64 " is out of bounds for a matrix with size (%" PRIu64 ", %" PRIu64 ").", i,
                     this->rows, this->cols);
        return NULL;
    }

    return (PyObject *)sparse_vec_to_python(this->row_data + i);
}

static int lil_mat_set_row(lil_mat_object_t *this, PyObject *py_idx, PyObject *arg)
{
    const uint64_t idx = PyLong_AsUnsignedLongLong(py_idx);
    if (PyErr_Occurred())
    {
        return -1;
    }
    if (idx >= this->rows)
    {
        PyErr_Format(PyExc_KeyError,
                     "Row index %" PRIu64 " is out of bounds for a matrix with size (%" PRIu64 ", %" PRIu64 ").", idx,
                     this->rows, this->cols);
        return -1;
    }
    if (arg == NULL)
    {
        this->row_data[idx].count = 0;
        return 0;
    }

    if (!PyObject_TypeCheck(arg, &svec_type_object))
    {
        PyErr_Format(PyExc_TypeError, "Row of a LiLMatrix can only be set by a SparseVector (instead got %R)",
                     Py_TYPE(arg));
        return -1;
    }
    const svec_object_t *vec = (svec_object_t *)arg;

    if (vec->n != this->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Sparse vector does not have the dimensions matching the matrix columns.");
        return -1;
    }

    svector_t *const row = this->row_data + idx;
    if (sparse_vec_resize(row, vec->count, &SYSTEM_ALLOCATOR))
    {
        return -1;
    }

    row->count = vec->count;
    memcpy(row->entries, vec->entries, sizeof *vec->entries * vec->count);

    return 0;
}

static PyMappingMethods lil_mat_mapping = {.mp_length = (lenfunc)lil_mat_length,
                                           .mp_subscript = (binaryfunc)lil_mat_get_row,
                                           .mp_ass_subscript = (objobjargproc)lil_mat_set_row};

static PyObject *lil_mat_count_entries(const lil_mat_object_t *this, PyObject *Py_UNUSED(arg))
{
    uint64_t count = 0;
    for (uint64_t i = 0; i < this->rows; ++i)
    {
        count += this->row_data[i].count;
    }

    return PyLong_FromSize_t((size_t)count);
}

static PyObject *lil_mat_as_array(const lil_mat_object_t *this, PyObject *args, PyObject *kwds)
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

    const npy_intp sizes[2] = {(npy_intp)this->rows, (npy_intp)this->cols};

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, sizes, NPY_FLOAT64);
    if (!out)
        return NULL;
    npy_float64 *const ptr = PyArray_DATA(out);
    memset(ptr, 0, sizeof *ptr * sizes[0] * sizes[1]);
    for (uint64_t i_row = 0; i_row < this->rows; ++i_row)
    {
        const svector_t *row = this->row_data + i_row;
        for (uint64_t i = 0; i < row->count; ++i)
        {
            ptr[i_row * sizes[1] + row->entries[i].index] = row->entries[i].value;
        }
    }

    return (PyObject *)out;
}

static PyObject *lil_mat_from_full(PyTypeObject *type, PyObject *arg)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_FLOAT64), 2, 2,
                                                                  NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array)
        return NULL;

    const npy_intp *dims = PyArray_DIMS(array);
    lil_mat_object_t *const this = (lil_mat_object_t *)type->tp_alloc(type, dims[0]);
    const npy_float64 *const ptr = PyArray_DATA(array);

    for (uint64_t row = 0; row < dims[0]; ++row)
    {
        if (sparse_vector_new(this->row_data + row, dims[1], dims[1], &SYSTEM_ALLOCATOR))
        {
            Py_DECREF(this);
            Py_DECREF(array);
            return NULL;
        }
        uint64_t j, k;
        for (j = 0, k = 0; j < dims[1]; ++j)
        {
            const npy_float64 v = ptr[row * dims[1] + j];
            if (v != 0.0)
            {
                this->row_data[row].entries[k] = (entry_t){.index = j, .value = v};
                k += 1;
            }
        }
        this->row_data[row].count = k;
    }

    this->rows = dims[0];
    this->cols = dims[1];
    Py_DECREF(array);
    return (PyObject *)this;
}

static void lil_mat_delete(lil_mat_object_t *this)
{
    for (uint64_t i = 0; i < this->rows; ++i)
    {
        sparse_vec_del(this->row_data + i, &SYSTEM_ALLOCATOR);
    }
}

static PyObject *lil_mat_qr_decomposition(lil_mat_object_t *this, PyObject *args)
{
    long long int n_max = LLONG_MAX;
    if (!PyArg_ParseTuple(args, "|L", &n_max))
    {
        return NULL;
    }

    const lil_matrix_t tmp = {.rows = this->rows, .cols = this->cols, .row_data = this->row_data};
    uint64_t n_givens;
    givens_rotation_t *p_givens;

    for (uint64_t i = 0; i < this->rows; ++i)
    {
        ASSERT(this->row_data[i].n == this->cols,
               "Dimension of row %" PRIu64 " (%" PRIu64 ") does not match column count %" PRIu64 ".", i,
               this->row_data[i].n, this->cols);
    }
    int result;
    Py_BEGIN_ALLOW_THREADS;

    // Threads are allowed inside here, since this can be on the slow side.
    // As long as no one else has the bright idea to start messing with the
    // matrix in another thread we will be fine. If someone is retarded enough
    // to try that, may God have mercy on his soul.
    result = decompose_qr((int64_t)n_max, &tmp, &n_givens, &p_givens, &SYSTEM_ALLOCATOR);

    Py_END_ALLOW_THREADS;

    if (result)
    {
        return NULL;
    }

    givens_series_t *const series = givens_series_to_python(n_givens, p_givens);
    deallocate(&SYSTEM_ALLOCATOR, p_givens);

    return (PyObject *)series;
}

static PyObject *lil_mat_block_diagonal(PyTypeObject *type, PyObject *const *args, const Py_ssize_t nargs)
{
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(args[i], &lil_mat_type_object))
        {
            PyErr_Format(PyExc_TypeError, "All arguments must be LiLMatrix objects, but argument %u had the type %R.",
                         i, Py_TYPE(args[i]));
            return NULL;
        }
    }

    const lil_mat_object_t *const *matrices = (const lil_mat_object_t *const *)args;
    uint64_t n_rows = 0, n_cols = 0;
    for (unsigned i = 0; i < nargs; ++i)
    {
        n_rows += matrices[i]->rows;
        n_cols += matrices[i]->cols;
    }

    lil_mat_object_t *const this = (lil_mat_object_t *)type->tp_alloc(type, (Py_ssize_t)n_rows);

    if (!this)
        return NULL;

    this->rows = n_rows;
    uint64_t current_offset = 0, col_offset = 0;
    for (unsigned i = 0; i < nargs; ++i)
    {
        const lil_mat_object_t *mat = matrices[i];
        for (unsigned j = 0; j < mat->rows; ++j)
        {
            if (sparse_vector_copy(mat->row_data + j, this->row_data + current_offset, &SYSTEM_ALLOCATOR))
            {
                Py_DECREF(this);
                return NULL;
            }
            this->row_data[current_offset].n = n_cols;

            for (unsigned k = 0; k < this->row_data[current_offset].count; ++k)
            {
                this->row_data[current_offset].entries[k].index += col_offset;
            }

            current_offset += 1;
        }
        col_offset += mat->cols;
    }
    this->cols = col_offset;

    return (PyObject *)this;
}

static PyObject *lil_matrix_add_columns(lil_mat_object_t *this, PyObject *const *args, const Py_ssize_t nargs)
{
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(args[i], &svec_type_object))
        {
            PyErr_Format(PyExc_TypeError, "All columns must be SparseVectors, but parameter %u had the type %R", i,
                         Py_TYPE(args[i]));
            return NULL;
        }
    }

    const svec_object_t *const *const cols = (const svec_object_t *const *)args;
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (cols[i]->n != this->rows)
        {
            PyErr_Format(PyExc_ValueError,
                         "Column %u did not have the dimensions of the matrix rows (expected %u, got %u).", i,
                         (unsigned)this->rows, (unsigned)cols[i]->n);
            return NULL;
        }
    }

    const uint64_t base_cols = this->cols;
    this->cols += nargs;
    for (unsigned i = 0; i < this->rows; ++i)
    {
        this->row_data[i].n += nargs;
    }

    for (unsigned i = 0; i < nargs; ++i)
    {
        const svec_object_t *const vec = cols[i];
        for (unsigned j = 0; j < vec->count; ++j)
        {
            const entry_t *entry = vec->entries + j;
            if (entry->value == 0)
                continue;

            if (sparse_vector_append(this->row_data + entry->index,
                                     (entry_t){.index = base_cols + i, .value = entry->value}, &SYSTEM_ALLOCATOR))
            {
                return NULL;
            }
        }
    }

    Py_RETURN_NONE;
}

static PyObject *lil_matrix_add_rows(const lil_mat_object_t *this, PyObject *const *args, const Py_ssize_t nargs)
{
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(args[i], &svec_type_object))
        {
            PyErr_Format(PyExc_TypeError, "All rows must be SparseVectors, but parameter %u had the type %R", i,
                         Py_TYPE(args[i]));
            return NULL;
        }
    }

    const svec_object_t *const *const rows = (const svec_object_t *const *)args;
    for (unsigned i = 0; i < nargs; ++i)
    {
        if (rows[i]->n != this->cols)
        {
            PyErr_Format(PyExc_ValueError,
                         "Row %u did not have the dimensions of the matrix columns (expected %u, got %u).", i,
                         (unsigned)this->cols, (unsigned)rows[i]->n);
            return NULL;
        }
    }

    lil_mat_object_t *const self =
        (lil_mat_object_t *)lil_mat_type_object.tp_alloc(&lil_mat_type_object, (Py_ssize_t)(this->rows + nargs));
    if (!self)
        return NULL;

    self->cols = this->cols;
    self->rows = this->rows + nargs;

    for (uint64_t i = 0; i < this->rows; ++i)
    {
        self->row_data[i].n = this->cols;
        if (sparse_vector_copy(this->row_data + i, self->row_data + i, &SYSTEM_ALLOCATOR))
        {
            Py_DECREF(self);
            return NULL;
        }
    }

    for (uint64_t i = 0; i < nargs; ++i)
    {
        const uint64_t j = i + this->rows;
        self->row_data[j].n = this->cols;
        const svec_object_t *const vec = rows[i];
        const svector_t in = {.n = this->cols, .count = vec->count, .capacity = 0, .entries = (entry_t *)vec->entries};
        if (sparse_vector_copy(&in, self->row_data + j, &SYSTEM_ALLOCATOR))
        {
            Py_DECREF(self);
            return NULL;
        }
    }

    return (PyObject *)self;
}

static PyObject *lil_matrix_solve_upper_triangular(const lil_mat_object_t *this, PyObject *args, PyObject *kwds)
{
    PyObject *rhs;
    PyArrayObject *out = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O!", (char *[3]){"", "out", NULL}, &rhs, &PyArray_Type, &out))
    {
        return NULL;
    }

    if (out)
    {
        if (PyArray_TYPE(out) != NPY_FLOAT64)
        {
            PyErr_Format(PyExc_ValueError,
                         "Output array must have the data type of float64 (%u), but it had %d instead.", NPY_FLOAT64,
                         PyArray_TYPE(out));
            return NULL;
        }
        const unsigned needed_flags = (NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE);
        if ((PyArray_FLAGS(out) & needed_flags) != needed_flags)
        {
            PyErr_Format(PyExc_ValueError, "Output array must have at least flags %x, but it has %x.", needed_flags,
                         PyArray_FLAGS(out));
            return NULL;
        }
    }

    PyArrayObject *const in = (PyArrayObject *)PyArray_FromAny(rhs, PyArray_DescrFromType(NPY_FLOAT64), 1, 2,
                                                               NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!in)
        return NULL;

    const unsigned ndims = PyArray_NDIM(in);
    const npy_intp *const dims = PyArray_DIMS(in);

    if (out)
    {
        const unsigned out_dims = PyArray_NDIM(out);
        if (out_dims != ndims)
        {
            PyErr_Format(PyExc_ValueError,
                         "Output array does not have the same number of dimensions as the input array (%u against %u).",
                         out_dims, ndims);
            Py_DECREF(in);
            return NULL;
        }
        const npy_intp *const dims_out = PyArray_DIMS(out);
        for (unsigned i = 0; i < ndims; ++i)
        {
            if (dims[i] != dims_out[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Output array's dimension %u does not match input array's (%u against %u).", i, dims[i],
                             dims_out[i]);
                Py_DECREF(in);
                return NULL;
            }
        }
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(ndims, dims, NPY_FLOAT64);
        if (!out)
        {
            Py_DECREF(in);
            return NULL;
        }
    }

    const npy_float64 *ptr_in = PyArray_DATA(in);
    npy_float64 *ptr_out = PyArray_DATA(out);

    const unsigned n_cols = ndims == 2 ? dims[1] : 1;

    Py_BEGIN_ALLOW_THREADS;

    /**
     * Solve by back-substitution. This means we assume that we have a non-zero diagonals, which are the first element
     * in each row.
     */

    for (uint64_t i_row = this->rows; i_row > 0; --i_row)
    {
        const uint64_t r = i_row - 1;
        const svector_t *const row = this->row_data + r;
        ASSERT(row->entries[0].index == r, "Row %" PRIu64 " starts at column %" PRIu64 ", not the diagonal.", r,
               row->entries[0].index);
        for (unsigned i_col = 0; i_col < n_cols; ++i_col)
        {
            scalar_t v = 0.0;
            for (uint64_t j = 1; j < row->count; ++j)
            {
                v += row->entries[j].value * ptr_out[row->entries[j].index * n_cols + i_col];
            }
            ptr_out[r * n_cols + i_col] = (ptr_in[r * n_cols + i_col] - v) / row->entries[0].value;
        }
    }

    Py_END_ALLOW_THREADS;

    Py_DECREF(in);
    return (PyObject *)out;
}

static PyObject *lil_mat_empty_diagonal(PyTypeObject *type, PyObject *arg)
{
    const Py_ssize_t py_n = PyLong_AsSsize_t(arg);
    if (PyErr_Occurred())
        return NULL;

    if (py_n <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Value of dimension must be strictly positive (got %zd instead).", py_n);
        return NULL;
    }

    const uint64_t n = (uint64_t)py_n;

    lil_mat_object_t *const this = (lil_mat_object_t *)type->tp_alloc(type, py_n);
    if (!this)
        return NULL;

    this->rows = n;
    this->cols = n;

    for (uint64_t i = 0; i < n; ++i)
    {
        svector_t *const row = this->row_data + i;
        if (sparse_vector_new(row, n, 1, &SYSTEM_ALLOCATOR))
        {
            Py_DECREF(this);
            return NULL;
        }
        row->entries[0] = (entry_t){.index = i, .value = 0.0};
        row->count = 1;
    }

    return (PyObject *)this;
}

static uint64_t compute_usage(const lil_mat_object_t *this)
{
    uint64_t n = 0;
    for (uint64_t i = 0; i < this->rows; ++i)
    {
        n += this->row_data[i].count;
    }
    return n;
}

static PyObject *lil_mat_to_scipy(const lil_mat_object_t *this, PyObject *Py_UNUSED(arg))
{
    const uint64_t usage = compute_usage(this);
    const npy_intp dims[2] = {(npy_intp)usage, 2};
    PyArrayObject *const value_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    PyArrayObject *const index_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!value_array || !index_array)
    {
        Py_XDECREF(value_array);
        Py_XDECREF(index_array);
        return NULL;
    }

    npy_float64 *const ptr_vals = PyArray_DATA(value_array);
    npy_uint64 *const idx_array = PyArray_DATA(index_array);
    uint64_t p = 0;
    for (uint64_t row = 0; row < this->rows; ++row)
    {
        const svector_t *const v = this->row_data + row;
        for (uint64_t j = 0; j < v->count; ++j)
        {
            ptr_vals[j] = v->entries[j].value;
            idx_array[j << 1] = row;                     // 2 * j
            idx_array[j << 1 | 1] = v->entries[j].index; // 2 * j + 1
        }
    }

    return PyTuple_Pack(2, value_array, index_array);
}

static PyMethodDef lil_mat_methods[] = {
    {.ml_name = "count_entries",
     .ml_meth = (void *)lil_mat_count_entries,
     .ml_flags = METH_NOARGS,
     .ml_doc = "count_entries() -> int:\n"
               "Convert the matrix into a numpy array.\n"},
    {.ml_name = "__array__",
     .ml_meth = (void *)lil_mat_as_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray\n"
               "Convert the matrix into a numpy array.\n"},
    {.ml_name = "from_full",
     .ml_meth = (void *)lil_mat_from_full,
     .ml_flags = METH_O | METH_CLASS,
     .ml_doc = "from_full(mat: array, /) -> LiLMatrix:\n"
               "Create A LiLMatrix from a full matrix.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "mat : array\n"
               "    Full matrix to convert.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "LiLMatrix\n"
               "    Matrix represented in the LiLMatrix format.\n"},
    {.ml_name = "qr_decompose",
     .ml_meth = (void *)lil_mat_qr_decomposition,
     .ml_flags = METH_VARARGS,
     .ml_doc = "qr_decompose(n: int | None = None, /) -> GivensSeries:\n"
               "Decompose the matrix into a series of Givens rotations and a triangular matrix.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n : int, optional\n"
               "    Maximum number of steps to perform.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "(GivensRotation, ...)\n"
               "    Givens rotations in the order they were applied to the matrix.\n"
               "    This means that for the solution, they should be applied in the\n"
               "    reversed order.\n"},
    {.ml_name = "block_diag",
     .ml_meth = (void *)lil_mat_block_diagonal,
     .ml_flags = METH_FASTCALL | METH_CLASS,
     .ml_doc = "block_diag(*blocks: LiLMatrix) -> LiLMatrix:\n"
               "Construct a new matrix from blocks along the diagonal.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "*blocks : LiLMatrix\n"
               "    Block matrices. These are placed on the diagonal of the resulting matrix.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "LiLMatrix\n"
               "    Block diagonal matrix resulting from the blocks.\n"},
    {.ml_name = "add_columns",
     .ml_meth = (void *)lil_matrix_add_columns,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "add_columns(self, *cols: SparseVector) -> None:\n"
               "Add columns to the matrix.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "*cols : SparseVectors\n"
               "    Columns to be added to the matrix.\n"},
    {.ml_name = "add_rows",
     .ml_meth = (void *)lil_matrix_add_rows,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "add_rows(*rows: SparseVector) -> LiLMatrix:\n"
               "Create a new matrix with added rows.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "*rows : SparaseVector\n"
               "    Rows to be added.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "LiLMatrix\n"
               "    Matrix with new rows added.\n"},
    {.ml_name = "solve_upper_triangular",
     .ml_meth = (void *)lil_matrix_solve_upper_triangular,
     .ml_flags = METH_KEYWORDS | METH_VARARGS,
     .ml_doc = "solve_upper_triangular(rsh: array_like, /, out: array | None = None) -> array:\n"
               "Use back-substitution to solve find the right side.\n"
               "\n"
               "This assumes the matrix is upper triangualr.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "rhs : array_like\n"
               "    Vector or matrix that gives the right side of the equation.\n"
               "\n"
               "out : array, optional\n"
               "    Array to be used as output. If not given a new one will be created and\n"
               "    returned, otherwise, the given value is returned. It must match the shape\n"
               "    of the input array exactly and have the correct data type.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "array\n"
               "    Vector or matrix that yields the rhs when matrix multiplication is used.\n"
               "    If the ``out`` parameter is given, the value returned will be exactly that\n"
               "    matrix.\n"},
    {.ml_name = "empty_diagonal",
     .ml_meth = (void *)lil_mat_empty_diagonal,
     .ml_flags = METH_CLASS | METH_O,
     .ml_doc = "empty_diagonal(n: int, /) -> Self:\n"
               "Create empty square matrix with zeros on the diagonal.\n"
               "\n"
               "This is intended for padding that allows for computing QR decompositions.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n : int\n"
               "    Size of the square matrix.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "LiLMatrix\n"
               "    Sparse matrix that is square and has only zeros on its diagonal.\n"},
    {.ml_name = "to_scipy",
     .ml_meth = (PyCFunction)lil_mat_to_scipy,
     .ml_flags = METH_NOARGS,
     .ml_doc = "to_scipy() -> (array, array):\n"
               "Convert itself into an array of values and an array of coorinates.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "(N,) array of floats\n"
               "    Values of the entries stored in the matrx.\n"
               "\n"
               "(N, 2) array of uint64\n"
               "    Positions of entries as ``(row, col)``.\n"},
    {}, // Sentinel
};

static PyObject *lil_mat_get_shape(const lil_mat_object_t *this, void *Py_UNUSED(closure))
{
    return PyTuple_Pack(2, PyLong_FromSize_t((size_t)this->rows), PyLong_FromSize_t((size_t)this->cols));
}

static PyObject *lil_mat_get_usage(const lil_mat_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)compute_usage(this));
}

static PyGetSetDef lil_mat_getset[] = {
    {.name = "shape",
     .get = (getter)lil_mat_get_shape,
     .set = NULL,
     .doc = "(int, int) : Get the shape of the matrix.",
     .closure = NULL},
    {.name = "usage",
     .get = (getter)lil_mat_get_usage,
     .set = NULL,
     .doc = "int : Number of non-zero entries.",
     .closure = NULL},
    {}, // Sentinel
};

PyTypeObject lil_mat_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.LiLMatrix",
    .tp_basicsize = sizeof(lil_mat_object_t),
    .tp_itemsize = sizeof(svector_t),
    .tp_repr = (reprfunc)lil_mat_repr,
    .tp_as_mapping = &lil_mat_mapping,
    // .tp_str = ,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE,
    // .tp_doc = ,
    // .tp_richcompare = ,
    // .tp_iter = ,
    // .tp_iternext = ,
    .tp_methods = lil_mat_methods,
    .tp_getset = lil_mat_getset,
    .tp_new = lil_mat_new,
    .tp_finalize = (destructor)lil_mat_delete,
};
