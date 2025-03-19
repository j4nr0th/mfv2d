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
        for (uint64_t j = 0; j < dims[1]; ++j)
        {
            this->row_data[row].entries[j] = (entry_t){.index = j, .value = ptr[row * dims[1] + j]};
        }
        this->row_data[row].count = dims[1];
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

static PyObject *lil_mat_qr_decomposition(lil_mat_object_t *this, PyObject *Py_UNUSED(args))
{
    const lil_matrix_t tmp = {.rows = this->rows, .cols = this->cols, .row_data = this->row_data};
    uint64_t n_givens;
    givens_rotation_t *p_givens;

    Py_BEGIN_ALLOW_THREADS;

    // Threads are allowed inside here, since this can be on the slow side.
    // As long as no one else has the bright idea to start messing with the
    // matrix in another thread we will be fine. If someone is retarded enough
    // to try that, may God have mercy on his soul.
    if (decompose_qr(&tmp, &n_givens, &p_givens, &SYSTEM_ALLOCATOR))
    {
        return NULL;
    }

    Py_END_ALLOW_THREADS;

    PyTupleObject *out_tuple = (PyTupleObject *)PyTuple_New((Py_ssize_t)n_givens);
    if (!out_tuple)
    {
        deallocate(&SYSTEM_ALLOCATOR, p_givens);
        return NULL;
    }

    for (uint64_t i = 0; i < n_givens; ++i)
    {
        givens_object_t *const g = givens_to_python(p_givens + i);
        if (!g)
        {
            Py_DECREF(out_tuple);
            out_tuple = NULL;
            break;
        }
        PyTuple_SET_ITEM(out_tuple, i, g);
    }
    deallocate(&SYSTEM_ALLOCATOR, p_givens);

    return (PyObject *)out_tuple;
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
     .ml_flags = METH_NOARGS,
     .ml_doc = "qr_decompose() -> tuple[GivensRotation, ...]:\n"
               "Decompose the matrix into a series of Givens rotations and a triangular matrix.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "(GivensRotation, ...)\n"
               "    Givens rotations in the order they were applied to the matrix.\n"
               "    This means that for the solution, they should be applied in the\n"
               "    reversed order.\n"},
    {}, // Sentinel
};

static PyObject *lil_mat_get_shape(const lil_mat_object_t *this, void *Py_UNUSED(closure))
{
    return PyTuple_Pack(2, PyLong_FromSize_t((size_t)this->rows), PyLong_FromSize_t((size_t)this->cols));
}

static PyGetSetDef lil_mat_getset[] = {
    {.name = "shape",
     .get = (getter)lil_mat_get_shape,
     .set = NULL,
     .doc = "(int, int) : Get the shape of the matrix.",
     .closure = NULL},
    {}, // Sentinel
};

PyTypeObject lil_mat_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.LiLMatrix",
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
