//
// Created by jan on 19.3.2025.
//
#include "givens.h"
#include <numpy/ndarrayobject.h>

int apply_givens_rotation(const scalar_t c, const scalar_t s, const svector_t *row_i, const svector_t *row_j,
                          svector_t *restrict out_i, svector_t *restrict out_j, const unsigned cut_j,
                          const allocator_callbacks *allocator)
{
    ASSERT(row_i->n == row_j->n, "Input vectors must have the same size (%" PRIu64 " vs %" PRIu64 ").", row_i->n,
           row_j->n);
    ASSERT(out_i->n == out_j->n, "Output vectors must have the same size (%" PRIu64 " vs %" PRIu64 ").", out_i->n,
           out_j->n);
    uint64_t max_elements = row_i->count + row_j->count;

    // Can't have more elements than the full row.
    if (max_elements > row_i->n)
    {
        max_elements = row_i->n;
    }

    if (sparse_vec_resize(out_i, max_elements, allocator) || sparse_vec_resize(out_j, max_elements, allocator))
    {
        return -1;
    }

    uint64_t idx_i, idx_j, pos;
    for (idx_i = 0, idx_j = 0, pos = 0; idx_i < row_i->count || idx_j < row_j->count; ++pos)
    {
        scalar_t vi = 0.0, vj = 0.0;
        uint64_t pv;
        if (idx_i < row_i->count)
        {
            // row I still available
            if (idx_j < row_j->count)
            {
                // row J still available
                if (row_i->entries[idx_i].index < row_j->entries[idx_j].index)
                {
                    // I before J
                    vi = row_i->entries[idx_i].value;
                    pv = row_i->entries[idx_i].index;
                    idx_i += 1;
                }
                else if (row_i->entries[idx_i].index > row_j->entries[idx_j].index)
                {
                    // J before I
                    vj = row_j->entries[idx_j].value;
                    pv = row_j->entries[idx_j].index;
                    idx_j += 1;
                }
                else
                {
                    // I and J equal
                    vi = row_i->entries[idx_i].value;
                    vj = row_j->entries[idx_j].value;
                    pv = row_j->entries[idx_j].index;
                    idx_i += 1;
                    idx_j += 1;
                }
            }
            else
            {
                // Only row I is left
                vi = row_i->entries[idx_i].value;
                pv = row_i->entries[idx_i].index;
                idx_i += 1;
            }
        }
        else
        {
            // Only row J is left
            vj = row_j->entries[idx_j].value;
            pv = row_j->entries[idx_j].index;
            idx_j += 1;
        }
        out_i->entries[pos].value = c * vi + s * vj;
        out_i->entries[pos].index = pv;
        // Skip the first cut_j the account of the fact that they are eliminated.
        if (pos >= cut_j)
        {
            out_j->entries[pos - 1].value = -s * vi + c * vj;
            out_j->entries[pos - 1].index = pv;
        }
    }
    // Adjust the element counts for out_i and out_j.
    out_i->count = pos;
    out_j->count = pos - cut_j;

    return 0;
}

givens_object_t *givens_to_python(const givens_rotation_t *g)
{
    givens_object_t *const this =
        (givens_object_t *)givens_rotation_type_object.tp_alloc(&givens_rotation_type_object, 0);
    if (!this)
        return this;
    this->data = *g;
    return this;
}

int givens_rotate_sparse_vector_inplace(const givens_rotation_t *g, svector_t *vec,
                                        const allocator_callbacks *allocator)
{
    ASSERT(g->n == vec->n, "Givens rotation and the vectors must have the same dimensions.");
    const uint64_t max_size = vec->count < vec->n ? vec->count + 1 : vec->n;

    if (vec->count == 0)
    {
        // Empty
        return 0;
    }

    if (sparse_vec_resize(vec, max_size, allocator))
    {
        return -1;
    }
    uint64_t i1, i2;
    scalar_t g00, g01, g10, g11;
    if (g->k < g->l)
    {
        i1 = g->k;
        i2 = g->l;
        g00 = g->c;
        g01 = +g->s;
        g10 = -g->s;
        g11 = g->c;
    }
    else
    {
        i1 = g->l;
        i2 = g->k;
        g00 = g->c;
        g01 = -g->s;
        g10 = +g->s;
        g11 = g->c;
    }

    int b1 = 0, b2 = 0;
    scalar_t v1 = 0.0, v2 = 0.0;
    uint64_t pos_2;
    const uint64_t pos_1 = sparse_vector_find_first_geq(vec, i1, 0);
    if (pos_1 == vec->count)
    {
        b1 = 0;
        pos_2 = sparse_vector_find_first_geq(vec, i2, 0);
    }
    else
    {
        pos_2 = sparse_vector_find_first_geq(vec, i2, pos_1);
        if (vec->entries[pos_1].index == i1)
        {
            v1 = vec->entries[pos_1].value;
            b1 = 1;
        }
    }

    if (pos_2 != vec->count && vec->entries[pos_2].index == i2)
    {
        b2 = 1;
        v2 = vec->entries[pos_2].value;
    }

    // Check if there's even anything to rotate
    if (v1 == 0.0 && v2 == 0.0)
        return 0;

    const scalar_t r1 = g00 * v1 + g01 * v2;
    const scalar_t r2 = g10 * v1 + g11 * v2;

    if (b1)
    {
        // Just set the existing entry
        vec->entries[pos_1].value = r1;
    }
    else if (r1 != 0.0)
    {
        // Move other entries out of the way
        // if (ASSERT(vec->capacity >= vec->count + 1, "Vector not large enough (n: %zu, cap: %zu, cnt: %zu).", vec->n,
        //            vec->capacity, vec->count))
        // {
        //     printf("I1: %zu, I2: %zu\n", i1, i2);
        //     for (uint64_t p = 0; p < vec->count; ++p)
        //     {
        //         printf("(%zu, %g)\n", vec->entries[p].index, vec->entries[p].value);
        //     }
        //     exit(EXIT_FAILURE);
        // }

        memmove(vec->entries + pos_1 + 1, vec->entries + pos_1, sizeof(*vec->entries) * (vec->count - pos_1));
        vec->entries[pos_1] = (entry_t){.index = i1, .value = r1};
        pos_2 += 1;
        vec->count += 1;
    }

    if (b2)
    {
        // Just set the existing entry
        vec->entries[pos_2].value = r2;
    }
    else if (r2 != 0.0)
    {
        // Move other entries out of the way
        // ASSERT(vec->capacity >= vec->count + 1, "Vector not large enough");

        memmove(vec->entries + pos_2 + 1, vec->entries + pos_2, sizeof(*vec->entries) * (vec->count - pos_2));
        vec->entries[pos_2] = (entry_t){.index = i2, .value = r2};
        vec->count += 1;
    }

    // if (ASSERT(vec->entries[pos_1].index == i1 && vec->entries[pos_1].index < vec->entries[pos_1 + 1].index, "") ||
    //     ASSERT(vec->entries[pos_2].index == i2 && vec->entries[pos_2].index > vec->entries[pos_2 - 1].index, ""))
    // {
    //     printf("Wrongly inserted entries.\n");
    //     printf("pos_1 = %zu\n", pos_1);
    //     printf("pos_2 = %zu\n", pos_2);
    //     printf("vec->entries[pos_1].index = %zu\n", vec->entries[pos_1].index);
    //     printf("i1 = %zu\n", i1);
    //     printf("vec->entries[pos_1 + 1].index = %zu\n", vec->entries[pos_1 + 1].index);
    //     printf("vec->entries[pos_2].index = %zu\n", vec->entries[pos_2].index);
    //     printf("i2 = %zu\n", i2);
    //     printf("vec->entries[pos_2 - 1].index = %zu\n", vec->entries[pos_2 - 1].index);
    //     exit(EXIT_FAILURE);
    // }

    return 0;
}
givens_series_t *givens_series_to_python(const uint64_t n, const givens_rotation_t g[static n])
{
    ASSERT(n > 0, "Must have more than one Givens rotation.");
    for (uint64_t i = 1; i < n; ++i)
    {
        if (g[0].n != g[i].n)
        {
            PyErr_Format(PyExc_ValueError, "Can not make a series of Givens rotations with different dimensions.");
            return NULL;
        }
    }
    givens_series_t *const this =
        (givens_series_t *)givens_series_type_object.tp_alloc(&givens_series_type_object, (Py_ssize_t)n);
    if (!this)
        return NULL;

    Py_SET_SIZE(this, n);
    this->n = n > 0 ? g[0].n : 0;
    memcpy(this->data, g, sizeof(*this->data) * n);

    return this;
}

static PyObject *givens_repr(const givens_object_t *this)
{
    char float_buffer1[32], float_buffer2[32];
    snprintf(float_buffer1, sizeof(float_buffer1), "%g", this->data.c);
    snprintf(float_buffer2, sizeof(float_buffer2), "%g", this->data.s);
    return PyUnicode_FromFormat("GivensRotation(%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %s, %s)", this->data.n,
                                this->data.k, this->data.l, float_buffer1, float_buffer2);
}

static PyObject *givens_as_array(const givens_object_t *this, PyObject *args, PyObject *kwds)
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

    const uint64_t n = this->data.n;
    const npy_intp sizes[2] = {(npy_intp)n, (npy_intp)n};

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, sizes, NPY_FLOAT64);
    if (!out)
        return NULL;
    npy_float64 *const ptr = PyArray_DATA(out);
    memset(ptr, 0, sizeof *ptr * sizes[0] * sizes[1]);
    for (uint64_t i = 0; i < n; ++i)
    {
        ptr[i * n + i] = 1.0;
    }

    ptr[this->data.k * n + this->data.k] = this->data.c;
    ptr[this->data.l * n + this->data.l] = this->data.c;
    ptr[this->data.k * n + this->data.l] = this->data.s;
    ptr[this->data.l * n + this->data.k] = -this->data.s;

    return (PyObject *)out;
}

static PyMethodDef givens_methods[] = {
    {.ml_name = "__array__",
     .ml_meth = (void *)givens_as_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray\n"
               "Convert the object into a full numpy matrix.\n"},
    {}, // Sentinel
};

static PyObject *givens_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    givens_rotation_t g;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KKKdd", (char *[6]){"n", "i1", "i2", "c", "s", NULL}, &g.n, &g.k,
                                     &g.l, &g.c, &g.s))
    {
        return NULL;
    }

    if (g.k >= g.n || g.l >= g.n)
    {
        PyErr_Format(PyExc_ValueError, "Givens rotation indices must be less than the dimension of the rotation.");
        return NULL;
    }

    const double mag = hypot(g.c, g.s);
    g.c /= mag;
    g.s /= mag;

    givens_object_t *const this = (givens_object_t *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    this->data = g;

    return (PyObject *)this;
}

static PyObject *givens_get_n(const givens_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->data.n);
}

static PyObject *givens_get_i1(const givens_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->data.k);
}

static PyObject *givens_get_i2(const givens_object_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->data.l);
}

static PyObject *givens_get_c(const givens_object_t *this, void *Py_UNUSED(closure))
{
    return PyFloat_FromDouble(this->data.c);
}

static PyObject *givens_get_s(const givens_object_t *this, void *Py_UNUSED(closure))
{
    return PyFloat_FromDouble(this->data.s);
}

static PyObject *givens_transpose(const givens_object_t *this, void *Py_UNUSED(closure))
{
    const givens_rotation_t g = {
        .n = this->data.n, .k = this->data.k, .l = this->data.l, .c = this->data.c, .s = -this->data.s};
    return (PyObject *)givens_to_python(&g);
}

static PyGetSetDef givens_getset[] = {
    {.name = "n", .get = (getter)givens_get_n, .set = NULL, .doc = "int : Dimension of the rotation.", .closure = NULL},
    {.name = "i1", .get = (getter)givens_get_i1, .set = NULL, .doc = "int : First index of rotation.", .closure = NULL},
    {.name = "i2",
     .get = (getter)givens_get_i2,
     .set = NULL,
     .doc = "int : Second index of rotation.",
     .closure = NULL},
    {.name = "c", .get = (getter)givens_get_c, .set = NULL, .doc = "int : Cosine rotation value.", .closure = NULL},
    {.name = "s", .get = (getter)givens_get_s, .set = NULL, .doc = "int : Sine rotation value.", .closure = NULL},
    {.name = "T",
     .get = (getter)givens_transpose,
     .set = NULL,
     .doc = "GivensRotation : Inverse rotation.",
     .closure = NULL},
    {}, // Sentinel
};

static PyObject *givens_matmul(const givens_object_t *this, PyObject *other)
{
    if (Py_IS_TYPE(other, &svec_type_object))
    {
        const svec_object_t *const in = (svec_object_t *)other;
        // Sparse vector
        const svector_t iv = {.n = in->n, .count = in->count, .capacity = 0, .entries = (entry_t *)in->entries};
        svector_t tmp = {};
        if (sparse_vector_copy(&iv, &tmp, &SYSTEM_ALLOCATOR))
        {
            return NULL;
        }

        if (givens_rotate_sparse_vector_inplace(&this->data, &tmp, &SYSTEM_ALLOCATOR))
        {
            sparse_vec_del(&tmp, &SYSTEM_ALLOCATOR);
            return NULL;
        }

        svec_object_t *const out = sparse_vec_to_python(&tmp);
        sparse_vec_del(&tmp, &SYSTEM_ALLOCATOR);
        if (!out)
            return NULL;

        return (PyObject *)out;
    }

    PyArrayObject *const mat = (PyArrayObject *)PyArray_FromAny(other, PyArray_DescrFromType(NPY_FLOAT64), 1, 2,
                                                                NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!mat)
        return NULL;

    const unsigned ndim = PyArray_NDIM(mat);
    const unsigned n_rows = PyArray_DIM(mat, 0);
    const unsigned n_cols = ndim == 1 ? 1 : PyArray_DIM(mat, 1);
    if (n_rows != this->data.n)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Givens rotation of size %" PRIu64
            " can not be applied to the array which does not have that as its first dimension (instead it had %u).",
            this->data.n, n_rows);
        Py_DECREF(mat);
        return NULL;
    }

    const npy_float64 *ptr_in = PyArray_DATA(mat);

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim, PyArray_DIMS(mat), NPY_FLOAT64);
    if (!out)
    {
        Py_DECREF(mat);
        return NULL;
    }
    npy_float64 *ptr_out = PyArray_DATA(out);
    memcpy(ptr_out, ptr_in, sizeof *ptr_out * n_rows * n_cols);
    uint64_t i1, i2;
    scalar_t c00, c01, c10, c11;
    if (this->data.k < this->data.l)
    {
        i1 = this->data.k;
        i2 = this->data.l;
        c00 = this->data.c;
        c01 = this->data.s;
        c11 = this->data.c;
        c10 = -this->data.s;
    }
    else
    {
        i1 = this->data.l;
        i2 = this->data.k;
        c00 = this->data.c;
        c01 = -this->data.s;
        c11 = this->data.c;
        c10 = this->data.s;
    }

    for (uint64_t i_col = 0; i_col < n_cols; ++i_col)
    {
        const npy_float64 v1 = ptr_in[i_col + i1 * n_cols];
        const npy_float64 v2 = ptr_in[i_col + i2 * n_cols];
        ptr_out[i_col + i1 * n_cols] = c00 * v1 + c01 * v2;
        ptr_out[i_col + i2 * n_cols] = c10 * v1 + c11 * v2;
    }

    Py_DECREF(mat);
    return (PyObject *)out;
}

static PyNumberMethods number_methods = {
    .nb_matrix_multiply = (binaryfunc)givens_matmul,
};

PyTypeObject givens_rotation_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.GivensRotation",
    .tp_basicsize = sizeof(givens_object_t),
    .tp_repr = (reprfunc)givens_repr,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    // .tp_doc = ,
    // .tp_richcompare = ,
    .tp_methods = givens_methods,
    .tp_getset = givens_getset,
    .tp_new = givens_new,
    .tp_as_number = &number_methods,
};

static PyObject *givens_series_repr(const givens_series_t *this)
{
    const Py_ssize_t cnt = Py_SIZE(this);
    return PyUnicode_FromFormat("GivensSeries(%" PRIu64 ", %" PRIu64 ")", cnt > 0 ? this->data[0].n : 0, cnt);
}

static PyObject *givens_series_matmul(const givens_series_t *this, PyObject *other)
{
    if (Py_IS_TYPE(other, &svec_type_object))
    {
        svector_t ov = {};
        Py_BEGIN_ALLOW_THREADS;
        const svec_object_t *const in = (svec_object_t *)other;
        // Sparse vector
        const svector_t copy_in = {.n = in->n, .count = in->count, .capacity = 0, .entries = (entry_t *)in->entries};

        if (sparse_vector_copy(&copy_in, &ov, &SYSTEM_ALLOCATOR))
        {
            sparse_vec_del(&ov, &SYSTEM_ALLOCATOR);
            return NULL;
        }

        for (uint64_t j = 0; j < Py_SIZE(this); ++j)
        {
            if (givens_rotate_sparse_vector_inplace(this->data + j, &ov, &SYSTEM_ALLOCATOR))
            {
                sparse_vec_del(&ov, &SYSTEM_ALLOCATOR);
                return NULL;
            }
        }
        Py_END_ALLOW_THREADS;

        svec_object_t *const out = sparse_vec_to_python(&ov);
        sparse_vec_del(&ov, &SYSTEM_ALLOCATOR);
        if (!out)
            return NULL;

        return (PyObject *)out;
    }

    PyArrayObject *const mat = (PyArrayObject *)PyArray_FromAny(other, PyArray_DescrFromType(NPY_FLOAT64), 1, 2,
                                                                NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!mat)
        return NULL;

    const unsigned ndim = PyArray_NDIM(mat);
    const unsigned n_rows = PyArray_DIM(mat, 0);
    const unsigned n_cols = ndim == 1 ? 1 : PyArray_DIM(mat, 1);
    if (Py_SIZE(this) > 0 && n_rows != this->n)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Givens rotation of size %" PRIu64
            " can not be applied to the array which does not have that as its first dimension (instead it had %u).",
            this->n, n_rows);
        Py_DECREF(mat);
        return NULL;
    }

    const npy_float64 *ptr_in = PyArray_DATA(mat);

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim, PyArray_DIMS(mat), NPY_FLOAT64);
    if (!out)
    {
        Py_DECREF(mat);
        return NULL;
    }
    npy_float64 *ptr_out = PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS;
    memcpy(ptr_out, ptr_in, sizeof *ptr_out * n_rows * n_cols);
    for (uint64_t j = 0; j < Py_SIZE(this); ++j)
    {
        uint64_t i1, i2;
        scalar_t c00, c01, c10, c11;
        const givens_rotation_t *g = this->data + j;
        if (g->k < g->l)
        {
            i1 = g->k;
            i2 = g->l;
            c00 = g->c;
            c01 = g->s;
            c11 = g->c;
            c10 = -g->s;
        }
        else
        {
            i1 = g->l;
            i2 = g->k;
            c00 = g->c;
            c01 = -g->s;
            c11 = g->c;
            c10 = g->s;
        }

        for (uint64_t i_col = 0; i_col < n_cols; ++i_col)
        {
            const npy_float64 v1 = ptr_out[i_col + i1 * n_cols];
            const npy_float64 v2 = ptr_out[i_col + i2 * n_cols];
            ptr_out[i_col + i1 * n_cols] = c00 * v1 + c01 * v2;
            ptr_out[i_col + i2 * n_cols] = c10 * v1 + c11 * v2;
        }
    }
    Py_END_ALLOW_THREADS;
    Py_DECREF(mat);
    return (PyObject *)out;
}

static PyNumberMethods number_series_methods = {
    .nb_matrix_multiply = (binaryfunc)givens_series_matmul,
};

static PyObject *givens_series_get_n(const givens_series_t *this, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t((size_t)this->n);
}

static PyGetSetDef givens_series_getset[] = {
    {.name = "n",
     .get = (getter)givens_series_get_n,
     .set = NULL,
     .doc = "int : Dimension of the rotation.",
     .closure = NULL},
    {}, // Sentinel
};

static PyObject *givens_series_get(const givens_series_t *this, Py_ssize_t idx)
{
    if (PyErr_Occurred())
        return NULL;

    if (idx >= Py_SIZE(this) || idx < -Py_SIZE(this))
    {
        PyErr_Format(PyExc_IndexError, "Index %zd is out of bounds for a series of a length of %zd.", idx,
                     Py_SIZE(this));
        return NULL;
    }

    if (idx < 0)
    {
        idx = Py_SIZE(this) - idx;
    }

    return (PyObject *)givens_to_python(this->data + idx);
}

static Py_ssize_t givens_series_len(const givens_series_t *this)
{
    return Py_SIZE(this);
}

static PySequenceMethods series_seq_methods = {
    .sq_length = (lenfunc)givens_series_len,
    .sq_item = (ssizeargfunc)givens_series_get,
};

static PyObject *givens_series_new(PyTypeObject *type, PyObject *args, const PyObject *const kwds)
{
    if (kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "GivensSeries.__new__ takes no keyword arguments.");
        return NULL;
    }
    PyObject *const seq = PySequence_Fast(args, "Arguments must be given as a sequence.");
    const Py_ssize_t nargs = PySequence_Fast_GET_SIZE(seq);
    if (nargs == 1)
    {
        const size_t n = PyLong_AsSize_t(PySequence_Fast_GET_ITEM(args, 0));
        if (!PyErr_Occurred())
        {
            givens_series_t *const this = (givens_series_t *)type->tp_alloc(type, 0);
            if (!this)
                return NULL;
            this->n = n;
            return (PyObject *)this;
        }
        PyErr_Clear();
    }
    const givens_object_t *const *rotations = (const givens_object_t *const *)PySequence_Fast_ITEMS(seq);

    for (Py_ssize_t i = 0; i < nargs; ++i)
    {
        if (!Py_IS_TYPE(rotations[i], &givens_rotation_type_object))
        {
            PyErr_Format(PyExc_TypeError,
                         "All parameters must be of type GivensRotations, but argument %zd had the type %R.", i,
                         Py_TYPE(rotations[i]));
            Py_DECREF(seq);
            return NULL;
        }
    }

    givens_series_t *const this = (givens_series_t *)type->tp_alloc(type, nargs);
    if (!this)
    {
        Py_DECREF(seq);
        return NULL;
    }

    Py_SET_SIZE(this, nargs);

    for (Py_ssize_t i = 0; i < nargs; ++i)
    {
        this->data[i] = rotations[i]->data;
    }
    this->n = nargs ? this->data[0].n : 0;
    Py_DECREF(seq);

    return (PyObject *)this;
}

PyTypeObject givens_series_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.GivensSeries",
    .tp_basicsize = sizeof(givens_series_t),
    .tp_itemsize = sizeof(givens_rotation_t),
    .tp_repr = (reprfunc)givens_series_repr,
    .tp_flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    // .tp_doc = ,
    // .tp_richcompare = ,
    .tp_new = (newfunc)givens_series_new,
    .tp_getset = givens_series_getset,
    .tp_as_sequence = &series_seq_methods,
    .tp_as_number = &number_series_methods,
};
