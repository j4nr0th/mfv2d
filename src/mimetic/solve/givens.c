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

int givens_rotate_sparse_vector(const givens_rotation_t *g, const svector_t *vec, svector_t *out,
                                const allocator_callbacks *allocator)
{
    ASSERT(out->n == vec->n, "Input and output vectors should have the same dimensions.");
    ASSERT(g->n == vec->n, "Givens rotation and the vectors must have the same dimensions.");
    const uint64_t max_size = vec->count + 1 ? vec->count < vec->n : vec->n;

    if (sparse_vec_resize(out, max_size, allocator))
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

    scalar_t v1 = 0.0, v2 = 0.0;
    int b1 = 0;
    uint64_t p_out, p_in;
    for (p_out = 0, p_in = 0; p_in < vec->count && vec->entries[p_in].index < i1; ++p_in)
    {
        const entry_t *entry = vec->entries + p_in;
        if (entry->value == 0.0)
            continue;

        out->entries[p_out] = *entry;
        p_out += 1;
    }
    if (p_in == vec->count)
    {
        // Nothing in the range of rotation
        out->count = p_out;
        return 0;
    }
    const uint64_t p1 = p_out;
    if (p_in < vec->count && vec->entries[p_in].index == i1)
    {
        b1 = 1;
        v1 = vec->entries[p_in].value;
    }

    for (; p_in < vec->count && vec->entries[p_in].index < i2; ++p_in)
    {
        const entry_t *entry = vec->entries + p_in;
        if (entry->value == 0.0)
            continue;

        out->entries[p_out] = *entry;
        p_out += 1;
    }
    if (p_in == vec->count && b1 == 0)
    {
        // Nothing in the range of rotation
        out->count = p_out;
        return 0;
    }

    if (vec->entries[p_in].index == i2)
    {
        v2 = vec->entries[p_in].value;
        p_in += 1;
    }
    if (v1 != 0.0 || v2 != 0.0)
    {
        const scalar_t r1 = v1 * g00 + v2 * g01;
        const scalar_t r2 = v1 * g10 + v2 * g11;
        // Do the rotation now
        if (b1 == 0)
        {
            // Need space for the first entry, since there was a zero there before
            memmove(out->entries + p1 + 1, out->entries + p1, sizeof(*out->entries) * (p_out - p1));
            p_out += 1;
        }
        out->entries[p1] = (entry_t){.index = i1, .value = r1};
        out->entries[p_out] = (entry_t){.index = i2, .value = r2};
        p_out += 1;
    }

    // Finish it up
    for (; p_in < vec->count; ++p_in)
    {
        const entry_t *entry = vec->entries + p_in;
        if (entry->value == 0.0)
            continue;

        out->entries[p_out] = *entry;
        p_out += 1;
    }

    out->count = p_out;

    return 0;
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
        svector_t tmp;
        if (sparse_vector_new(&tmp, in->n, in->count + 1, &SYSTEM_ALLOCATOR))
        {
            return NULL;
        }

        if (givens_rotate_sparse_vector(&this->data, &iv, &tmp, &SYSTEM_ALLOCATOR))
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
