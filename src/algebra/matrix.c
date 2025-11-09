//
// Created by jan on 20.2.2025.
//

#include "matrix.h"
#include "../evaluation/incidence.h"

PyArrayObject *matrix_full_to_array(const matrix_full_t *mat)
{
    const npy_intp dims[2] = {mat->base.rows, mat->base.cols};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;

    double *const restrict p_out = PyArray_DATA(out);
    memcpy(p_out, mat->data, sizeof(*p_out) * mat->base.rows * mat->base.cols);
    return out;
}

mfv2d_result_t matrix_full_copy(const matrix_full_t *this, matrix_full_t *out, const allocator_callbacks *allocator)
{
    double *const restrict ptr = allocate(allocator, sizeof(*ptr) * this->base.rows * this->base.cols);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;
    memcpy(ptr, this->data, sizeof(*ptr) * this->base.rows * this->base.cols);
    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = this->base.rows, .cols = this->base.cols},
                           .data = ptr};
    return MFV2D_SUCCESS;
}

mfv2d_result_t matrix_full_multiply(const matrix_full_t *left, const matrix_full_t *right, matrix_full_t *out,
                                    const allocator_callbacks *allocator)
{
    if (left->base.cols != right->base.rows)
        return MFV2D_DIMS_MISMATCH;

    const unsigned n = right->base.rows;

    const unsigned n_rows = left->base.rows;
    const unsigned n_cols = right->base.cols;
    double *const restrict ptr = allocate(allocator, sizeof(*ptr) * n_rows * n_cols);
    const matrix_full_t this = {.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};

    for (unsigned row = 0; row < n_rows; ++row)
    {
        for (unsigned col = 0; col < n_cols; ++col)
        {
            double v = 0.0;
            for (unsigned k = 0; k < n; ++k)
            {
                v += left->data[row * n + k] * right->data[k * n_cols + col];
            }
            ptr[row * n_cols + col] = v;
        }
    }

    *out = this;
    return MFV2D_SUCCESS;
}

mfv2d_result_t matrix_full_add_inplace(const matrix_full_t *in, matrix_full_t *out)
{
    if (in->base.cols != out->base.cols || in->base.rows != out->base.rows)
        return MFV2D_DIMS_MISMATCH;

    for (unsigned row = 0; row < in->base.rows; ++row)
    {
        for (unsigned col = 0; col < in->base.cols; ++col)
        {
            out->data[row * out->base.cols + col] += in->data[row * out->base.cols + col];
        }
    }

    return MFV2D_SUCCESS;
}

mfv2d_result_t matrix_multiply(error_stack_t *error_stack, const unsigned order, const matrix_t *right,
                               const matrix_t *left, matrix_t *out, const allocator_callbacks *allocator)
{
    (void)error_stack;
    double k_right = right->coefficient, k_left = left->coefficient;
    mfv2d_result_t res = MFV2D_SUCCESS;
    switch (right->type)
    {
    case MATRIX_TYPE_IDENTITY:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY};
            break;

        case MATRIX_TYPE_INCIDENCE:
            *out = *left;
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_copy((matrix_full_t *)left, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INVALID:
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY};
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_INCIDENCE:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = *right;
            break;

        case MATRIX_TYPE_INCIDENCE: {
            matrix_full_t tmp;
            if ((res = incidence_to_full(right->incidence.incidence, order, &tmp, allocator)) != MFV2D_SUCCESS)
            {
                MFV2D_ERROR(error_stack, res, "Could not make a full incidence matrix %u.", right->incidence.incidence);
                return res;
            }
            res = apply_incidence_to_full_left(left->incidence.incidence, order, &tmp, (matrix_full_t *)out, allocator);
            deallocate(allocator, tmp.data);
        }
        break;

        case MATRIX_TYPE_FULL:
            res = apply_incidence_to_full_right(right->incidence.incidence, order, &left->full, (matrix_full_t *)out,
                                                allocator);
            break;

        case MATRIX_TYPE_INVALID:
            *out = *right;
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_FULL:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            res = matrix_full_copy(&right->full, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INCIDENCE:
            res = apply_incidence_to_full_left(left->incidence.incidence, order, &right->full, (matrix_full_t *)out,
                                               allocator);
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_multiply(&left->full, &right->full, (matrix_full_t *)out, allocator);
            break;

        case MATRIX_TYPE_INVALID:
            res = matrix_full_copy(&right->full, (matrix_full_t *)out, allocator);
            k_left = 1.0;
            break;
        }
        break;

    case MATRIX_TYPE_INVALID:
        switch (left->type)
        {
        case MATRIX_TYPE_IDENTITY:
            *out = *left;
            k_right = 1.0;
            break;

        case MATRIX_TYPE_INCIDENCE:
            *out = *left;
            k_right = 1.0;
            break;

        case MATRIX_TYPE_FULL:
            res = matrix_full_copy((matrix_full_t *)left, (matrix_full_t *)out, allocator);
            k_right = 1.0;
            break;

        case MATRIX_TYPE_INVALID:
            return MFV2D_BAD_ENUM;
        }
        break;
    }
    out->coefficient = k_right * k_left;
    return res;
}

mfv2d_result_t matrix_add(const unsigned order, matrix_t *right, matrix_t *left, matrix_t *out,
                          const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if (left->type == MATRIX_TYPE_INVALID || right->type == MATRIX_TYPE_INVALID)
        return MFV2D_BAD_ENUM;

    int free_left = 1, free_right = 1;
    if (left->type == MATRIX_TYPE_FULL)
    {
        free_left = 0;
        matrix_multiply_inplace(&left->full, left->coefficient);
        left->coefficient = 1.0;
    }

    if (right->type == MATRIX_TYPE_FULL)
    {
        free_right = 0;
        matrix_multiply_inplace(&right->full, right->coefficient);
        right->coefficient = 1.0;
    }

    if (left->type == right->type)
    {
        if (left->type == MATRIX_TYPE_IDENTITY)
        {
            *out = (matrix_t){.type = MATRIX_TYPE_IDENTITY, .coefficient = left->coefficient + right->coefficient};
            return MFV2D_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_INCIDENCE && left->incidence.incidence == right->incidence.incidence)
        {
            *out = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = left->incidence.incidence},
                              .coefficient = left->coefficient + right->coefficient};
            return MFV2D_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_FULL)
        {
            mfv2d_result_t res = matrix_full_copy(&right->full, &out->full, allocator);
            if (res != MFV2D_SUCCESS)
            {
                return res;
            }
            out->coefficient = 1.0;
            out->type = MATRIX_TYPE_FULL;
            res = matrix_full_add_inplace(&left->full, &out->full);
            return res;
        }
    }

    if (!free_left && !free_right)
    {
        return MFV2D_WRONG_MAT_TYPES;
    }

    matrix_full_t full_old;
    matrix_t *to_convert;
    if (free_left)
    {
        to_convert = left;
        full_old = right->full;
    }
    else
    {
        to_convert = right;
        full_old = left->full;
    }

    if (to_convert->type == MATRIX_TYPE_IDENTITY)
    {
        if (full_old.base.rows != full_old.base.cols)
            return MFV2D_DIMS_MISMATCH;

        res = matrix_full_copy(&full_old, &out->full, allocator);
        if (res != MFV2D_SUCCESS)
            return res;
        matrix_add_diagonal_inplace(&out->full, to_convert->coefficient);
    }

    // to_convert->type == MATRIX_TYPE_INCIDENCE
    res = incidence_to_full(to_convert->incidence.incidence, order, &out->full, allocator);
    if (res != MFV2D_SUCCESS)
        return res;
    res = matrix_full_add_inplace(&full_old, &out->full);
    out->coefficient = 1.0;

    return res;
}

void matrix_multiply_inplace(const matrix_full_t *this, const double k)
{
    if (k == 1.0)
        return;
    for (unsigned i = 0; i < this->base.rows * this->base.cols; ++i)
    {
        this->data[i] *= k;
    }
}

void matrix_add_diagonal_inplace(const matrix_full_t *this, const double k)
{
    if (k == 0)
        return;
    const unsigned n = this->base.cols > this->base.rows ? this->base.rows : this->base.cols;
    for (unsigned i = 0; i < n; ++i)
    {
        this->data[i * this->base.cols + i] += k;
    }
}

void matrix_cleanup(matrix_t *this, const allocator_callbacks *allocator)
{
    if (this->type == MATRIX_TYPE_FULL)
    {
        deallocate(allocator, this->full.data);
    }
    *this = (matrix_t){};
}

static const char *const incidence_type_strings[INCIDENCE_TYPE_CNT] = {
    [INCIDENCE_TYPE_10] = "INCIDENCE_TYPE_10",
    [INCIDENCE_TYPE_21] = "INCIDENCE_TYPE_21",
    [INCIDENCE_TYPE_10_T] = "INCIDENCE_TYPE_10_T",
    [INCIDENCE_TYPE_21_T] = "INCIDENCE_TYPE_21_T",
};

const char *incidence_type_str(const incidence_type_t type)
{
    if (type < INCIDENCE_TYPE_10 || type > INCIDENCE_TYPE_21_T)
        return "Invalid";

    // vvv Glorious AI slop (yeah, let me just return a STACK ALLOCATED ARRAY, surely that works!)
    // return (const char[]){type - INCIDENCE_TYPE_10 + '0', '\0'};
    return incidence_type_strings[type];
}
void matrix_print(const matrix_t *mtx)
{
    switch (mtx->type)
    {
    case MATRIX_TYPE_FULL: {
        const matrix_full_t *this = &mtx->full;
        printf("Full matrix (%u, %u):\n", this->base.rows, this->base.cols);
        for (unsigned i = 0; i < this->base.rows; ++i)
        {
            printf("\t");
            for (unsigned j = 0; j < this->base.cols; ++j)
            {
                printf("%g ", this->data[i * this->base.cols + j]);
            }
            printf("\n");
        }
    }
    break;

    case MATRIX_TYPE_INVALID:
        printf("Invalid matrix\n");
        break;

    case MATRIX_TYPE_IDENTITY:
        printf("Identity matrix\n");
        break;

    case MATRIX_TYPE_INCIDENCE: {
        const matrix_incidence_t *this = &mtx->incidence;
        const unsigned base =
            this->incidence < INCIDENCE_TYPE_10_T ? this->incidence : this->incidence - INCIDENCE_TYPE_10_T;
        printf("Incidence matrix E(%u, %u)%s\n", base + 1, base, this->incidence >= INCIDENCE_TYPE_10_T ? "^T" : "");
    }
    break;
    }
}

MFV2D_INTERNAL
mfv2d_result_t matrix_full_invert(const matrix_full_t *this, matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    if (this->base.rows != this->base.cols)
        return MFV2D_DIMS_MISMATCH;
    const matrix_full_t out = {.base = this->base,
                               .data = allocate(allocator, sizeof(*out.data) * this->base.rows * this->base.cols)};
    if (!out.data)
        return MFV2D_FAILED_ALLOC;
    memset(out.data, 0, sizeof(*out.data) * this->base.rows * this->base.cols);

    double *const buffer = allocate(allocator, sizeof(*out.data) * this->base.rows * this->base.cols);
    if (!buffer)
    {
        deallocate(allocator, out.data);
        return MFV2D_FAILED_ALLOC;
    }
    unsigned *pivots = allocate(allocator, sizeof(*pivots) * this->base.rows);
    if (!pivots)
    {
        deallocate(allocator, out.data);
        deallocate(allocator, buffer);
        return MFV2D_FAILED_ALLOC;
    }
    const mfv2d_result_t res = decompose_pivoted_lu(this->base.rows, this->data, buffer, pivots, 1e-12);
    if (res != MFV2D_SUCCESS)
    {
        deallocate(allocator, out.data);
        deallocate(allocator, buffer);
        deallocate(allocator, pivots);
        return res;
    }
    // for (unsigned i = 0; i < out.base.rows; ++i)
    // {
    //     out.data[i * out.base.cols + i] = 1.0;
    // }
    // unpivot_matrix(this->base.rows, this->base.cols, pivots, buffer, out.data);

    // Manually perform what pivots would do to the identity matrix.
    for (unsigned i = 0; i < out.base.rows; ++i)
    {
        out.data[i * out.base.cols + pivots[i]] = 1.0;
    }
    deallocate(allocator, pivots);

    solve_lu(this->base.rows, this->base.cols, buffer, out.data, out.data);
    deallocate(allocator, buffer);

    *p_out = out;
    return MFV2D_SUCCESS;
}

/**
 * Decompose the matrix into a pivoted LU decomposition, where the diagonal of L is assumed to be 1.
 *
 * @param n Dimension of the matrix.
 * @param mat Matrix which to invert. Can be equal to ``out``.
 * @param lu Buffer that receives the decomposed LU values.
 * @param pivots Order of rows that serve as pivots
 * @param pivot_tol The lowest acceptable magnitude that the pivot can have before an error is returned.
 * @return MFV2D_SUCCESS if successful. Failure can occur due to not being able to find the pivots
 * for all rows, which can be the fault of the algorithm.
 */
MFV2D_INTERNAL
mfv2d_result_t decompose_pivoted_lu(const size_t n, const double mat[static n * n], double lu[restrict n * n],
                                    unsigned pivots[n], const double pivot_tol)
{
    // Find pivot rows using a greedy algorithm
    unsigned unpivoted_min = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned pivot = n;
        double pivot_val = -1.0;
        for (unsigned j = unpivoted_min; j < n; ++j)
        {
            const double new_val = fabs(mat[j * n + i]);
            if (new_val > pivot_val)
            {
                unsigned found_pos;
                for (found_pos = 0; found_pos < i; ++found_pos)
                {
                    if (pivots[found_pos] == j)
                    {
                        break;
                    }
                }
                if (found_pos == i)
                {
                    pivot = j;
                    pivot_val = new_val;
                }
            }
        }
        if (pivot_val < pivot_tol)
        {
            return MFV2D_PIVOT_FAILED;
        }
        if (pivot == unpivoted_min)
        {
            unpivoted_min += 1;
        }
        pivots[i] = pivot;
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        //  Deal with a row of L
        for (uint_fast32_t j = 0; j < i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *li = lu + n * i;
            //  Column of U
            const double *uj = lu + j;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += li[k] * uj[k * n];
            }
            lu[n * i + j] = (mat[n * pivots[i] + j] - v) / uj[n * j];
        }

        //  Deal with a column of U
        for (uint_fast32_t j = 0; j <= i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *lj = lu + n * j;
            //  Column of U
            const double *ui = lu + i;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += lj[k] * ui[k * n];
            }
            lu[n * j + i] = mat[n * pivots[j] + i] - v;
        }
    }
    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
void solve_lu(const size_t n, const size_t m, const double lu[static restrict n * n], const double rhs[static n * m],
              double lhs[n * m])
{
    // First solve `L @ B = C` for `B`

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < m; ++j)
        {
            double v = rhs[i * m + j];
#pragma omp simd reduction(- : v)
            for (unsigned k = 0; k < i; ++k)
            {
                v -= lu[i * n + k] * lhs[k * m + j];
            }
            lhs[i * m + j] = v;
        }
    }
    // Now solve `U @ (A^{-1}) = B` for `A^{-1}`

    for (unsigned i = n; i > 0; --i)
    {
        for (unsigned j = 0; j < m; ++j)
        {
            double v = lhs[(i - 1) * m + j];
#pragma omp simd reduction(- : v)
            for (unsigned k = i; k < n; ++k)
            {
                v -= lu[(i - 1) * n + k] * lhs[k * m + j];
            }
            lhs[(i - 1) * m + j] = v / lu[(i - 1) * n + (i - 1)];
        }
    }
}

MFV2D_INTERNAL
void unpivot_matrix(const size_t n, const size_t m, const unsigned pivots[static restrict n],
                    const double in[restrict static n * m], double out[restrict n * m])
{
    for (unsigned i = 0; i < n; ++i)
    {
        memcpy(out + i * m, in + pivots[i] * m, sizeof(*out) * m);
    }
}

MFV2D_INTERNAL
PyObject *python_compute_matrix_inverse(PyObject *Py_UNUSED(self), PyObject *arg)
{
    PyArrayObject *const array =
        (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 2, 2, (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!array)
        return NULL;

    const unsigned rows = PyArray_DIM(array, 0);
    const unsigned cols = PyArray_DIM(array, 1);
    if (rows != cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix must be square");
        return NULL;
    }

    const double *const data = (const double *)PyArray_DATA(array);

    matrix_full_t out;
    const mfv2d_result_t res = matrix_full_invert(
        &(matrix_full_t){.base = {MATRIX_TYPE_FULL, rows, cols}, .data = (double *)data}, &out, &SYSTEM_ALLOCATOR);
    Py_DECREF(array);
    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to invert %u by %u matrix: %s", rows, cols, mfv2d_result_str(res));
        return NULL;
    }
    PyArrayObject *const out_array = matrix_full_to_array(&out);
    deallocate(&SYSTEM_ALLOCATOR, out.data);
    return (PyObject *)out_array;
}

MFV2D_INTERNAL
PyObject *python_solve_linear_system(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *py_mat, *py_rhs;
    if (!PyArg_ParseTuple(args, "OO", &py_mat, &py_rhs))
        return NULL;

    PyArrayObject *const mat =
        (PyArrayObject *)PyArray_FROMANY(py_mat, NPY_DOUBLE, 2, 2, (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!mat)
        return NULL;

    PyArrayObject *const rhs =
        (PyArrayObject *)PyArray_FROMANY(py_rhs, NPY_DOUBLE, 1, 2, (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!rhs)
    {
        Py_DECREF(mat);
        return NULL;
    }

    const unsigned mat_rows = PyArray_DIM(mat, 0);
    const unsigned mat_cols = PyArray_DIM(mat, 1);
    if (mat_rows != mat_cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix must be square");
        return NULL;
    }

    const unsigned rhs_rows = PyArray_DIM(rhs, 0);
    const unsigned rhs_cols = PyArray_NDIM(rhs) == 1 ? 1 : PyArray_DIM(rhs, 1);
    if (mat_cols != rhs_rows)
    {
        PyErr_Format(PyExc_ValueError, "Right-hand side must have the same number of columns as the matrix (%u vs %u)",
                     mat_cols, rhs_rows);
        Py_DECREF(mat);
        Py_DECREF(rhs);
        return NULL;
    }

    const npy_intp out_dims[2] = {rhs_rows, rhs_cols};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(rhs), out_dims, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(mat);
        Py_DECREF(rhs);
        return NULL;
    }
    double *const out_ptr = PyArray_DATA(out);
    const double *const mat_ptr = (const double *)PyArray_DATA(mat);
    const double *const rhs_ptr = (const double *)PyArray_DATA(rhs);

    double *const decomp = allocate(&SYSTEM_ALLOCATOR, sizeof(*decomp) * mat_rows * mat_cols);
    unsigned *const pivots = allocate(&SYSTEM_ALLOCATOR, sizeof(*pivots) * mat_rows);
    if (!decomp || !pivots)
    {
        deallocate(&SYSTEM_ALLOCATOR, decomp);
        deallocate(&SYSTEM_ALLOCATOR, pivots);
        Py_DECREF(mat);
        Py_DECREF(rhs);
        Py_DECREF(out);
        return NULL;
    }
    const mfv2d_result_t res = decompose_pivoted_lu(mat_rows, mat_ptr, decomp, pivots, 1e-12);
    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not decompose the matrix due to pivoting failing: %s.",
                     mfv2d_result_str(res));
        deallocate(&SYSTEM_ALLOCATOR, decomp);
        deallocate(&SYSTEM_ALLOCATOR, pivots);
        Py_DECREF(mat);
        Py_DECREF(rhs);
        Py_DECREF(out);
        return NULL;
    }

    unpivot_matrix(mat_rows, rhs_cols, pivots, rhs_ptr, out_ptr);
    solve_lu(mat_rows, rhs_cols, decomp, out_ptr, out_ptr);
    deallocate(&SYSTEM_ALLOCATOR, pivots);
    deallocate(&SYSTEM_ALLOCATOR, decomp);
    Py_DECREF(mat);
    Py_DECREF(rhs);
    return (PyObject *)out;
}

MFV2D_INTERNAL
const char compute_matrix_inverse_docstr[] =
    "_compute_matrix_inverse(x: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray[numpy.double]\n"
    "Compute inverse of a matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "x : array_like\n"
    "    Matrix to invert.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "Inverse of the matrix, which when multiplied with ``x`` should return\n"
    "the identity matrix (or when accounting for numerical/rounding errors, be very\n"
    "close to it).\n";

MFV2D_INTERNAL
const char solve_linear_system_docstr[] = "_solve_linear_system(mat: numpy.typing.ArrayLike, rhs: "
                                          "numpy.typing.ArrayLike, /) -> numpy.typing.NDArray[numpy.double]\n"
                                          "Solve a linear system.\n"
                                          "\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "x : array_like\n"
                                          "    System matrix.\n"
                                          "\n"
                                          "rhs : array_like\n"
                                          "    Right side of the system."
                                          "\n"
                                          "Returns\n"
                                          "-------\n"
                                          "array\n"
                                          "    Matrix/vector, which when pre-multiplied with the system matrix\n"
                                          "    results in a vector equal to the ``rhs``, after accounting for\n"
                                          "    rounding errors.\n";
