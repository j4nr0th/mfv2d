//
// Created by jan on 20.2.2025.
//

#include "matrix.h"
#include "incidence.h"

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

eval_result_t matrix_full_copy(const matrix_full_t *this, matrix_full_t *out, const allocator_callbacks *allocator)
{
    double *const restrict ptr = allocate(allocator, sizeof(*ptr) * this->base.rows * this->base.cols);
    if (!ptr)
        return EVAL_FAILED_ALLOC;
    memcpy(ptr, this->data, sizeof(*ptr) * this->base.rows * this->base.cols);
    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = this->base.rows, .cols = this->base.cols},
                           .data = ptr};
    return EVAL_SUCCESS;
}

eval_result_t matrix_full_multiply(const matrix_full_t *left, const matrix_full_t *right, matrix_full_t *out,
                                   const allocator_callbacks *allocator)
{
    if (left->base.cols != right->base.rows)
        return EVAL_DIMS_MISMATCH;

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
    return EVAL_SUCCESS;
}

eval_result_t matrix_full_add_inplace(const matrix_full_t *in, matrix_full_t *out)
{
    if (in->base.cols != out->base.cols || in->base.rows != out->base.rows)
        return EVAL_DIMS_MISMATCH;

    for (unsigned row = 0; row < in->base.rows; ++row)
    {
        for (unsigned col = 0; col < in->base.cols; ++col)
        {
            out->data[row * out->base.cols + col] += in->data[row * out->base.cols + col];
        }
    }

    return EVAL_SUCCESS;
}

eval_result_t matrix_multiply(error_stack_t *error_stack, const unsigned order, const matrix_t *right,
                              const matrix_t *left, matrix_t *out, const allocator_callbacks *allocator)
{
    (void)error_stack;
    double k_right = right->coefficient, k_left = left->coefficient;
    eval_result_t res = EVAL_SUCCESS;
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
            if ((res = incidence_to_full(right->incidence.incidence, order, &tmp, allocator)) != EVAL_SUCCESS)
            {
                EVAL_ERROR(error_stack, res, "Could not make a full incidence matrix %u.", right->incidence.incidence);
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
            return EVAL_BAD_ENUM;
        }
        break;
    }
    out->coefficient = k_right * k_left;
    return res;
}

// TODO: THIS SHOULD ADD, NOT MULTIPLY!!!
eval_result_t matrix_add(const unsigned order, matrix_t *right, matrix_t *left, matrix_t *out,
                         const allocator_callbacks *allocator)
{
    eval_result_t res = EVAL_SUCCESS;
    if (left->type == MATRIX_TYPE_INVALID || right->type == MATRIX_TYPE_INVALID)
        return EVAL_BAD_ENUM;

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
            return EVAL_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_INCIDENCE && left->incidence.incidence == right->incidence.incidence)
        {
            *out = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = left->incidence.incidence},
                              .coefficient = left->coefficient + right->coefficient};
            return EVAL_SUCCESS;
        }

        if (left->type == MATRIX_TYPE_FULL)
        {
            eval_result_t res = matrix_full_copy(&right->full, &out->full, allocator);
            if (res != EVAL_SUCCESS)
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
        return EVAL_WRONG_MAT_TYPES;
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
            return EVAL_DIMS_MISMATCH;

        res = matrix_full_copy(&full_old, &out->full, allocator);
        if (res != EVAL_SUCCESS)
            return res;
        matrix_add_diagonal_inplace(&out->full, to_convert->coefficient);
    }

    // to_convert->type == MATRIX_TYPE_INCIDENCE
    res = incidence_to_full(to_convert->incidence.incidence, order, &out->full, allocator);
    if (res != EVAL_SUCCESS)
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
eval_result_t matrix_full_invert(const matrix_full_t *this, matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    if (this->base.rows != this->base.cols)
        return EVAL_DIMS_MISMATCH;
    const matrix_full_t out = {.base = this->base, .data = allocate(allocator, sizeof(*out.data) * this->base.rows * this->base.cols)};
    if (!out.data)
        return EVAL_FAILED_ALLOC;
    invert_matrix(this->base.rows, this->data, out.data, out.data);
    *p_out = out;
    return EVAL_SUCCESS;
}

/**
 * Invert the matrix by non-pivoted LU decomposition, where the diagonal of L is assumed to be 1.
 *
 * @param n Dimension of the matrix.
 * @param mat Matrix which to invert. Can be equal to ``out``.
 * @param buffer Buffer used for intermediate calculations. Receives the LU decomposition of the matrix.
 * @param out Where to write the resulting inverse matrix to. Can be equal to ``mat``.
 */
MFV2D_INTERNAL
void invert_matrix(const unsigned n, const double mat[static n * n], double buffer[restrict n * n],
                          double out[n * n])
{
    for (uint32_t i = 0; i < n; ++i)
    {
        //  Deal with a row of L
        for (uint_fast32_t j = 0; j < i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *li = buffer + n * i;
            //  Column of U
            const double *uj = buffer + j;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += li[k] * uj[k * n];
            }
            buffer[n * i + j] = (mat[n * i + j] - v) / uj[n * j];
        }

        //  Deal with a column of U
        for (uint_fast32_t j = 0; j <= i; ++j)
        {
            double v = 0;
            //  Row of L
            const double *lj = buffer + n * j;
            //  Column of U
            const double *ui = buffer + i;
            for (uint_fast32_t k = 0; k < j; ++k)
            {
                v += lj[k] * ui[k * n];
            }
            buffer[n * j + i] = mat[n * j + i] - v;
        }
    }

    // Now use back and forward substitution to compute inverse of the matrix
    // based on the following expression:
    //
    //      L @ U @ (A^{-1}) = I
    //
    // First solve `L @ B = I` for `B`

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < n; ++j)
        {
            double v = j == i;
            for (unsigned k = 0; k < i; ++k)
            {
                v -= buffer[i * n + k] * out[k * n + j];
            }
            out[i * n + j] = v;
        }
    }
    // Now solve `U @ (A^{-1}) = B` for `A^{-1}`

    for (unsigned i = n; i > 0; --i)
    {
        for (unsigned j = 0; j < n; ++j)
        {
            double v = out[(i - 1) * n + j];
            for (unsigned k = i; k < n; ++k)
            {
                v -= buffer[(i - 1) * n + k] * out[k * n + j];
            }
            out[(i - 1) * n + j] = v / buffer[(i - 1) * n + (i - 1)];
        }
    }
}