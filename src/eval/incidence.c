//
// Created by jan on 16.2.2025.
//

#include "incidence.h"

static mfv2d_result_t check_dims(const matrix_full_t *const mat, const unsigned rows, const unsigned cols)
{
    if ((rows != 0 && mat->base.rows != rows) || (cols != 0 && mat->base.cols != cols))
        return MFV2D_DIMS_MISMATCH;
    return MFV2D_SUCCESS;
}

mfv2d_result_t apply_e10_left(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, (order + 1) * (order + 1), 0)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = 2 * order * (order + 1);
    const unsigned n_cols = in->base.cols;

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;

    for (unsigned i_col = 0; i_col < n_cols; ++i_col)
    {
        // Horizontal lines
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = (order + 1) * row + col;
                const unsigned col_e2 = (order + 1) * row + col + 1;
                ptr[row_e * n_cols + i_col] =
                    in->data[col_e1 * in->base.cols + i_col] - in->data[col_e2 * in->base.cols + i_col];
            }
        }

        // Vertical lines
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col;
                const unsigned col_e1 = (order + 1) * (row + 1) + col;
                const unsigned col_e2 = (order + 1) * row + col;
                ptr[row_e * n_cols + i_col] =
                    in->data[col_e1 * in->base.cols + i_col] - in->data[col_e2 * in->base.cols + i_col];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e21_left(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 2 * order * (order + 1), 0)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = order * order;
    const unsigned n_cols = in->base.cols;

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;

    for (unsigned i_col = 0; i_col < n_cols; ++i_col)
    {
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = order * row + col;
                const unsigned col_e2 = order * (row + 1) + col;
                const unsigned col_e3 = order * (order + 1) + (order + 1) * row + col;
                const unsigned col_e4 = order * (order + 1) + (order + 1) * row + col + 1;
                ptr[row_e * n_cols + i_col] =
                    in->data[col_e1 * in->base.cols + i_col] - in->data[col_e2 * in->base.cols + i_col] +
                    in->data[col_e3 * in->base.cols + i_col] - in->data[col_e4 * in->base.cols + i_col];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e10t_left(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 2 * (order + 1) * order, 0)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = (order + 1) * (order + 1);
    const unsigned n_cols = in->base.cols;

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;
    memset(ptr, 0, n_rows * n_cols * sizeof *ptr);

    for (unsigned i_col = 0; i_col < n_cols; ++i_col)
    {
        // Nodes with horizontal lines on the right
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * (order + 1) + col;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] += in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Nodes with horizontal lines on the left
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * (order + 1) + col + 1;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] -= in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Nodes with vertical lines on their top
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = row * (order + 1) + col;
                const unsigned col_e1 = order * (order + 1) + (order + 1) * row + col;
                ptr[row_e * n_cols + i_col] -= in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Nodes with vertical lines on their bottom
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = (row + 1) * (order + 1) + col;
                const unsigned col_e1 = order * (order + 1) + (order + 1) * row + col;
                ptr[row_e * n_cols + i_col] += in->data[col_e1 * in->base.cols + i_col];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e21t_left(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, order * order, 0)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = 2 * order * (order + 1);
    const unsigned n_cols = in->base.cols;

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;
    memset(ptr, 0, n_rows * n_cols * sizeof *ptr);
    for (unsigned i_col = 0; i_col < n_cols; ++i_col)
    {
        // Lines with surfaces above
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] += in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Lines with surfaces bottom
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = (row + 1) * order + col;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] -= in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Lines with surfaces on the left
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] += in->data[col_e1 * in->base.cols + i_col];
            }
        }

        // Lines with surfaces on the right
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col + 1;
                const unsigned col_e1 = order * row + col;
                ptr[row_e * n_cols + i_col] -= in->data[col_e1 * in->base.cols + i_col];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e10_right(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 0, 2 * (order + 1) * order)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = in->base.rows;
    const unsigned n_cols = (order + 1) * (order + 1);

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;
    memset(ptr, 0, n_rows * n_cols * sizeof *ptr);
    memset(ptr, 0, n_rows * n_cols * sizeof *ptr);

    for (unsigned i_row = 0; i_row < n_rows; ++i_row)
    {
        // Nodes with horizontal lines on the right
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * (order + 1) + col;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] += in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Nodes with horizontal lines on the left
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * (order + 1) + col + 1;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] -= in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Nodes with vertical lines on their top
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = row * (order + 1) + col;
                const unsigned col_e1 = order * (order + 1) + (order + 1) * row + col;
                ptr[i_row * n_cols + row_e] -= in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Nodes with vertical lines on their bottom
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = (row + 1) * (order + 1) + col;
                const unsigned col_e1 = order * (order + 1) + (order + 1) * row + col;
                ptr[i_row * n_cols + row_e] += in->data[i_row * in->base.cols + col_e1];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e21_right(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 0, order * order)) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = in->base.rows;
    const unsigned n_cols = 2 * order * (order + 1);

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;
    memset(ptr, 0, n_rows * n_cols * sizeof *ptr);
    for (unsigned i_row = 0; i_row < n_rows; ++i_row)
    {
        // Lines with surfaces above
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] += in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Lines with surfaces bottom
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = (row + 1) * order + col;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] -= in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Lines with surfaces on the left
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] += in->data[i_row * in->base.cols + col_e1];
            }
        }

        // Lines with surfaces on the right
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col + 1;
                const unsigned col_e1 = order * row + col;
                ptr[i_row * n_cols + row_e] -= in->data[i_row * in->base.cols + col_e1];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e10t_right(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                                const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 0, (order + 1) * (order + 1))) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = in->base.rows;
    const unsigned n_cols = 2 * order * (order + 1);

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;

    for (unsigned i_row = 0; i_row < n_rows; ++i_row)
    {
        // Horizontal lines
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = (order + 1) * row + col;
                const unsigned col_e2 = (order + 1) * row + col + 1;
                ptr[i_row * n_cols + row_e] =
                    in->data[i_row * in->base.cols + col_e1] - in->data[i_row * in->base.cols + col_e2];
            }
        }

        // Vertical lines
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                const unsigned row_e = order * (order + 1) + row * (order + 1) + col;
                const unsigned col_e1 = (order + 1) * (row + 1) + col;
                const unsigned col_e2 = (order + 1) * row + col;
                ptr[i_row * n_cols + row_e] =
                    in->data[i_row * in->base.cols + col_e1] - in->data[i_row * in->base.cols + col_e2];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_e21t_right(const unsigned order, const matrix_full_t *in, matrix_full_t *out,
                                const allocator_callbacks *allocator)
{
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = check_dims(in, 0, 2 * order * (order + 1))) != MFV2D_SUCCESS)
        return res;

    const unsigned n_rows = in->base.rows;
    const unsigned n_cols = order * order;

    double *const restrict ptr = allocate(allocator, n_rows * n_cols * sizeof *ptr);
    if (!ptr)
        return MFV2D_FAILED_ALLOC;

    for (unsigned i_row = 0; i_row < n_rows; ++i_row)
    {
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                const unsigned row_e = row * order + col;
                const unsigned col_e1 = order * row + col;
                const unsigned col_e2 = order * (row + 1) + col;
                const unsigned col_e3 = order * (order + 1) + (order + 1) * row + col;
                const unsigned col_e4 = order * (order + 1) + (order + 1) * row + col + 1;
                ptr[i_row * n_cols + row_e] =
                    in->data[i_row * in->base.cols + col_e1] - in->data[i_row * in->base.cols + col_e2] +
                    in->data[i_row * in->base.cols + col_e3] - in->data[i_row * in->base.cols + col_e4];
            }
        }
    }

    *out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n_rows, .cols = n_cols}, .data = ptr};
    return res;
}

mfv2d_result_t apply_incidence_to_full_left(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                            matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        return apply_e10_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_10_T:
        return apply_e10t_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21:
        return apply_e21_left(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21_T:
        return apply_e21t_left(order, in, p_out, allocator);
    default:
        return MFV2D_BAD_ENUM;
    }
}

mfv2d_result_t apply_incidence_to_full_right(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                             matrix_full_t *p_out, const allocator_callbacks *allocator)
{
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        return apply_e10_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_10_T:
        return apply_e10t_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21:
        return apply_e21_right(order, in, p_out, allocator);
    case INCIDENCE_TYPE_21_T:
        return apply_e21t_right(order, in, p_out, allocator);
    default:
        return MFV2D_BAD_ENUM;
    }
}

mfv2d_result_t incidence_to_full(const incidence_type_t type, const unsigned order, matrix_full_t *p_out,
                                 const allocator_callbacks *allocator)
{
    unsigned n_row, n_col;
    switch (type)
    {
    case INCIDENCE_TYPE_10:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_1, order, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_0, order, order);
        break;
    case INCIDENCE_TYPE_10_T:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_0, order, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_1, order, order);
        break;
    case INCIDENCE_TYPE_21:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_2, order, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_1, order, order);
        break;
    case INCIDENCE_TYPE_21_T:
        n_row = form_degrees_of_freedom_count(FORM_ORDER_1, order, order);
        n_col = form_degrees_of_freedom_count(FORM_ORDER_2, order, order);
        break;
    default:
        return 0;
    }

    const matrix_full_t this = {.base = {.type = MATRIX_TYPE_FULL, .rows = n_row, .cols = n_col},
                                .data = allocate(allocator, n_row * n_col * sizeof(*this.data))};
    if (!this.data)
        return 0;
    memset(this.data, 0, sizeof(*this.data) * this.base.rows * this.base.cols);
    switch (type) // Only going to cover types we can encounter.
    {
    case INCIDENCE_TYPE_10:
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[(row * order + col) * this.base.cols + (order + 1) * row + col] = +1.0;
                this.data[(row * order + col) * this.base.cols + (order + 1) * row + col + 1] = -1.0;
            }
        }
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                this.data[((order + 1) * order + row * (order + 1) * col) * this.base.cols + (order + 1) * row + col] =
                    -1.0;
                this.data[((order + 1) * order + row * (order + 1) * col) * this.base.cols + (order + 1) * (row + 1) +
                          col] = +1.0;
            }
        }
        break;
    case INCIDENCE_TYPE_10_T:
        for (unsigned row = 0; row < order + 1; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[((order + 1) * row + col) * this.base.cols + row * order + col] = +1.0;
                this.data[((order + 1) * row + col + 1) * this.base.cols + row * order + col] = -1.0;
            }
        }
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order + 1; ++col)
            {
                this.data[((order + 1) * row + col) * this.base.cols + (order + 1) * order + row * (order + 1) * col] =
                    -1.0;
                this.data[((order + 1) * (row + 1) + col) * this.base.cols + (order + 1) * order +
                          row * (order + 1) * col] = +1.0;
            }
        }
        break;

    case INCIDENCE_TYPE_21:
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[(row * order + col) * this.base.cols + (row * order + col)] = +1.0;
                this.data[(row * order + col) * this.base.cols + ((row + 1) * order + col)] = -1.0;
                this.data[(row * order + col) * this.base.cols + order * (order + 1) + (row * (order + 1) + col)] =
                    +1.0;
                this.data[(row * order + col) * this.base.cols + order * (order + 1) + (row * (order + 1) + col + 1)] =
                    -1.0;
            }
        }
        break;

    case INCIDENCE_TYPE_21_T:
        for (unsigned row = 0; row < order; ++row)
        {
            for (unsigned col = 0; col < order; ++col)
            {
                this.data[((row * order + col)) * this.base.cols + row * order + col] = +1.0;
                this.data[(((row + 1) * order + col)) * this.base.cols + row * order + col] = -1.0;
                this.data[(order * (order + 1) + (row * (order + 1) + col)) * this.base.cols + row * order + col] =
                    +1.0;
                this.data[(order * (order + 1) + (row * (order + 1) + col + 1)) * this.base.cols + row * order + col] =
                    -1.0;
            }
        }
        break;

    default:
        return MFV2D_BAD_ENUM;
    }

    *p_out = this;
    return 1;
}
