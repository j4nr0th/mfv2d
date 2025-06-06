//
// Created by jan on 15.2.2025.
//

#include "evaluation.h"

#include "incidence.h"

static void clean_stack(matrix_t *stack, unsigned cnt, const allocator_callbacks *allocator)
{
    for (unsigned i = cnt; i > 0; --i)
    {
        matrix_cleanup(stack + i - 1, allocator);
    }
}

static mfv2d_result_t matrix_as_full(error_stack_t *error_stack, const matrix_t *this, unsigned order,
                                     form_order_t form, matrix_full_t *p_out, const allocator_callbacks *allocator)
{

    switch (this->type)
    {
    case MATRIX_TYPE_IDENTITY: {
        const unsigned n = form_degrees_of_freedom_count(form, order, order);
        double *const restrict ptr = allocate(allocator, sizeof *ptr * n * n);
        if (!ptr)
        {
            MFV2D_ERROR(error_stack, MFV2D_FAILED_ALLOC, "Could not allocate memory for output identity matrix.");
            return MFV2D_FAILED_ALLOC;
        }
        memset(ptr, 0, sizeof *ptr * n * n);
        for (unsigned i = 0; i < n; ++i)
        {
            ptr[i * n + i] = this->coefficient;
        }
        *p_out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n, .cols = n}, .data = ptr};
    }
    break;

    case MATRIX_TYPE_FULL:
        matrix_multiply_inplace(&this->full, this->coefficient);
        *p_out = this->full;
        break;

    case MATRIX_TYPE_INCIDENCE:
        const mfv2d_result_t res = incidence_to_full(this->incidence.incidence, order, p_out, allocator);
        if (res != MFV2D_SUCCESS)
            return res;
        matrix_multiply_inplace(p_out, this->coefficient);
        break;

    default:
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Invalid matrix type for the output matrix %u.", this->type);
        return MFV2D_BAD_ENUM;
    }

    return MFV2D_SUCCESS;
}

typedef mfv2d_result_t (*const bytecode_operation)(const void *operations[static MATOP_COUNT],
                                                   error_stack_t *error_stack, unsigned order, unsigned remaining,
                                                   const bytecode_t code[static remaining], precompute_t *precomp,
                                                   const field_information_t *vector_fields, unsigned n_stack,
                                                   unsigned stack_pos, matrix_t stack[restrict n_stack],
                                                   const allocator_callbacks *allocator, matrix_t *current,
                                                   const matrix_full_t *initial);

static mfv2d_result_t execute_next(const bytecode_operation operations[static MATOP_COUNT], error_stack_t *error_stack,
                                   unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                   precompute_t *precomp, const field_information_t *vector_fields, unsigned n_stack,
                                   unsigned stack_pos, matrix_t stack[restrict n_stack],
                                   const allocator_callbacks *allocator, matrix_t *current,
                                   const matrix_full_t *initial)
{
    if (remaining == 0)
    {
        return MFV2D_SUCCESS;
    }
    const matrix_op_t op = code->op;
    if (op <= MATOP_INVALID || op >= MATOP_COUNT)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Invalid bytecode instruction %u.", (unsigned)op);
        return MFV2D_BAD_ENUM;
    }
    const bytecode_operation fn = operations[op];

    return fn((void *)operations, error_stack, order, remaining - 1, code + 1, precomp, vector_fields, n_stack,
              stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_identity(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                         unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                         precompute_t *precomp, const field_information_t *vector_fields,
                                         unsigned n_stack, unsigned stack_pos, matrix_t stack[restrict n_stack],
                                         const allocator_callbacks *allocator, matrix_t *current,
                                         const matrix_full_t *initial)
{
    if (current->type == MATRIX_TYPE_INVALID)
    {
        current->type = MATRIX_TYPE_IDENTITY;
        current->coefficient = 1.0;
    }
    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_scale(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                      unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                      precompute_t *precomp, const field_information_t *vector_fields, unsigned n_stack,
                                      unsigned stack_pos, matrix_t stack[restrict n_stack],
                                      const allocator_callbacks *allocator, matrix_t *current,
                                      const matrix_full_t *initial)
{
    if (remaining < 1)
    {
        MFV2D_ERROR(error_stack, MFV2D_OUT_OF_INSTRUCTIONS, "Scale instruction with no instructions remaining.");
        return MFV2D_OUT_OF_INSTRUCTIONS;
    }
    if (current->type == MATRIX_TYPE_INVALID)
    {
        current->type = MATRIX_TYPE_IDENTITY;
        current->coefficient = code->f64;
    }
    else
    {
        current->coefficient *= code->f64;
    }
    code += 1;
    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining - 1, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_push(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                     unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                     precompute_t *precomp, const field_information_t *vector_fields, unsigned n_stack,
                                     unsigned stack_pos, matrix_t stack[restrict n_stack],
                                     const allocator_callbacks *allocator, matrix_t *current,
                                     const matrix_full_t *initial)
{
    if (stack_pos == n_stack)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_OVERFLOW, "Stack is full.");
        return MFV2D_STACK_OVERFLOW;
    }
    stack[stack_pos] = *current;
    stack_pos += 1;
    *current = (matrix_t){
        .type = MATRIX_TYPE_INVALID,
        .coefficient = 0.0,
    };
    if (initial)
    {
        current->coefficient = 1.0;
        const mfv2d_result_t res = matrix_full_copy(initial, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not create copy of the initial state.");
            return res;
        }
    }
    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_incidence(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                          unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                          precompute_t *precomp, const field_information_t *vector_fields,
                                          unsigned n_stack, unsigned stack_pos, matrix_t stack[restrict n_stack],
                                          const allocator_callbacks *allocator, matrix_t *current,
                                          const matrix_full_t *initial)
{
    if (remaining < 2)
    {
        MFV2D_ERROR(error_stack, MFV2D_OUT_OF_INSTRUCTIONS,
                    "Incidence matrix instruction with less than 2 instructions remaining.");
        return MFV2D_OUT_OF_INSTRUCTIONS;
    }

    incidence_type_t t = code->u32;
    code += 1;
    const unsigned dual = code->u32;
    code += 1;
    if (t < INCIDENCE_TYPE_10 || t >= INCIDENCE_TYPE_CNT)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Incidence type specified by current matrix is %u which is not valid.",
                    t);
        return MFV2D_BAD_ENUM;
    }
    if (dual)
    {
        t = 3 - t; // ((t - 1) ^ 1) + 1;
    }

    switch (current->type)
    {
    case MATRIX_TYPE_INVALID:
        *current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t}, .coefficient = 1.0};
        break;
    case MATRIX_TYPE_IDENTITY:
        *current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t},
                              .coefficient = current->coefficient};
        break;

        matrix_t new_mat;
        mfv2d_result_t res;

    case MATRIX_TYPE_INCIDENCE:
        res = incidence_to_full(current->type, order, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not create the incidence matrix.");
            return res;
        }
        // FALLTHROUGH

    case MATRIX_TYPE_FULL:
        res = apply_incidence_to_full_left(t, order, &current->full, &new_mat.full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res,
                        "Could apply an incidence matrix (type %u) to a full matrix of dimensions  (%u, %u).",
                        (unsigned)t, current->full.base.rows, current->full.base.cols);
            return res;
        }
        new_mat.coefficient = current->coefficient;
        matrix_cleanup(current, allocator);
        *current = new_mat;
        break;
    default:
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Current matrix has an unknown type %u.", current->type);
        return MFV2D_BAD_ENUM;
    }
    if (dual)
    {
        current->coefficient *= -1;
    }

    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining - 2, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_mass(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                     unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                     precompute_t *precomp, const field_information_t *vector_fields, unsigned n_stack,
                                     unsigned stack_pos, matrix_t stack[restrict n_stack],
                                     const allocator_callbacks *allocator, matrix_t *current,
                                     const matrix_full_t *initial)
{
    if (remaining < 2)
    {
        MFV2D_ERROR(error_stack, MFV2D_OUT_OF_INSTRUCTIONS,
                    "Mass matrix instruction with less than 2 instructions remaining.");
        return MFV2D_OUT_OF_INSTRUCTIONS;
    }

    mass_mtx_indices_t t = code->u32;
    code += 1;
    const unsigned inverse = code->u32;
    code += 1;
    if (t < MASS_0 || t > MASS_2)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Mass type specified by current matrix is %u which is not valid.", t);
        return MFV2D_BAD_ENUM;
    }
    if (inverse)
    {
        t += (MASS_0_I - MASS_0);
    }

    matrix_t this = {.type = MATRIX_TYPE_FULL, .coefficient = 1.0};
    {
        const matrix_full_t *p_mass = precompute_get_matrix(precomp, t, allocator);
        if (!p_mass)
        {
            MFV2D_ERROR(error_stack, MFV2D_FAILED_ALLOC, "Failed allocating and computing a mass matrix.");
            return MFV2D_FAILED_ALLOC;
        }
        this.full = *p_mass;
    }
    mfv2d_result_t res;
    switch (current->type)
    {
    case MATRIX_TYPE_INVALID:
        current->coefficient = 1.0;
        // FALLTHROUGH
    case MATRIX_TYPE_IDENTITY:
        res = matrix_full_copy(&this.full, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not copy the mass matrix.");
            return res;
        }
        break;
    case MATRIX_TYPE_INCIDENCE:

        res = apply_incidence_to_full_right(current->incidence.incidence, order, &this.full, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could right-apply incidence matrix to the mass matrix.");
            return res;
        }
        break;
    case MATRIX_TYPE_FULL:
        matrix_t new_mat = {};
        res = matrix_full_multiply(&this.full, &current->full, &new_mat.full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could multiply two full matrices.");
            return res;
        }
        matrix_cleanup(current, allocator);
        *current = new_mat;
        current->coefficient = 1.0;
        break;
    }

    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining - 2, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_matmul(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                       unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                       precompute_t *precomp, const field_information_t *vector_fields,
                                       unsigned n_stack, unsigned stack_pos, matrix_t stack[restrict n_stack],
                                       const allocator_callbacks *allocator, matrix_t *current,
                                       const matrix_full_t *initial)
{
    if (stack_pos == 0)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_UNDERFLOW, "Matrix multiply operation with nothing on the stack.");
        return MFV2D_STACK_UNDERFLOW;
    }
    stack_pos -= 1;

    matrix_t right = stack[stack_pos];
    stack[stack_pos] = (matrix_t){.type = MATRIX_TYPE_INVALID};
    matrix_t new_mat;
    const mfv2d_result_t res = matrix_multiply(error_stack, order, &right, current, &new_mat, allocator);
    matrix_cleanup(stack + stack_pos, allocator);
    matrix_cleanup(current, allocator);
    if (res != MFV2D_SUCCESS)
    {
        MFV2D_ERROR(error_stack, res, "Failed multiplying two matrices (%u x %u and %u).", right.base.rows,
                    right.base.cols, current->type);
        return res;
    }
    *current = new_mat;

    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

static mfv2d_result_t operation_sum(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                    unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                    precompute_t *precomp, const field_information_t *vector_fields, unsigned n_stack,
                                    unsigned stack_pos, matrix_t stack[restrict n_stack],
                                    const allocator_callbacks *allocator, matrix_t *current,
                                    const matrix_full_t *initial)
{
    if (remaining < 1)
    {
        MFV2D_ERROR(error_stack, MFV2D_OUT_OF_INSTRUCTIONS, "Sum instruction with no more bytecode units.");
        return MFV2D_OUT_OF_INSTRUCTIONS;
    }

    const unsigned count = code->u32;
    code += 1;
    if (count > stack_pos)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_UNDERFLOW, "Sum for %u matrices specified, but only %u are on stack.",
                    count, stack_pos);
        return MFV2D_STACK_UNDERFLOW;
    }
    for (unsigned j = 0; j < count; ++j)
    {
        matrix_t new;
        stack_pos -= 1;
        matrix_t *left = stack + stack_pos;
        const mfv2d_result_t res = matrix_add(order, current, left, &new, allocator);
        matrix_cleanup(current, allocator);
        matrix_cleanup(left, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not add two matrices.");
            return res;
        }
        *current = new;
    }

    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining - 1, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

typedef enum
{
    MIXED_MATRIX_01 = 1,
    MIXED_MATRIX_11 = 2,
    MIXED_MATRIX_12 = 3,
} mixed_matrix_t;

static mfv2d_result_t compute_mixed_matrix(matrix_t *p_out, const allocator_callbacks *allocator,
                                           const precompute_t *precomp, const double *field, int b_reorder,
                                           mixed_matrix_t type, double coeff)
{
    const unsigned n_int = precomp->basis->n_int, n_basis = precomp->basis->order;
    const jacobian_t *const jac = precomp->jacobian;
    unsigned rows, cols;
    const double *ptr = NULL;
    switch (type)
    {
    case MIXED_MATRIX_01:
        rows = (n_basis + 1) * (n_basis + 1);
        cols = 2 * (n_basis + 1) * n_basis;
        ptr = precomp->basis->mix_10;
        break;
    case MIXED_MATRIX_11:
        rows = 2 * (n_basis + 1) * n_basis;
        cols = 2 * (n_basis + 1) * n_basis;
        break;
    case MIXED_MATRIX_12:
        rows = 2 * (n_basis + 1) * n_basis;
        cols = n_basis * n_basis;
        ptr = precomp->basis->mix_21;
        break;
    }

    unsigned o_r, o_c;
    if (b_reorder)
    {
        o_r = cols;
        o_c = rows;
    }
    else
    {
        o_r = rows;
        o_c = cols;
    }

    const matrix_full_t mat = {.base = {.type = MATRIX_TYPE_FULL, .rows = o_r, .cols = o_c},
                               .data = allocate(allocator, sizeof *mat.data * rows * cols)};
    if (!mat.data)
        return MFV2D_FAILED_ALLOC;

    switch (type)
    {
    case MIXED_MATRIX_01:
        // Mix 10
        //  Left half, which is involved with eta basis
        for (unsigned row = 0; row < (n_basis + 1) * (n_basis + 1); ++row)
        {
            for (unsigned col = 0; col < n_basis * (n_basis + 1); ++col)
            {
                const unsigned offset = (row * cols + col) * n_int * n_int;
                double val = 0.0;
                for (unsigned i = 0; i < n_int; ++i)
                {
                    for (unsigned j = 0; j < n_int; ++j)
                    {
                        const jacobian_t *const p_jac = jac + (i * n_int + j);
                        const double vector_comp = field[i * (2 * n_int) + 2 * j + 0] * p_jac->j11 -
                                                   field[i * (2 * n_int) + 2 * j + 1] * p_jac->j01;
                        val += ptr[offset + i * n_int + j] * vector_comp;
                    }
                }
                if (b_reorder)
                {
                    mat.data[col * rows + row] = val;
                }
                else
                {

                    mat.data[row * cols + col] = val;
                }
            }
        }
        //  Right half, which is involved with xi basis
        for (unsigned row = 0; row < (n_basis + 1) * (n_basis + 1); ++row)
        {
            for (unsigned col = n_basis * (n_basis + 1); col < 2 * n_basis * (n_basis + 1); ++col)
            {
                const unsigned offset = (row * cols + col) * n_int * n_int;
                double val = 0.0;
                for (unsigned i = 0; i < n_int; ++i)
                {
                    for (unsigned j = 0; j < n_int; ++j)
                    {
                        const jacobian_t *const p_jac = jac + (i * n_int + j);
                        const double vector_comp = -(field[i * (2 * n_int) + 2 * j + 1] * p_jac->j00 -
                                                     field[i * (2 * n_int) + 2 * j + 0] * p_jac->j10);
                        val += ptr[offset + i * n_int + j] * vector_comp;
                    }
                }
                if (b_reorder)
                {
                    mat.data[col * rows + row] = val;
                }
                else
                {

                    mat.data[row * cols + col] = val;
                }
            }
        }
        break;
    case MIXED_MATRIX_12:
        // Mix 21
        //  Top half, which is involved with eta basis
        for (unsigned row = 0; row < rows / 2; ++row)
        {
            for (unsigned col = 0; col < cols; ++col)
            {
                const unsigned offset = (row * cols + col) * n_int * n_int;
                double val = 0.0;
                for (unsigned i = 0; i < n_int; ++i)
                {
                    for (unsigned j = 0; j < n_int; ++j)
                    {
                        const jacobian_t *const p_jac = jac + (i * n_int + j);
                        const double vector_comp = -(field[i * (2 * n_int) + 2 * j + 0] * p_jac->j01 +
                                                     field[i * (2 * n_int) + 2 * j + 1] * p_jac->j11) /
                                                   p_jac->det;

                        val += ptr[offset + i * n_int + j] * vector_comp;
                    }
                }
                if (b_reorder)
                {
                    mat.data[col * rows + row] = val;
                }
                else
                {

                    mat.data[row * cols + col] = val;
                }
            }
        }
        //  Bottom half, which is involved with xi basis
        for (unsigned row = rows / 2; row < rows; ++row)
        {
            for (unsigned col = 0; col < cols; ++col)
            {
                const unsigned offset = (row * cols + col) * n_int * n_int;
                double val = 0.0;
                for (unsigned i = 0; i < n_int; ++i)
                {
                    for (unsigned j = 0; j < n_int; ++j)
                    {
                        const jacobian_t *const p_jac = jac + (i * n_int + j);
                        const double vector_comp = -(field[i * (2 * n_int) + 2 * j + 0] * p_jac->j00 +
                                                     field[i * (2 * n_int) + 2 * j + 1] * p_jac->j10) /
                                                   p_jac->det;
                        val += ptr[offset + i * n_int + j] * vector_comp;
                    }
                }
                if (b_reorder)
                {
                    mat.data[col * rows + row] = val;
                }
                else
                {

                    mat.data[row * cols + col] = val;
                }
            }
        }
        break;
    case MIXED_MATRIX_11:
        // Compute edge mass matrix (00) part
        const unsigned n_edge = n_basis * (n_basis + 1);
        if (!b_reorder)
        {
            for (unsigned row = 0; row < n_edge; ++row)
            {
                for (unsigned col = 0; col <= row; ++col)
                {
                    const unsigned offset = (row * n_edge + col) * n_int * n_int;
                    double val = 0.0;
                    for (unsigned i = 0; i < n_int; ++i)
                    {
                        for (unsigned j = 0; j < n_int; ++j)
                        {
                            const double jac_term =
                                field[i * (2 * n_int) + 2 * j + 0] * (jac[i * n_int + j].j11 * jac[i * n_int + j].j11 +
                                                                      jac[i * n_int + j].j01 * jac[i * n_int + j].j01);
                            val += precomp->basis->mass_edge_00[offset + i * n_int + j] * jac_term /
                                   jac[i * n_int + j].det;
                        }
                    }
                    // Use symmetry
                    {
                        mat.data[row * cols + col] = val;
                        mat.data[col * cols + row] = val;
                    }
                }
            }

            // Compute edge mass matrix (01) part
            for (unsigned row = 0; row < n_edge; ++row)
            {
                for (unsigned col = 0; col < n_edge; ++col)
                {
                    const unsigned offset = (row * n_edge + col) * n_int * n_int;
                    double val = 0.0;
                    for (unsigned i = 0; i < n_int; ++i)
                    {
                        for (unsigned j = 0; j < n_int; ++j)
                        {
                            const double jac_term =
                                field[i * (2 * n_int) + 2 * j + 0] * (jac[i * n_int + j].j10 * jac[i * n_int + j].j11 +
                                                                      jac[i * n_int + j].j00 * jac[i * n_int + j].j01);
                            val += precomp->basis->mass_edge_01[offset + i * n_int + j] * jac_term /
                                   jac[i * n_int + j].det;
                        }
                    }
                    // Use symmetry
                    {
                        mat.data[(n_edge + row) * cols + col] = val;
                        mat.data[col * cols + (n_edge + row)] = val;
                    }
                }
            }

            // Compute edge mass matrix (11) part
            for (unsigned row = 0; row < n_edge; ++row)
            {
                for (unsigned col = 0; col <= row; ++col)
                {
                    const unsigned offset = (row * n_edge + col) * n_int * n_int;
                    double val = 0.0;
                    for (unsigned i = 0; i < n_int; ++i)
                    {
                        for (unsigned j = 0; j < n_int; ++j)
                        {
                            const double jac_term =
                                field[i * (2 * n_int) + 2 * j + 0] * (jac[i * n_int + j].j10 * jac[i * n_int + j].j10 +
                                                                      jac[i * n_int + j].j00 * jac[i * n_int + j].j00);
                            val += precomp->basis->mass_edge_11[offset + i * n_int + j] * jac_term /
                                   jac[i * n_int + j].det;
                        }
                    }
                    {
                        mat.data[(n_edge + row) * cols + (n_edge + col)] = val;
                        mat.data[(n_edge + col) * cols + (n_edge + row)] = val;
                    }
                }
            }
        }
        else
        {
            for (unsigned row = 0; row < n_edge; ++row)
            {
                for (unsigned col = 0; col < n_edge; ++col)
                {
                    // Zeroing the (00) block
                    mat.data[row * cols + col] = 0;
                    // Zeroing the (11) block
                    mat.data[(row + n_edge) * cols + (col + n_edge)] = 0;
                }
            }
            // Compute edge mass matrix (01) part
            for (unsigned row = 0; row < n_edge; ++row)
            {
                for (unsigned col = 0; col < n_edge; ++col)
                {
                    const unsigned offset = (row * n_edge + col) * n_int * n_int;
                    double val = 0.0;
                    for (unsigned i = 0; i < n_int; ++i)
                    {
                        for (unsigned j = 0; j < n_int; ++j)
                        {
                            val += precomp->basis->mass_edge_01[offset + i * n_int + j] *
                                   field[i * (2 * n_int) + 2 * j + 0];
                        }
                    }
                    // Use anti-symmetry
                    {
                        mat.data[(n_edge + row) * cols + col] = -val;
                        mat.data[col * cols + (n_edge + row)] = val;
                    }
                }
            }
        }
        break;
    }

    *p_out = (matrix_t){.coefficient = coeff, .full = mat};

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_interprod(const void *operations[static MATOP_COUNT], error_stack_t *error_stack,
                                          unsigned order, unsigned remaining, const bytecode_t code[static remaining],
                                          precompute_t *precomp, const field_information_t *vector_fields,
                                          unsigned n_stack, unsigned stack_pos, matrix_t stack[restrict n_stack],
                                          const allocator_callbacks *allocator, matrix_t *current,
                                          const matrix_full_t *initial)
{
    if (remaining < 4)
    {
        MFV2D_ERROR(error_stack, MFV2D_OUT_OF_INSTRUCTIONS,
                    "InterProd instruction with less than 3 instructions remaining.");
        return MFV2D_OUT_OF_INSTRUCTIONS;
    }

    const unsigned starting_index = code->u32;
    code += 1;
    const unsigned field_index = code->u32;
    code += 1;
    const unsigned dual = code->u32;
    code += 1;
    const unsigned adjoint = code->u32;
    code += 1;
    if (starting_index != 1 && starting_index != 2)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM,
                    "InterProd specified with starting index that is not 1 or 2, but is %u", starting_index);
        return MFV2D_BAD_ENUM;
    }
    if (field_index >= vector_fields->n_fields)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM,
                    "InterProd specified with vector field with index %u, but only %u fields exits.", field_index,
                    vector_fields->n_fields);
        return MFV2D_BAD_ENUM;
    }
    const double *const restrict field = vector_fields->fields[field_index];

    matrix_t this = {.type = MATRIX_TYPE_FULL, .coefficient = 1.0};
    int reorder;
    mixed_matrix_t type;
    double coeff;

    if (!adjoint)
    {
        if (!dual)
        {
            if (starting_index == 1)
            {
                type = MIXED_MATRIX_01;
                reorder = 0;
                coeff = 1.0;
            }
            else if (starting_index == 2)
            {
                type = MIXED_MATRIX_12;
                reorder = 0;
                coeff = 1.0;
            }
        }
        else
        {
            if (starting_index == 1)
            {
                type = MIXED_MATRIX_12;
                reorder = 1;
                coeff = -1.0;
            }
            else if (starting_index == 2)
            {
                type = MIXED_MATRIX_01;
                reorder = 1;
                coeff = 1.0;
            }
        }
    }
    else
    {
        if (!dual)
        {
            if (starting_index == 1)
            {
                type = MIXED_MATRIX_01;
                reorder = 0;
                coeff = -1.0;
            }
            else if (starting_index == 2)
            {
                type = MIXED_MATRIX_11;
                reorder = 0;
                coeff = -1.0;
            }
        }
        else
        {
            if (starting_index == 1)
            {
                type = MIXED_MATRIX_12;
                reorder = 1;
                coeff = -1.0;
            }
            else if (starting_index == 2)
            {
                type = MIXED_MATRIX_11;
                reorder = 1;
                coeff = 1.0;
            }
        }
    }

    mfv2d_result_t res = compute_mixed_matrix(&this, allocator, precomp, field, reorder, type, coeff);
    this.coefficient = coeff;
    switch (current->type)
    {
    case MATRIX_TYPE_INVALID:
        current->coefficient = 1.0;
        // FALLTHROUGH
    case MATRIX_TYPE_IDENTITY:
        res = matrix_full_copy(&this.full, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not copy the mass matrix.");
            return res;
        }
        break;
    case MATRIX_TYPE_INCIDENCE:

        res = apply_incidence_to_full_right(current->incidence.incidence, order, &this.full, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could right-apply incidence matrix to the mass matrix.");
            return res;
        }
        break;
    case MATRIX_TYPE_FULL:
        matrix_t new_mat = {};
        res = matrix_full_multiply(&this.full, &current->full, &new_mat.full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could multiply two full matrices.");
            return res;
        }
        matrix_cleanup(current, allocator);
        *current = new_mat;
        current->coefficient = 1.0;
        break;
    }
    matrix_cleanup(&this, allocator);
    current->coefficient *= coeff;

    return execute_next((const bytecode_operation *)operations, error_stack, order, remaining - 4, code, precomp,
                        vector_fields, n_stack, stack_pos, stack, allocator, current, initial);
}

MFV2D_INTERNAL
mfv2d_result_t evaluate_element_term_sibling(error_stack_t *error_stack, form_order_t form, unsigned order,
                                             const bytecode_t *code, precompute_t *precomp,
                                             const field_information_t *vector_fields, unsigned n_stack,
                                             matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                             matrix_full_t *p_out, const matrix_full_t *initial)
{
    const unsigned n_ops = code[0].u32;
    code += 1;

    unsigned stack_pos = 0;
    matrix_t current = {
        .type = MATRIX_TYPE_INVALID,
        .coefficient = 0.0,
    };
    if (initial)
    {
        current.coefficient = 1.0;
        const mfv2d_result_t res = matrix_full_copy(initial, &current.full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not initialize matrix.");
            return res;
        }
    }

    const bytecode_operation operations[MATOP_COUNT] = {
        [MATOP_INVALID] = NULL,
        [MATOP_IDENTITY] = operation_identity,
        [MATOP_MASS] = operation_mass,
        [MATOP_INCIDENCE] = operation_incidence,
        [MATOP_PUSH] = operation_push,
        [MATOP_MATMUL] = operation_matmul,
        [MATOP_SCALE] = operation_scale,
        [MATOP_SUM] = operation_sum,
        [MATOP_INTERPROD] = operation_interprod,
    };

    const mfv2d_result_t res = execute_next(operations, error_stack, order, n_ops, code, precomp, vector_fields,
                                            n_stack, 0, stack, allocator, &current, initial);
    clean_stack(stack, n_stack, allocator);
    if (res != MFV2D_SUCCESS)
    {
        matrix_cleanup(&current, allocator);
        return res;
    }

    return matrix_as_full(error_stack, &current, order, form, p_out, allocator);
}
