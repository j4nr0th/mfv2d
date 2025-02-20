//
// Created by jan on 15.2.2025.
//

#include "evaluation.h"

#include "incidence.h"

INTERPLIB_INTERNAL
int system_template_create(system_template_t *this, PyObject *orders, PyObject *expr_matrix,
                           const allocator_callbacks *allocator)
{
    // Find number of forms
    {
        PyArrayObject *const order_array = (PyArrayObject *)PyArray_FromAny(
            orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
        if (!order_array)
            return 0;

        const unsigned n_forms = PyArray_DIM(order_array, 0);
        this->n_forms = n_forms;
        this->form_orders = allocate(allocator, sizeof(*this->form_orders) * n_forms);
        const unsigned *restrict p_o = PyArray_DATA(order_array);
        for (unsigned i = 0; i < n_forms; ++i)
        {
            const unsigned o = p_o[i];
            if (o > 2)
            {
                PyErr_Format(PyExc_ValueError, "Form can not be of order higher than 2 (it was %u)", o);
                Py_DECREF(order_array);
                return 0;
            }
            this->form_orders[i] = o + 1;
        }
        Py_DECREF(order_array);
    }

    // Now go though the rows
    ssize_t row_count = PySequence_Size(expr_matrix);
    if (row_count < 0)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }

    if (row_count != this->n_forms)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of forms deduced from order array (%u) does not match the number of expression rows (%u).",
                     this->n_forms, row_count);
        deallocate(allocator, this->form_orders);
        return 0;
    }

    this->bytecodes = allocate(allocator, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    if (!this->bytecodes)
    {
        deallocate(allocator, this->form_orders);
        return 0;
    }
    memset(this->bytecodes, 0, sizeof(*this->bytecodes) * this->n_forms * this->n_forms);
    unsigned max_stack = 1;
    for (unsigned row = 0; row < this->n_forms; ++row)
    {
        PyObject *row_expr = PySequence_GetItem(expr_matrix, row);
        if (!row_expr)
        {
            goto failed_row;
        }
        row_count = PySequence_Size(row_expr);
        if (row_count < 0)
        {
            goto failed_row;
        }
        if (row_count != this->n_forms)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Number of forms deduced from order array (%u) does not match the number of expression in row %u (%u).",
                this->n_forms, row, row_count);
            goto failed_row;
        }

        for (unsigned col = 0; col < this->n_forms; ++col)
        {
            PyObject *expr = PySequence_GetItem(row_expr, col);
            if (!expr)
            {
                goto failed_row;
            }
            if (Py_IsNone(expr))
            {
                Py_DECREF(expr);
                continue;
            }
            PyObject *seq = PySequence_Fast(expr, "Bytecode must be a given as a sequence.");
            if (!seq)
            {
                Py_DECREF(expr);
                goto failed_row;
            }
            row_count = PySequence_Fast_GET_SIZE(seq);
            if (row_count < 0)
            {
                Py_DECREF(expr);
                goto failed_row;
            }

            bytecode_t *bc = allocate(allocator, sizeof(**this->bytecodes) * (row_count + 1));
            if (!bc)
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            this->bytecodes[row * this->n_forms + col] = bc;
            unsigned stack;
            if (!convert_bytecode(row_count, bc, PySequence_Fast_ITEMS(seq), &stack))
            {
                Py_DECREF(seq);
                Py_DECREF(expr);
                goto failed_row;
            }
            if (stack > max_stack)
            {
                max_stack = stack;
            }
            Py_DECREF(seq);
            Py_DECREF(expr);
        }

        continue;

    failed_row: {
        Py_XDECREF(row_expr);
        for (unsigned i = row; i > 0; --i)
        {
            for (unsigned j = this->n_forms; j > 0; --j)
            {
                deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
            }
        }
        deallocate(allocator, this->bytecodes);
        deallocate(allocator, this->form_orders);
        Py_DECREF(row_expr);
        return 0;
    }
    }

    this->max_stack = max_stack;

    return 1;
}

void system_template_destroy(system_template_t *this, const allocator_callbacks *allocator)
{
    deallocate(allocator, this->form_orders);
    for (unsigned i = this->n_forms; i > 0; --i)
    {
        for (unsigned j = this->n_forms; j > 0; --j)
        {
            deallocate(allocator, this->bytecodes[(i - 1) * this->n_forms + (j - 1)]);
        }
    }
    *this = (system_template_t){};
}

static void clean_stack(matrix_t *stack, unsigned cnt, const allocator_callbacks *allocator)
{
    for (unsigned i = cnt; i > 0; ++i)
    {
        matrix_cleanup(stack + i - 1, allocator);
    }
}

INTERPLIB_INTERNAL
int evaluate_element_term(error_stack_t *error_stack, form_order_t form, unsigned order, const bytecode_t *code,
                          precompute_t *precomp, unsigned n_stack, matrix_t stack[restrict n_stack],
                          const allocator_callbacks *allocator, matrix_full_t *p_out)
{
    const unsigned n_ops = code[0].u32;
    code += 1;

    unsigned stack_pos = 0;
    matrix_t current = {
        .type = MATRIX_TYPE_INVALID,
        .coefficient = 0.0,
    };

    for (unsigned i = 0; i < n_ops; ++i)
    {
        const matrix_op_t op = code[i].op;
        unsigned remaining = n_ops - i - 1;
        // printf("Current operation %s, stack at %u. Matrix type %u (%u x %u).", matrix_op_str(op), stack_pos,
        // current.type, current.base.rows, current.base.cols);
        switch (op)
        {
        case MATOP_IDENTITY:
            if (current.type == MATRIX_TYPE_INVALID)
            {
                current.type = MATRIX_TYPE_IDENTITY;
                current.coefficient = 1.0;
            }
            break;

        case MATOP_SCALE:
            if (remaining < 1)
            {
                EVAL_ERROR(error_stack, EVAL_OUT_OF_INSTRUCTIONS, "Scale instruction with no instructions remaining.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            i += 1;
            if (current.type == MATRIX_TYPE_INVALID)
            {
                current.type = MATRIX_TYPE_IDENTITY;
                current.coefficient = code[i].f64;
            }
            else
            {
                current.coefficient *= code[i].f64;
            }
            break;

        case MATOP_TRANSPOSE: {
            const unsigned tmp = current.base.rows;
            current.base.rows = current.base.cols;
            current.base.cols = tmp;
        }
            switch (current.type)
            {
            case MATRIX_TYPE_INCIDENCE: {
                incidence_type_t t = current.incidence.incidence;
                if (t < INCIDENCE_TYPE_10 || t >= INCIDENCE_TYPE_CNT)
                {
                    EVAL_ERROR(error_stack, EVAL_BAD_ENUM,
                               "Incidence type specified by current matrix is %u which is not valid.", t);
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }

                t += (INCIDENCE_TYPE_10_T - INCIDENCE_TYPE_10);
                current.incidence.incidence = t;
            }
            break;

            case MATRIX_TYPE_FULL: {
                matrix_full_t *const this = &current.full;
                const unsigned n_row = this->base.rows;
                const unsigned n_col = this->base.cols;
                for (unsigned idx = 0; idx < n_row; ++idx)
                {
                    for (unsigned j = 0; j * (n_row - 1) < idx * (n_col - 1); ++j)
                    {
                        const double tmp = this->data[idx * n_col + j];
                        this->data[idx * n_col + j] = this->data[j * n_row + idx];
                        this->data[j * n_row + idx] = tmp;
                    }
                }
                this->base.rows = n_col;
                this->base.cols = n_row;
            }
            break;

            case MATRIX_TYPE_IDENTITY:
                /*NO-OP*/
                break;

            default:
                EVAL_ERROR(error_stack, EVAL_BAD_ENUM, "Current matrix type is %u which is not valid.", current.type);
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_BAD_ENUM;
            }
            break;

        case MATOP_PUSH:
            if (stack_pos == n_stack)
            {
                EVAL_ERROR(error_stack, EVAL_STACK_OVERFLOW, "Stack is full.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_STACK_OVERFLOW;
            }
            stack[stack_pos] = current;
            stack_pos += 1;
            current = (matrix_t){
                .type = MATRIX_TYPE_INVALID,
                .coefficient = 0.0,
            };
            break;

        case MATOP_INCIDENCE:
            if (remaining < 2)
            {
                EVAL_ERROR(error_stack, EVAL_OUT_OF_INSTRUCTIONS,
                           "Incidence matrix instruction with less than 2 instructions remaining.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                incidence_type_t t = code[i].u32;
                i += 1;
                const unsigned dual = code[i].u32;
                if (t < INCIDENCE_TYPE_10 || t >= INCIDENCE_TYPE_CNT)
                {
                    EVAL_ERROR(error_stack, EVAL_BAD_ENUM,
                               "Incidence type specified by current matrix is %u which is not valid.", t);
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }
                if (dual)
                {
                    t = ((t - 1) ^ 1) + 1;
                }

                switch (current.type)
                {
                case MATRIX_TYPE_INVALID:
                    current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t},
                                         .coefficient = 1.0};
                    break;
                case MATRIX_TYPE_IDENTITY:
                    current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t},
                                         .coefficient = current.coefficient};
                    break;

                    matrix_t new_mat;
                    eval_result_t res;

                case MATRIX_TYPE_INCIDENCE:
                    res = incidence_to_full(current.type, order, &current.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could not create the incidence matrix.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    // FALLTHROUGH

                case MATRIX_TYPE_FULL:
                    res = apply_incidence_to_full_left(t, order, &current.full, &new_mat.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could apply an incidence matrix to a full matrix.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    new_mat.coefficient = current.coefficient;
                    matrix_cleanup(&current, allocator);
                    current = new_mat;
                    break;
                default:
                    EVAL_ERROR(error_stack, EVAL_BAD_ENUM, "Current matrix has an unknown type %u.", current.type);
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }
                if (dual)
                {
                    current.coefficient *= -1;
                }
                // matrix_t this = { .incidence={.base = {.type = MATRIX_TYPE_INCIDENCE}, .incidence = t}, .coefficient
                // = 1.0}; matrix_t new_mat; const eval_result_t res = matrix_multiply(order, &current, &this, &new_mat,
                // allocator); matrix_cleanup(&current, allocator); if (res != EVAL_SUCCESS)
                // {
                //     return res;
                // }
                // current = new_mat;
            }
            break;

        case MATOP_MASS:
            if (remaining < 2)
            {
                EVAL_ERROR(error_stack, EVAL_OUT_OF_INSTRUCTIONS,
                           "Mass matrix instruction with less than 2 instructions remaining.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                mass_mtx_indices_t t = code[i].u32;
                i += 1;
                const unsigned inverse = code[i].u32;
                if (t < MASS_0 || t > MASS_2)
                {
                    EVAL_ERROR(error_stack, EVAL_BAD_ENUM,
                               "Mass type specified by current matrix is %u which is not valid.", t);
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_BAD_ENUM;
                }
                if (inverse)
                {
                    t += (MASS_0_I - MASS_0);
                }
                matrix_t this = {.type = MATRIX_TYPE_FULL, .coefficient = 1.0};
                this.full = precomp->mass_matrices[t];
                // printf("Getting the matrix with id %s.\n", mass_mtx_indices_str(t));
                // matrix_print(&this);
                eval_result_t res;
                switch (current.type)
                {
                case MATRIX_TYPE_INVALID:
                    current.coefficient = 1.0;
                    // FALLTHROUGH
                case MATRIX_TYPE_IDENTITY:
                    res = matrix_full_copy(&this.full, &current.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could not copy the mass matrix.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    break;
                case MATRIX_TYPE_INCIDENCE:
                    res = apply_incidence_to_full_right(current.incidence.incidence, order, &this.full, &current.full,
                                                        allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could right-apply incidence matrix to the mass matrix.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    break;
                    matrix_t new_mat;
                case MATRIX_TYPE_FULL:
                    res = matrix_full_multiply(&this.full, &current.full, &new_mat.full, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could multiply two full matrices.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    matrix_cleanup(&current, allocator);
                    current = new_mat;
                    current.coefficient = 1.0;
                    break;
                }
                // printf("After operation the matrix is this:");
                // matrix_print(&current);
                // const eval_result_t res = matrix_multiply(order, &current, &this, &new_mat, allocator);
                // matrix_cleanup(&current, allocator);
                // if (res != EVAL_SUCCESS)
                // {
                //     return res;
                // }
                // current = new_mat;
            }
            break;

        case MATOP_MATMUL:
            if (stack_pos == 0)
            {
                EVAL_ERROR(error_stack, EVAL_STACK_UNDERFLOW, "Matrix multiply operation with nothing on the stack.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_STACK_UNDERFLOW;
            }
            stack_pos -= 1;
            {
                matrix_t right = stack[stack_pos];
                matrix_t new_mat;
                const eval_result_t res = matrix_multiply(error_stack, order, &right, &current, &new_mat, allocator);
                matrix_cleanup(stack + stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                if (res != EVAL_SUCCESS)
                {
                    EVAL_ERROR(error_stack, res, "Failed multiplying two matrices (%u x %u and %u).", right.base.rows,
                               right.base.cols, current.type);
                    clean_stack(stack, stack_pos, allocator);
                    return res;
                }
                current = new_mat;
            }
            break;

        case MATOP_SUM:
            if (remaining < 1)
            {
                EVAL_ERROR(error_stack, EVAL_OUT_OF_INSTRUCTIONS, "Sum instruction with no more bytecode units.");
                clean_stack(stack, stack_pos, allocator);
                matrix_cleanup(&current, allocator);
                return EVAL_OUT_OF_INSTRUCTIONS;
            }
            {
                i += 1;
                const unsigned count = code[i].u32;
                if (count > stack_pos)
                {
                    EVAL_ERROR(error_stack, EVAL_STACK_UNDERFLOW,
                               "Sum for %u matrices specified, but only %u are on stack.", count, stack_pos);
                    clean_stack(stack, stack_pos, allocator);
                    matrix_cleanup(&current, allocator);
                    return EVAL_STACK_UNDERFLOW;
                }
                for (unsigned j = 0; j < count; ++j)
                {
                    matrix_t new;
                    stack_pos -= 1;
                    matrix_t *left = stack + stack_pos;
                    const eval_result_t res = matrix_add(order, &current, left, &new, allocator);
                    matrix_cleanup(&current, allocator);
                    matrix_cleanup(left, allocator);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(error_stack, res, "Could not add two matrices.");
                        clean_stack(stack, stack_pos, allocator);
                        matrix_cleanup(&current, allocator);
                        return res;
                    }
                    current = new;
                }
            }
            break;

        default:
            EVAL_ERROR(error_stack, EVAL_BAD_ENUM, "Unknown bytecode instruction %u.", (unsigned)op);

            clean_stack(stack, stack_pos, allocator);
            matrix_cleanup(&current, allocator);
            return EVAL_BAD_ENUM;
        }
        // printf(" Resulting matrix type %u (%u x %u), stack at %u.\n", current.type, current.base.rows,
        // current.base.cols, stack_pos);
    }
    clean_stack(stack, stack_pos, allocator);
    if (stack_pos != 0)
    {
        matrix_cleanup(&current, allocator);
        EVAL_ERROR(error_stack, EVAL_STACK_NOT_EMPTY, "Stack had %u elements at the end.", (unsigned)stack_pos);
        return EVAL_STACK_NOT_EMPTY;
    }

    switch (current.type)
    {
    case MATRIX_TYPE_IDENTITY: {
        const unsigned n = form_degrees_of_freedom_count(form, order);
        double *const restrict ptr = allocate(allocator, sizeof *ptr * n * n);
        if (!ptr)
        {
            EVAL_ERROR(error_stack, EVAL_FAILED_ALLOC, "Could not allocate memory for output identity matrix.");
            return EVAL_FAILED_ALLOC;
        }
        memset(ptr, 0, sizeof *ptr * n * n);
        for (unsigned i = 0; i < n; ++i)
        {
            ptr[i * n + i] = current.coefficient;
        }
        *p_out = (matrix_full_t){.base = {.type = MATRIX_TYPE_FULL, .rows = n, .cols = n}, .data = ptr};
    }
    break;

    case MATRIX_TYPE_FULL:
        matrix_multiply_inplace(&current.full, current.coefficient);
        *p_out = current.full;
        break;

    case MATRIX_TYPE_INCIDENCE:
        const eval_result_t res = incidence_to_full(current.incidence.incidence, order, p_out, allocator);
        if (res != EVAL_SUCCESS)
            return res;
        matrix_multiply_inplace(p_out, current.coefficient);
        break;

    default:
        EVAL_ERROR(error_stack, EVAL_BAD_ENUM, "Invalid matrix type for the output matrix %u.", current.type);
        return EVAL_BAD_ENUM;
    }

    return EVAL_SUCCESS;
}
