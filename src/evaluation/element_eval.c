//
// Created by jan on 15.2.2025.
//

#include "element_eval.h"
#include "../fem_space/fem_space.h"
#include "incidence.h"
#include "integrating_fields.h"

static void clean_stack(matrix_t *stack, const unsigned cnt, const allocator_callbacks *allocator)
{
    for (unsigned i = cnt; i > 0; --i)
    {
        matrix_cleanup(stack + i - 1, allocator);
    }
}

static mfv2d_result_t matrix_as_full(error_stack_t *error_stack, const matrix_t *this, const unsigned order,
                                     const form_order_t form, matrix_full_t *p_out,
                                     const allocator_callbacks *allocator)
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

static mfv2d_result_t operation_identity(matrix_t *current)
{
    if (current->type == MATRIX_TYPE_INVALID)
    {
        current->type = MATRIX_TYPE_IDENTITY;
        current->coefficient = 1.0;
    }
    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_scale(matrix_t *current, const matrix_op_t *operation)
{
    const matrix_op_scale_t *scale = (const matrix_op_scale_t *)operation;
    if (current->type == MATRIX_TYPE_INVALID)
    {
        current->type = MATRIX_TYPE_IDENTITY;
        current->coefficient = scale->k;
    }
    else
    {
        current->coefficient *= scale->k;
    }
    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_push(matrix_t *current, unsigned n_stack, unsigned *p_stack_pos,
                                     matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                     error_stack_t *error_stack, const matrix_full_t *initial)
{
    if (*p_stack_pos == n_stack)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_OVERFLOW, "Stack is full.");
        return MFV2D_STACK_OVERFLOW;
    }
    stack[*p_stack_pos] = *current;
    *p_stack_pos += 1;
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

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_incidence(matrix_t *current, const matrix_op_t *operation,
                                          const element_fem_space_2d_t *element_fem_space,
                                          const allocator_callbacks *allocator, error_stack_t *error_stack)
{
    const matrix_op_incidence_t *inc = (const matrix_op_incidence_t *)operation;
    const form_order_t beginning_order = inc->order;
    const unsigned transpose = inc->transpose;

    if (beginning_order < FORM_ORDER_0 || beginning_order > FORM_ORDER_1)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM,
                    "Incidence beginning order specified by current matrix is %s which is not valid.",
                    form_order_str(beginning_order));
        return MFV2D_BAD_ENUM;
    }

    const incidence_type_t t = (beginning_order - 1) + (transpose ? 2 : 0);

    matrix_t new_mat = {.type = MATRIX_TYPE_INVALID, .coefficient = 1.0};
    mfv2d_result_t res;
    switch (current->type)
    {
    case MATRIX_TYPE_INVALID:
        *current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t}, .coefficient = 1.0};
        break;
    case MATRIX_TYPE_IDENTITY:
        *current = (matrix_t){.incidence = {.base.type = MATRIX_TYPE_INCIDENCE, .incidence = t},
                              .coefficient = current->coefficient};
        break;

    case MATRIX_TYPE_INCIDENCE:
        res = incidence_to_full(current->type, element_fem_space->basis_xi->order, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not create the incidence matrix.");
            return res;
        }
        // FALLTHROUGH

    case MATRIX_TYPE_FULL:
        res = apply_incidence_to_full_left(t, element_fem_space->basis_xi->order, &current->full, &new_mat.full,
                                           allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res,
                        "Could apply an incidence matrix %s to a full matrix of dimensions  (%u, %u).",
                        incidence_type_str(t), current->full.base.rows, current->full.base.cols);
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

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_mass(matrix_t *current, const matrix_op_t *operation,
                                     element_fem_space_2d_t *element_fem_space, const allocator_callbacks *allocator,
                                     error_stack_t *error_stack)
{
    const matrix_op_mass_t *mass = (const matrix_op_mass_t *)operation;
    const form_order_t order = mass->order;
    if (order < FORM_ORDER_0 || order > FORM_ORDER_2)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM, "Mass type specified by current matrix is %s which is not valid.",
                    form_order_str(order));
        return MFV2D_BAD_ENUM;
    }
    const unsigned inverse = mass->invert;
    // const mass_mtx_indices_t t = (order - 1) + (inverse ? 3 : 0);

    const matrix_full_t *mass_mat = NULL;

    switch (order)
    {
    case FORM_ORDER_0:
        if (!inverse)
            mass_mat = element_mass_cache_get_node(element_fem_space);
        else
            mass_mat = element_mass_cache_get_node_inv(element_fem_space);
        break;

    case FORM_ORDER_1:
        if (!inverse)
            mass_mat = element_mass_cache_get_edge(element_fem_space);
        else
            mass_mat = element_mass_cache_get_edge_inv(element_fem_space);
        break;

    case FORM_ORDER_2:
        if (!inverse)
            mass_mat = element_mass_cache_get_surf(element_fem_space);
        else
            mass_mat = element_mass_cache_get_surf_inv(element_fem_space);
        break;

    default:
        ASSERT(0, "This was range checked before.");
    }

    if (mass_mat == NULL)
    {
        MFV2D_ERROR(error_stack, MFV2D_FAILED_ALLOC,
                    "Could not allocate memory for the mass matrix of order %s (inverse - %u).", form_order_str(order),
                    inverse);
        return MFV2D_FAILED_ALLOC;
    }

    mfv2d_result_t res;

    switch (current->type)
    {
    case MATRIX_TYPE_INVALID:
        current->coefficient = 1.0;
        // FALLTHROUGH
    case MATRIX_TYPE_IDENTITY:
        res = matrix_full_copy(mass_mat, &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not copy the mass matrix.");
            return res;
        }
        break;
    case MATRIX_TYPE_INCIDENCE:

        res = apply_incidence_to_full_right(current->incidence.incidence, element_fem_space->basis_xi->order, mass_mat,
                                            &current->full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res,
                        "Could right-apply incidence matrix %s to the mass matrix of order %s (inverse - %u).",
                        incidence_type_str(current->incidence.incidence), form_order_str(order), inverse);
            return res;
        }
        break;
    case MATRIX_TYPE_FULL:
        matrix_t new_mat = {};
        res = matrix_full_multiply(mass_mat, &current->full, &new_mat.full, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not multiply two full matrices with dims (%u, %u) and (%u, %u).",
                        mass_mat->base.rows, mass_mat->base.cols, current->base.rows, current->base.cols);
            return res;
        }
        matrix_cleanup(current, allocator);
        *current = new_mat;
        current->coefficient = 1.0;
        break;
    }

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_matmul(matrix_t *current, const element_fem_space_2d_t *element_fem_space,
                                       unsigned n_stack, unsigned *p_stack_pos, matrix_t stack[restrict n_stack],
                                       const allocator_callbacks *allocator, error_stack_t *error_stack)
{
    if (*p_stack_pos == 0)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_UNDERFLOW, "Matrix multiply operation with nothing on the stack.");
        return MFV2D_STACK_UNDERFLOW;
    }
    *p_stack_pos -= 1;

    const matrix_t right = stack[*p_stack_pos];
    stack[*p_stack_pos] = (matrix_t){.type = MATRIX_TYPE_INVALID};
    matrix_t new_mat;
    const mfv2d_result_t res =
        matrix_multiply(error_stack, element_fem_space->basis_xi->order, &right, current, &new_mat, allocator);
    matrix_cleanup(stack + *p_stack_pos, allocator);
    matrix_cleanup(current, allocator);
    if (res != MFV2D_SUCCESS)
    {
        MFV2D_ERROR(error_stack, res, "Failed multiplying two matrices (%u x %u and %u).", right.base.rows,
                    right.base.cols, current->type);
        return res;
    }
    *current = new_mat;

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_sum(matrix_t *current, const matrix_op_t *operation,
                                    const element_fem_space_2d_t *element_fem_space, unsigned n_stack,
                                    unsigned *p_stack_pos, matrix_t stack[restrict n_stack],
                                    const allocator_callbacks *allocator, error_stack_t *error_stack)
{
    const matrix_op_sum_t *sum = (const matrix_op_sum_t *)operation;
    const unsigned count = sum->n;

    if (count > *p_stack_pos)
    {
        MFV2D_ERROR(error_stack, MFV2D_STACK_UNDERFLOW, "Sum for %u matrices specified, but only %u are on stack.",
                    count, *p_stack_pos);
        return MFV2D_STACK_UNDERFLOW;
    }
    for (unsigned j = 0; j < count; ++j)
    {
        matrix_t new;
        *p_stack_pos -= 1;
        matrix_t *left = stack + *p_stack_pos;
        const mfv2d_result_t res = matrix_add(element_fem_space->basis_xi->order, current, left, &new, allocator);
        matrix_cleanup(current, allocator);
        matrix_cleanup(left, allocator);
        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Could not add two matrices.");
            return res;
        }
        *current = new;
    }

    return MFV2D_SUCCESS;
}

static mfv2d_result_t operation_interprod(matrix_t *current, const matrix_op_t *operation,
                                          const element_fem_space_2d_t *element_fem_space,
                                          const field_information_t *integration_fields,
                                          const allocator_callbacks *allocator, error_stack_t *error_stack)
{
    const matrix_op_interprod_t *const interprod = (const matrix_op_interprod_t *)operation;
    const form_order_t order = interprod->order;
    const unsigned field_index = interprod->field_index;
    const unsigned dual = interprod->dual;
    const unsigned adjoint = interprod->adjoint;

    if (order != FORM_ORDER_1 && order != FORM_ORDER_2)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM,
                    "InterProd specified with starting order that is not 1 or 2, but is %s", form_order_str(order));
        return MFV2D_BAD_ENUM;
    }
    if (field_index >= integration_fields->n_fields)
    {
        MFV2D_ERROR(error_stack, MFV2D_BAD_ENUM,
                    "InterProd specified with field with index %u, but only %u fields exits.", field_index,
                    integration_fields->n_fields);
        return MFV2D_BAD_ENUM;
    }
    const double *const restrict field = integration_fields->fields[field_index];

    matrix_t this = {.type = MATRIX_TYPE_FULL, .coefficient = 1.0};
    mfv2d_result_t res;

    if (!adjoint)
    {
        if (!dual)
        {
            if (order == FORM_ORDER_1)
            {
                res = compute_mass_matrix_node_edge(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = 1.0;
            }
            else // if (order == FORM_ORDER_2)
            {
                res = compute_mass_matrix_edge_surf(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = 1.0;
            }
        }
        else
        {
            if (order == FORM_ORDER_1)
            {
                res = compute_mass_matrix_edge_surf(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = -1.0;
            }
            else // if (order == FORM_ORDER_2)
            {
                res = compute_mass_matrix_node_edge(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = 1.0;
            }
        }
    }
    else
    {
        if (!dual)
        {
            if (order == FORM_ORDER_1)
            {
                res = compute_mass_matrix_node_edge(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = -1.0;
            }
            else // if (order == FORM_ORDER_2)
            {
                res = compute_mass_matrix_edge_edge(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = -1.0;
            }
        }
        else
        {
            if (order == FORM_ORDER_1)
            {
                res = compute_mass_matrix_edge_surf(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = -1.0;
            }
            else // if (order == FORM_ORDER_2)
            {
                res = compute_mass_matrix_edge_edge(element_fem_space->fem_space, &this.full, allocator, field,
                                                    (int)dual);
                this.coefficient = 1.0;
            }
        }
    }

    if (res != MFV2D_SUCCESS)
    {
        MFV2D_ERROR(error_stack, res, "Could not compute the mixed matrix.");
        return res;
    }

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

        res = apply_incidence_to_full_right(current->incidence.incidence, element_fem_space->basis_xi->order,
                                            &this.full, &current->full, allocator);
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
    current->coefficient *= this.coefficient;
    matrix_cleanup(&this, allocator);

    return MFV2D_SUCCESS;
}

MFV2D_INTERNAL
mfv2d_result_t evaluate_block(error_stack_t *error_stack, const form_order_t form, const unsigned order,
                              const bytecode_t *code, element_fem_space_2d_t *element_fem_space,
                              const field_information_t *value_fields, const unsigned n_stack,
                              matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                              matrix_full_t *p_out, const matrix_full_t *initial)
{
    const unsigned n_ops = code->count;

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

    unsigned stack_pos = 0;
    mfv2d_result_t res = MFV2D_SUCCESS;
    for (unsigned i = 0; i < n_ops; ++i)
    {
        const matrix_op_t *const op = &code->ops[i];
        switch (op->type)
        {
        case MATOP_IDENTITY:
            res = operation_identity(&current);
            break;

        case MATOP_MASS:
            res = operation_mass(&current, op, element_fem_space, allocator, error_stack);
            break;

        case MATOP_INCIDENCE:
            res = operation_incidence(&current, op, element_fem_space, allocator, error_stack);
            break;

        case MATOP_PUSH:
            res = operation_push(&current, n_stack, &stack_pos, stack, allocator, error_stack, initial);
            break;

        case MATOP_MATMUL:
            res = operation_matmul(&current, element_fem_space, n_stack, &stack_pos, stack, allocator, error_stack);
            break;

        case MATOP_SCALE:
            res = operation_scale(&current, op);
            break;

        case MATOP_SUM:
            res = operation_sum(&current, op, element_fem_space, n_stack, &stack_pos, stack, allocator, error_stack);
            break;

        case MATOP_INTERPROD:
            res = operation_interprod(&current, op, element_fem_space, value_fields, allocator, error_stack);
            break;

        default:
            res = MFV2D_BAD_ENUM;
            break;
        }

        if (res != MFV2D_SUCCESS)
        {
            MFV2D_ERROR(error_stack, res, "Failed evaluating matrix operation %u (%s).", i,
                        matrix_op_type_str(op->type));
            break;
        }
    }

    clean_stack(stack, n_stack, allocator);
    if (res != MFV2D_SUCCESS)
    {
        matrix_cleanup(&current, allocator);
        return res;
    }

    return matrix_as_full(error_stack, &current, order, form, p_out, allocator);
}
