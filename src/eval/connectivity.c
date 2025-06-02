//
// Created by jan on 21.2.2025.
//

#include "connectivity.h"

static unsigned *allocate_equation(const unsigned n_indices, unsigned *p_capacity_offsets, unsigned *p_capacity_indices,
                                   unsigned *p_count_offsets, unsigned **pp_offsets, unsigned **pp_indices,
                                   const allocator_callbacks *allocator)
{
    // Check that we have enough length in the offsets array
    const unsigned n_eq = *p_count_offsets;
    unsigned capacity_offsets = *p_capacity_offsets;
    unsigned *p_offsets = *pp_offsets;
    if ((n_eq + 1) >= capacity_offsets)
    {
        capacity_offsets += (capacity_offsets > 0) ? (capacity_offsets) : 64;
        *p_capacity_offsets = capacity_offsets;
        unsigned *const new_ptr = reallocate(allocator, p_offsets, sizeof *p_offsets * capacity_offsets);
        if (!new_ptr)
            return NULL;
        p_offsets = new_ptr;
        *pp_offsets = p_offsets;
    }
    *p_count_offsets = (n_eq + 1);

    // Check that we have enough length in the indices array
    const unsigned n = (n_eq == 0 ? 0 : p_offsets[n_eq - 1]) + n_indices;
    unsigned capacity_indices = *p_capacity_indices;
    unsigned *p_indices = *pp_indices;
    if (n >= capacity_indices)
    {
        while ((n >= capacity_indices))
        {
            capacity_indices += (capacity_indices > 0) ? (capacity_indices) : 64;
        }
        *p_capacity_indices = capacity_indices;
        unsigned *const new_ptr = reallocate(allocator, p_indices, sizeof *p_indices * capacity_indices);
        if (!new_ptr)
            return NULL;
        p_indices = new_ptr;
        *pp_indices = p_indices;
    }
    p_offsets[n_eq] = n;

    return p_indices + n - n_indices;
}

static void edge_dof_offset_stride(const unsigned order, const unsigned side, unsigned *offset, int *stride)
{
    switch (side)
    {
    case 0: // Bottom
        *offset = 0;
        *stride = +1;
        break;
    case 1: // Right
        *offset = order * (order + 1) + order;
        *stride = (int)order + 1;
        break;
    case 2: // Top
        *offset = order * (order + 1) - 1;
        *stride = -1;
        break;
    case 3: // Left
        *offset = 2 * order * (order + 1) - (order + 1);
        *stride = -((int)order + 1);
        break;
    default:
        return;
    }
}

static eval_result_t line_continuity_equations(const unsigned order_0, const unsigned side_0, const unsigned offset_0,
                                               const unsigned order_1, const unsigned side_1, const unsigned offset_1,
                                               unsigned *p_capacity_offsets, unsigned *p_capacity_indices,
                                               unsigned *p_count_offsets, unsigned **pp_offsets, unsigned **pp_indices,
                                               const allocator_callbacks *allocator, error_stack_t *error_stack)
{
    if (order_0 != order_1)
    {
        EVAL_ERROR(error_stack, EVAL_ORDER_MISMATCH,
                   "Elements with mismatched order are not (yet) allowed (%u and %u).", order_0, order_1);
        return EVAL_ORDER_MISMATCH;
    }

    unsigned o_0, o_1;
    int s_0, s_1;

    edge_dof_offset_stride(order_0, side_0, &o_0, &s_0);
    edge_dof_offset_stride(order_1, side_1, &o_1, &s_1);

    for (unsigned i = 0; i < order_0; ++i)
    {
        const unsigned p0 = ((int)o_0 + i * s_0) + offset_0;
        const unsigned p1 = ((int)o_1 + (order_0 - 1 - i) * s_1) + offset_1;

        unsigned *const ptr = allocate_equation(2, p_capacity_offsets, p_capacity_indices, p_count_offsets, pp_offsets,
                                                pp_indices, allocator);
        if (!ptr)
        {
            EVAL_ERROR(error_stack, EVAL_FAILED_ALLOC, "Could not allocate memory for a new equation.");
            return EVAL_FAILED_ALLOC;
        }
        ptr[0] = p0;
        ptr[1] = p1;
    }

    return EVAL_SUCCESS;
}

static void node_dof_offset_stride(const unsigned order, const unsigned side, unsigned *offset, int *stride)
{
    switch (side)
    {
    case 0: // Bottom
        *offset = 0;
        *stride = +1;
        break;
    case 1: // Right
        *offset = order;
        *stride = (int)order + 1;
        break;
    case 2: // Top
        *offset = (order + 1) * (order + 1) - 1;
        *stride = -1;
        break;
    case 3: // Left
        *offset = order * (order + 1);
        *stride = -((int)order + 1);
        break;
    default:
        return;
    }
}

static eval_result_t inner_node_continuity_equations(const unsigned order_0, const unsigned side_0,
                                                     const unsigned offset_0, const unsigned order_1,
                                                     const unsigned side_1, const unsigned offset_1,
                                                     unsigned *p_capacity_offsets, unsigned *p_capacity_indices,
                                                     unsigned *p_count_offsets, unsigned **pp_offsets,
                                                     unsigned **pp_indices, const allocator_callbacks *allocator,
                                                     error_stack_t *error_stack)
{
    if (order_0 != order_1)
    {
        EVAL_ERROR(error_stack, EVAL_ORDER_MISMATCH,
                   "Elements with mismatched order are not (yet) allowed (%u and %u).", order_0, order_1);
        return EVAL_ORDER_MISMATCH;
    }

    unsigned o_0, o_1;
    int s_0, s_1;

    node_dof_offset_stride(order_0, side_0, &o_0, &s_0);
    node_dof_offset_stride(order_1, side_1, &o_1, &s_1);

    // Skip first (0) and last (order_0)
    for (unsigned i = 1; i < order_0; ++i)
    {
        const unsigned p0 = ((int)o_0 + i * s_0) + offset_0;
        const unsigned p1 = ((int)o_1 + (order_0 - i) * s_1) + offset_1;

        unsigned *const ptr = allocate_equation(2, p_capacity_offsets, p_capacity_indices, p_count_offsets, pp_offsets,
                                                pp_indices, allocator);
        if (!ptr)
        {
            EVAL_ERROR(error_stack, EVAL_FAILED_ALLOC, "Could not allocate memory for a new equation.");
            return EVAL_FAILED_ALLOC;
        }
        ptr[0] = p0;
        ptr[1] = p1;
    }

    return EVAL_SUCCESS;
}

static eval_result_t corner_node_continuity_equations(const unsigned order_0, const unsigned side_0,
                                                      const unsigned offset_0, const unsigned order_1,
                                                      const unsigned side_1, const unsigned offset_1,
                                                      unsigned *p_capacity_offsets, unsigned *p_capacity_indices,
                                                      unsigned *p_count_offsets, unsigned **pp_offsets,
                                                      unsigned **pp_indices, const allocator_callbacks *allocator,
                                                      error_stack_t *error_stack)
{
    if (order_0 != order_1)
    {
        EVAL_ERROR(error_stack, EVAL_ORDER_MISMATCH,
                   "Elements with mismatched order are not (yet) allowed (%u and %u).", order_0, order_1);
        return EVAL_ORDER_MISMATCH;
    }

    unsigned o_0, o_1;
    int s_0, s_1;

    node_dof_offset_stride(order_0, side_0, &o_0, &s_0);
    node_dof_offset_stride(order_1, side_1, &o_1, &s_1);

    // Only first (0) one
    const unsigned p0 = ((int)o_0 + 0 * s_0) + offset_0;
    const unsigned p1 = ((int)o_1 + (order_0 - 0) * s_1) + offset_1;

    unsigned *const ptr = allocate_equation(2, p_capacity_offsets, p_capacity_indices, p_count_offsets, pp_offsets,
                                            pp_indices, allocator);
    if (!ptr)
    {
        EVAL_ERROR(error_stack, EVAL_FAILED_ALLOC, "Could not allocate memory for a new equation.");
        return EVAL_FAILED_ALLOC;
    }
    ptr[0] = p0;
    ptr[1] = p1;

    return EVAL_SUCCESS;
}

static eval_result_t find_rect_surf_index(const surface_t *surf, const unsigned idx, unsigned *p_out)
{
    if (surf->n_lines != 4)
        return EVAL_NOT_SQUARE;

    for (unsigned i = 0; i < 4; ++i)
    {
        if (surf->values[i].index == idx)
        {
            *p_out = i;
            return EVAL_SUCCESS;
        }
    }

    return EVAL_NOT_IN_SURFACE;
}

eval_result_t generate_connectivity_equations(const manifold2d_object_t *primal, const manifold2d_object_t *dual,
                                              const unsigned n_elements, const unsigned n_forms,
                                              const form_order_t forms[const static n_forms],
                                              const unsigned element_orders[const static n_elements],
                                              const unsigned element_offsets[const static n_elements + 1],
                                              const unsigned dof_offsets[const static n_elements * (n_forms + 1)],
                                              unsigned *out_n_equations, unsigned **out_equation_lengths,
                                              unsigned **out_indices, const allocator_callbacks *allocator,
                                              error_stack_t *error_stack)
{
    unsigned capacity_offsets = 0;
    unsigned capacity_indices = 0;
    unsigned count_offsets = 0;
    unsigned *equation_offsets = NULL;
    unsigned *equation_indices = NULL;

    // Continuity with the dual lines (1-forms and 0-forms on the interior of the edges).
    for (unsigned i_line = 0; i_line < dual->n_lines; ++i_line)
    {
        const line_t *dual_line = dual->lines + i_line;

        const geo_id_t idx_other = dual_line->begin;
        const geo_id_t idx_self = dual_line->end;

        if (idx_other.index == GEO_ID_INVALID || idx_self.index == GEO_ID_INVALID)
        {
            continue;
        }

        const unsigned offset_self = element_offsets[idx_self.index];
        const unsigned offset_other = element_offsets[idx_other.index];

        for (unsigned i_form = 0; i_form < n_forms; ++i_form)
        {
            const form_order_t form = forms[i_form];
            eval_result_t res = EVAL_SUCCESS;

            const unsigned self_var_offset = dof_offsets[i_form * n_elements + idx_self.index];
            const unsigned other_var_offset = dof_offsets[i_form * n_elements + idx_other.index];

            const unsigned col_off_self = offset_self + self_var_offset;
            const unsigned col_off_other = offset_other + other_var_offset;

            const surface_t surf_self = {.n_lines = primal->surf_counts[idx_self.index + 1] -
                                                    primal->surf_counts[idx_self.index],
                                         .values = primal->surf_lines + primal->surf_counts[idx_self.index]};
            const surface_t surf_other = {.n_lines = primal->surf_counts[idx_other.index + 1] -
                                                     primal->surf_counts[idx_other.index],
                                          .values = primal->surf_lines + primal->surf_counts[idx_other.index]};
            unsigned side_self, side_other;
            res = find_rect_surf_index(&surf_self, i_line, &side_self);
            if (res != EVAL_SUCCESS)
            {
                EVAL_ERROR(error_stack, res,
                           "Line with index %u was not contained in the primal surface %u even though the dual point "
                           "of that surface was in the dual line.",
                           i_line, idx_self.index);
                deallocate(allocator, equation_offsets);
                deallocate(allocator, equation_indices);
                return res;
            }
            res = find_rect_surf_index(&surf_other, i_line, &side_other);
            if (res != EVAL_SUCCESS)
            {
                EVAL_ERROR(error_stack, res,
                           "Line with index %u was not contained in the primal surface %u even though the dual point "
                           "of that surface was in the dual line.",
                           i_line, idx_other.index);
                deallocate(allocator, equation_offsets);
                deallocate(allocator, equation_indices);
                return res;
            }

            switch (form)
            {
            case FORM_ORDER_0:
                // Just do internal continuity now, we do external continuity later.
                res = inner_node_continuity_equations(element_orders[idx_self.index], side_self, col_off_self,
                                                      element_orders[idx_other.index], side_other, col_off_other,
                                                      &capacity_offsets, &capacity_indices, &count_offsets,
                                                      &equation_offsets, &equation_indices, allocator, error_stack);
                break;
            case FORM_ORDER_1:
                // Just do internal continuity.
                res = line_continuity_equations(element_orders[idx_self.index], side_self, col_off_self,
                                                element_orders[idx_other.index], side_other, col_off_other,
                                                &capacity_offsets, &capacity_indices, &count_offsets, &equation_offsets,
                                                &equation_indices, allocator, error_stack);
                break;
            case FORM_ORDER_2:
                // Nothing to do, so skip it.
                break;
            default:
                EVAL_ERROR(error_stack, EVAL_BAD_ENUM, "Invalid form order enum value %u.", (unsigned)form);
                deallocate(allocator, equation_offsets);
                deallocate(allocator, equation_indices);
                return EVAL_BAD_ENUM;
            }
            if (res != EVAL_SUCCESS)
            {
                EVAL_ERROR(error_stack, res,
                           "Failed applying continuity for the %u-th form between elements %u and %u.", i_form,
                           idx_self.index, idx_other.index);
                deallocate(allocator, equation_offsets);
                deallocate(allocator, equation_indices);
                return res;
            }
        }
    }

    // Continuity with the dual surfaces (0-forms on the corners).
    for (unsigned i_surf = 0; i_surf < dual->n_surfaces; ++i_surf)
    {
        const surface_t dual_surf = {.n_lines = dual->surf_counts[i_surf + 1] - dual->surf_counts[i_surf],
                                     .values = dual->surf_lines + dual->surf_counts[i_surf]};
        unsigned valid = 0;
        for (unsigned i = 0; i < dual_surf.n_lines; ++i)
        {
            const line_t *dual_line = primal->lines + dual_surf.values[i].index;
            if (dual_line->begin.index != GEO_ID_INVALID && dual_line->end.index != GEO_ID_INVALID)
            {
                valid += 1;
            }
        }

        if (valid == dual_surf.n_lines)
        {
            valid -= 1;
        }

        for (unsigned i_ln = 0, proper = 0; i_ln < dual_surf.n_lines && proper < valid; ++i_ln)
        {
            const geo_id_t id_line = dual_surf.values[i_ln];
            const line_t *dual_line = dual->lines + id_line.index;
            geo_id_t idx_other, idx_self;

            if (id_line.reverse)
            {
                idx_other = dual_line->end;
                idx_self = dual_line->begin;
            }
            else
            {
                idx_other = dual_line->begin;
                idx_self = dual_line->end;
            }

            if (idx_other.index == GEO_ID_INVALID || idx_self.index == GEO_ID_INVALID)
            {
                continue;
            }
            proper += 1;

            const unsigned offset_self = element_offsets[idx_self.index];
            const unsigned offset_other = element_offsets[idx_other.index];

            for (unsigned i_form = 0; i_form < n_forms; ++i_form)
            {
                const form_order_t form = forms[i_form];
                if (form != FORM_ORDER_0)
                    continue;

                const unsigned self_var_offset = dof_offsets[i_form * n_elements + idx_self.index];
                const unsigned other_var_offset = dof_offsets[i_form * n_elements + idx_other.index];

                const unsigned col_off_self = offset_self + self_var_offset;
                const unsigned col_off_other = offset_other + other_var_offset;

                const surface_t surf_self = {.n_lines = primal->surf_counts[idx_self.index + 1] -
                                                        primal->surf_counts[idx_self.index],
                                             .values = primal->surf_lines + primal->surf_counts[idx_self.index]};
                const surface_t surf_other = {.n_lines = primal->surf_counts[idx_other.index + 1] -
                                                         primal->surf_counts[idx_other.index],
                                              .values = primal->surf_lines + primal->surf_counts[idx_other.index]};
                unsigned side_self, side_other;
                eval_result_t res = find_rect_surf_index(&surf_self, id_line.index, &side_self);
                if (res != EVAL_SUCCESS)
                {
                    EVAL_ERROR(
                        error_stack, res,
                        "Line with index %u was not contained in the primal surface %u even though the dual point "
                        "of that surface was in the dual surface %u.",
                        id_line.index, idx_self.index, i_surf);
                    deallocate(allocator, equation_offsets);
                    deallocate(allocator, equation_indices);
                    return res;
                }
                res = find_rect_surf_index(&surf_other, id_line.index, &side_other);
                if (res != EVAL_SUCCESS)
                {
                    EVAL_ERROR(
                        error_stack, res,
                        "Line with index %u was not contained in the primal surface %u even though the dual point "
                        "of that surface was in the dual surface %u.",
                        id_line.index, idx_other.index, i_surf);
                    deallocate(allocator, equation_offsets);
                    deallocate(allocator, equation_indices);
                    return res;
                }

                res = corner_node_continuity_equations(element_orders[idx_self.index], side_self, col_off_self,
                                                       element_orders[idx_other.index], side_other, col_off_other,
                                                       &capacity_offsets, &capacity_indices, &count_offsets,
                                                       &equation_offsets, &equation_indices, allocator, error_stack);

                if (res != EVAL_SUCCESS)
                {
                    EVAL_ERROR(error_stack, res,
                               "Failed applying continuity for the %u-th form between elements %u and %u.", i_form,
                               idx_self.index, idx_other.index);
                    deallocate(allocator, equation_offsets);
                    deallocate(allocator, equation_indices);
                    return res;
                }
            }
        }
    }

    *out_equation_lengths = equation_offsets;
    *out_indices = equation_indices;
    *out_n_equations = count_offsets;

    return EVAL_SUCCESS;
}
