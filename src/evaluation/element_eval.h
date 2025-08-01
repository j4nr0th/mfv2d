//
// Created by jan on 15.2.2025.
//

#ifndef ELEMENT_EVAL_H
#define ELEMENT_EVAL_H

#include "../common/common.h"
#include "../common/common_defines.h"

#include "../common/error.h"
#include "../fem_space/element_fem_space.h"
#include "../fem_space/fem_space.h"
#include "bytecode.h"
#include "system_template.h"

MFV2D_INTERNAL
mfv2d_result_t evaluate_block(error_stack_t *error_stack, form_order_t form, unsigned order, const bytecode_t *code,
                              element_fem_space_2d_t *element_cache, const field_information_t *value_fields,
                              unsigned n_stack, matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                              matrix_full_t *p_out, const matrix_full_t *initial);

#endif // ELEMENT_EVAL_H
