//
// Created by jan on 15.2.2025.
//

#ifndef ELEMENT_EVAL_H
#define ELEMENT_EVAL_H

#include "../common.h"
#include "../common_defines.h"

#include "fem_space.h"
#include "bytecode.h"
#include "error.h"
#include "system_template.h"

MFV2D_INTERNAL
eval_result_t evaluate_block(error_stack_t *error_stack, form_order_t form, unsigned order,
                                            const bytecode_t *code, const fem_space_2d_t *fem_space,
                                            const field_information_t *vector_fields, unsigned n_stack,
                                            matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                            matrix_full_t *p_out, const matrix_full_t *initial);

#endif // ELEMENT_EVAL_H
