//
// Created by jan on 15.2.2025.
//

#ifndef EVALUATION_H
#define EVALUATION_H

#include "../common.h"
#include "../common_defines.h"

#include "../error.h"
#include "bytecode.h"
#include "precomp.h"
#include "system_template.h"

MFV2D_INTERNAL
mfv2d_result_t evaluate_element_term_sibling(error_stack_t *error_stack, form_order_t form, unsigned order,
                                             const bytecode_t *code, precompute_t *precomp,
                                             const field_information_t *vector_fields, unsigned n_stack,
                                             matrix_t stack[restrict n_stack], const allocator_callbacks *allocator,
                                             matrix_full_t *p_out, const matrix_full_t *initial);

#endif // EVALUATION_H
