//
// Created by jan on 16.2.2025.
//

#ifndef INCIDENCE_H
#define INCIDENCE_H

#include "error.h"
#include "evaluation.h"

MFV2D_INTERNAL
eval_result_t apply_e10_left(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                             const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e21_left(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                             const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e10_right(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e21_right(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e10t_left(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e21t_left(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                              const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e10t_right(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_e21t_right(unsigned order, const matrix_full_t *in, matrix_full_t *out,
                               const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_incidence_to_full_left(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                           matrix_full_t *p_out, const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t apply_incidence_to_full_right(const incidence_type_t type, const unsigned order, const matrix_full_t *in,
                                            matrix_full_t *p_out, const allocator_callbacks *allocator);

MFV2D_INTERNAL
eval_result_t incidence_to_full(const incidence_type_t type, const unsigned order, matrix_full_t *p_out,
                                const allocator_callbacks *allocator);

#endif // INCIDENCE_H
