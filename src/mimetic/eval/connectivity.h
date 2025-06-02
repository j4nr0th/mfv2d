//
// Created by jan on 21.2.2025.
//

#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include "../manifold2d.h"
#include "error.h"
#include "evaluation.h"

/**
 *
 * @param primal Primal manifold.
 * @param dual Dual manifold.
 * @param n_elements Number of elements.
 * @param n_forms Number of unknown forms.
 * @param forms Orders of differential forms.
 * @param element_orders Orders of elements.
 * @param element_offsets Offsets of element DoFs.
 * @param dof_offsets Offsets of DoFs inside the element.
 * @param out_n_equations Pointer to receive the number of equations.
 * @param out_equation_lengths Pointer to receive array of element lengths.
 * @param out_indices Pointer to receive the array of equation indices.
 * @param allocator Allocator to use to allocate memory for the arrays.
 * @param error_stack
 * @return EVAL_SUCCESS if successful.
 */
eval_result_t generate_connectivity_equations(const manifold2d_object_t *primal, const manifold2d_object_t *dual,
                                              unsigned n_elements, unsigned n_forms,
                                              const form_order_t forms[const static n_forms],
                                              const unsigned element_orders[const static n_elements],
                                              const unsigned element_offsets[const static n_elements + 1],
                                              const unsigned dof_offsets[const static n_elements * (n_forms + 1)],
                                              unsigned *out_n_equations, unsigned **out_equation_lengths,
                                              unsigned **out_indices, const allocator_callbacks *allocator,
                                              error_stack_t *error_stack);

#endif // CONNECTIVITY_H
