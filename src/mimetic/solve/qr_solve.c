//
// Created by jan on 18.3.2025.
//

#include "qr_solve.h"
#include "../../common_defines.h"

static void print_svec(const svector_t *this)
{
    if (this->count != 0)
    {
        printf("(%" PRIu64 ", %g)", this->entries[0].index, this->entries[0].value);
    }
    for (uint64_t i = 1; i < this->count; ++i)
    {
        printf(", (%" PRIu64 ", %g)", this->entries[i].index, this->entries[i].value);
    }
}

/**
 * Perform QR decomposition by applying Givens rotations to the split matrix.
 *
 * @param mat Matrix that will be decomposed.
 * @param p_ng Pointer that will receive the number of Given's rotations required.
 * @param p_givens Pointer that will receive an array (pointer) of p_ng Given's rotations that were performed
 * @param allocator Allocator callbacks that will be used.
 * @return Zero if successful.
 */
int decompose_qr(const lil_matrix_t *mat, uint64_t *p_ng, givens_rotation_t **const p_givens,
                 const allocator_callbacks *allocator)
{
    // Try and guess how many Givens rotations will be needed
    uint64_t n_givens = 0;
    uint64_t capacity_givens = mat->rows;
    givens_rotation_t *givens = allocate(allocator, sizeof(*givens) * capacity_givens);
    if (!givens)
    {
        return -1;
    }

    svector_t out_j, out_i;
    // Create two new vectors, which will be re-used by swapping with old rows.
    // This way, we (ideally) only need to allocate twice. Worst case, we need
    // to still reallocate each row once, but that's it.
    if (sparse_vector_new(&out_i, mat->rows, 1, allocator) || sparse_vector_new(&out_j, mat->rows, 1, allocator))
    {
        deallocate(allocator, givens);
        sparse_vec_del(&out_i, allocator);
        sparse_vec_del(&out_j, allocator);
        return -1;
    }

    for (uint64_t j_row = 1; j_row < mat->rows; ++j_row)
    {
        // Row that will be eliminated
        const svector_t *const row_j = mat->row_data + j_row;
        uint64_t i_row;
        while (row_j->count > 0 && (i_row = row_j->entries[0].index) < j_row)
        {
            // Loop while there are entries which are below the diagonal.
            if (row_j->entries[0].value == 0)
            {
                // Already eliminated.
                memmove(row_j->entries, row_j->entries + 1, sizeof(*row_j->entries) * (row_j->count - 1));
                mat->row_data[j_row].count -= 1;
                continue;
            }

            const svector_t *const row_i = mat->row_data + i_row;
            ASSERT(row_i->entries[0].index == i_row, "Top row does not begin on the diagonal.");
            givens_rotation_t g = {
                .n = mat->rows, .k = i_row, .l = j_row, .c = row_i->entries[0].value, .s = row_j->entries[0].value};
            const scalar_t mag = hypot(g.c, g.s);
            g.c /= mag;
            g.s /= mag;
            if (apply_givens_rotation(g.c, g.s, row_i, row_j, &out_i, &out_j, 1, allocator))
            {
                // Failed a malloc or something like that
                deallocate(allocator, givens);
                sparse_vec_del(&out_i, allocator);
                sparse_vec_del(&out_j, allocator);
            }

            if (capacity_givens <= n_givens)
            {
                const uint64_t new_capacity = capacity_givens + (capacity_givens > 0 ? capacity_givens : 8 + n_givens);
                givens_rotation_t *new_ptr = reallocate(allocator, givens, sizeof *new_ptr * new_capacity);
                if (!new_ptr)
                {
                    deallocate(allocator, givens);
                    sparse_vec_del(&out_i, allocator);
                    sparse_vec_del(&out_j, allocator);
                    return -1;
                }
                givens = new_ptr;
                capacity_givens = new_capacity;
            }
            givens[n_givens] = g;
            n_givens += 1;

            // Swap output buffer with rows
            {
                svector_t tmp = mat->row_data[i_row];

                mat->row_data[i_row] = out_i;
                out_i = tmp;

                tmp = mat->row_data[j_row];
                mat->row_data[j_row] = out_j;
                out_j = tmp;
            }
        }
    }
    sparse_vec_del(&out_i, allocator);
    sparse_vec_del(&out_j, allocator);

    *p_ng = n_givens;
    *p_givens = givens;

    return 0;
}
