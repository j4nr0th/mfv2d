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

int apply_givens_rotation(const scalar_t c, const scalar_t s, const svector_t *row_i, const svector_t *row_j,
                          svector_t *restrict out_i, svector_t *restrict out_j, const unsigned cut_j,
                          const allocator_callbacks *allocator)
{
    (void)cut_j;
    ASSERT(row_i->n == row_j->n, "Input vectors must have the same size (%" PRIu64 " vs %" PRIu64 ").", row_i->n,
           row_j->n);
    ASSERT(out_i->n == out_j->n, "Output vectors must have the same size (%" PRIu64 " vs %" PRIu64 ").", out_i->n,
           out_j->n);
    uint64_t max_elements = row_i->count + row_j->count;

    // Can't have more elements than the full row.
    if (max_elements > row_i->n)
    {
        max_elements = row_i->n;
    }

    if (sparse_vec_resize(out_i, max_elements, allocator) || sparse_vec_resize(out_j, max_elements, allocator))
    {
        return -1;
    }

    uint64_t idx_i, idx_j, pos;
    for (idx_i = 0, idx_j = 0, pos = 0; idx_i < row_i->count && idx_j < row_j->count; ++pos)
    {
        scalar_t vi = 0.0, vj = 0.0;
        uint64_t pv;
        // row I still available
        // row J still available
        if (row_i->entries[idx_i].index < row_j->entries[idx_j].index)
        {
            // I before J
            vi = row_i->entries[idx_i].value;
            pv = row_i->entries[idx_i].index;
            idx_i += 1;
        }
        else if (row_i->entries[idx_i].index > row_j->entries[idx_j].index)
        {
            // J before I
            vj = row_j->entries[idx_j].value;
            pv = row_j->entries[idx_j].index;
            idx_j += 1;
        }
        else
        {
            // I and J equal
            vi = row_i->entries[idx_i].value;
            vj = row_j->entries[idx_j].value;
            pv = row_j->entries[idx_j].index;
            idx_i += 1;
            idx_j += 1;
        }
        out_i->entries[pos].value = c * vi + s * vj;
        out_i->entries[pos].index = pv;
        // Skip the first cut_j the account of the fact that they are eliminated.
        if (pos >= 1)
        {
            out_j->entries[pos - 1].value = -s * vi + c * vj;
            out_j->entries[pos - 1].index = pv;
        }
    }

    // Remainder of I or J still needs to be handled (Unless they're of equal length)
    if (idx_i < row_i->count)
    {
        while (idx_i < row_i->count)
        {
            const scalar_t vi = row_i->entries[idx_i].value;
            const uint64_t pv = row_i->entries[idx_i].index;
            idx_i += 1;
            out_i->entries[pos].value = c * vi;
            out_i->entries[pos].index = pv;
            // Skip the first cut_j the account of the fact that they are eliminated.
            out_j->entries[pos - 1].value = -s * vi;
            out_j->entries[pos - 1].index = pv;
            pos += 1;
        }
    }
    else // if (idx_j < row_j->count)
    {
        while (idx_j < row_j->count)
        {

            // row J still available
            const scalar_t vj = row_j->entries[idx_j].value;
            const uint64_t pv = row_j->entries[idx_j].index;
            idx_j += 1;
            out_i->entries[pos].value = s * vj;
            out_i->entries[pos].index = pv;
            out_j->entries[pos - 1].value = c * vj;
            out_j->entries[pos - 1].index = pv;
            pos += 1;
        }
    }

    // Adjust the element counts for out_i and out_j.
    out_i->count = pos;
    out_j->count = pos - 1;

    return 0;
}
/**
 * Perform QR decomposition by applying Givens rotations to the split matrix.
 *
 * @param n_max Maximum number of operations allowed.
 * @param mat Matrix that will be decomposed.
 * @param p_ng Pointer that will receive the number of Given's rotations required.
 * @param p_givens Pointer that will receive an array (pointer) of p_ng Given's rotations that were performed
 * @param allocator Allocator callbacks that will be used.
 * @return Zero if successful.
 */
int decompose_qr(const int64_t n_max, const lil_matrix_t *mat, uint64_t *p_ng, givens_rotation_t **const p_givens,
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
    if (sparse_vector_new(&out_i, mat->cols, 1, allocator) || sparse_vector_new(&out_j, mat->cols, 1, allocator))
    {
        deallocate(allocator, givens);
        sparse_vec_del(&out_i, allocator);
        sparse_vec_del(&out_j, allocator);
        return -1;
    }

    for (uint64_t j_row = 1; j_row < mat->rows && (int64_t)n_givens < n_max; ++j_row)
    {
        // Row that will be eliminated
        const svector_t *const row_j = mat->row_data + j_row;
        uint64_t i_row;
        while (row_j->count > 0 && (i_row = row_j->entries[0].index) < j_row && (int64_t)n_givens < n_max)
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
            const scalar_t c = row_i->entries[0].value;
            const scalar_t s = row_j->entries[0].value;
            const scalar_t a = atan2(s, c);
            givens_rotation_t g = {.n = mat->rows, .k = i_row, .l = j_row, .c = cos(a), .s = sin(a)};
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
