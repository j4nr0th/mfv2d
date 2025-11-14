//
// Created by jan on 19.3.2025.
//

#ifndef SVECTOR_H
#define SVECTOR_H

#include "../common/common.h"
#include "../common/error.h"

typedef double scalar_t;
typedef struct
{
    uint64_t index;
    scalar_t value;
} entry_t;

typedef struct
{
    uint64_t n, count, capacity;
    entry_t *restrict entries;
} svector_t;

typedef struct
{
    PyObject_VAR_HEAD;
    uint64_t n, count;
    entry_t entries[];
} svec_object_t;

MFV2D_INTERNAL
// extern PyTypeObject svec_type_object;
extern PyType_Spec svec_type_spec;

/**
 * Create a new sparse vector with no entries and desired capacity.
 *
 * @param this Memory where the resulting vector is initialized.
 * @param n Dimension of the vector.
 * @param capacity Desired capacity of the vector.
 * @param allocator Allocator used to get the memory for the vector.
 * @return Zero on success.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_vector_new(svector_t *this, uint64_t n, uint64_t capacity, const allocator_callbacks *allocator);

/**
 * Clean up the memory used by the vector and clear the memory where it was stored.
 *
 * @param this Memory where the vector is stored.
 * @param allocator Allocator to release the memory with.
 */
MFV2D_INTERNAL
void sparse_vector_del(svector_t *this, const allocator_callbacks *allocator);

/**
 * Increase the size of the vector if too small.
 *
 * @param this Memory where the vector is stored.
 * @param new_capacity New required capacity.
 * @param allocator Allocator that can be used to reallocate the buffers as needed.
 * @return Zero on success.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_vec_resize(svector_t *this, uint64_t new_capacity, const allocator_callbacks *allocator);

/**
 * Append an entry to a sparse vector, resizing if needed.
 *
 * @param this Vector to which to append to.
 * @param e Entry that is to be appended
 * @param allocator Allocator to use for reallocating memory if needed.
 * @return Zero on success.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_vector_append(svector_t *this, entry_t e, const allocator_callbacks *allocator);

/**
 * Create a Python sparse vector object based on the given sparse vector.
 *
 * @param svec_type_object Type object to use to create the Python type.
 * @param this Vector that is to be converted.
 * @return Pointer to the object or NULL on allocation failure.
 */
MFV2D_INTERNAL
svec_object_t *sparse_vector_to_python(PyTypeObject *svec_type_object, const svector_t *this);

/**
 * Performs a deep copy of a vector to another memory location.
 *
 * @param src Vector to copy from.
 * @param dst Vector to copy to.
 * @param allocator Allocator to use to potentially allocate the memory with.
 * @return Zero on success.
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_vector_copy(const svector_t *src, svector_t *dst, const allocator_callbacks *allocator);

/**
 * Search the vector for the first entry with the given index, or one which is greater than it. The length of the vector
 * must be at least 1.
 *
 * @param this Vector which to search.
 * @param v Index of the entry to find.
 * @param start Index where to begin the search. Useful if continuing from a previous search.
 * @return Index of the first entry with an index equal to or greater than the specified value. If none are found,
 * this->count is returned.
 */
MFV2D_INTERNAL
uint64_t sparse_vector_find_first_geq(const svector_t *this, uint64_t v, uint64_t start);

/**
 * Add the contents of one sparse vector to another sparse vector in place.
 * The operation overwrites the entries of the first vector with the result.
 * Both input vectors must have matching dimensions, and their capacity may
 * be resized if necessary. If there are duplicate indices in the two vectors,
 * the values from the second vector are added to the values in the first vector.
 *
 * @param first Sparse vector to be modified by the addition operation. Must
 *              have sufficient capacity to hold the combined unique indices,
 *              or it will be resized.
 * @param second Sparse vector whose values will be added to the first vector.
 *               Its content will not be modified.
 * @param allocator Allocator used to manage memory during operations such as
 *                  resizing the first vector.
 * @return Zero on successful addition or a nonzero value if an error occurs
 *         (e.g., memory allocation failure during resizing).
 */
MFV2D_INTERNAL
mfv2d_result_t sparse_vector_add_inplace(svector_t *first, const svector_t *second,
                                         const allocator_callbacks *allocator);

#endif // SVECTOR_H
