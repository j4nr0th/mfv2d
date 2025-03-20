//
// Created by jan on 19.3.2025.
//

#ifndef SVECTOR_H
#define SVECTOR_H

#include "../../common.h"

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
    PyObject_HEAD;
    uint64_t n, count, capacity;
    entry_t entries[];
} svec_object_t;

INTERPLIB_INTERNAL
extern PyTypeObject svec_type_object;

/**
 * Create a new sparse vector with no entries and desired capacity.
 *
 * @param this Memory where the resulting vector is initialized.
 * @param n Dimension of the vector.
 * @param capacity Desired capacity of the vector.
 * @param allocator Allocator used to get the memory for the vector.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int sparse_vector_new(svector_t *this, uint64_t n, uint64_t capacity, const allocator_callbacks *allocator);

/**
 * Clean up the memory used by the vector and clear the memory where it was stored.
 *
 * @param this Memory where the vector is stored.
 * @param allocator Allocator to release the memory with.
 */
INTERPLIB_INTERNAL
void sparse_vec_del(svector_t *this, const allocator_callbacks *allocator);

/**
 * Increase the size of the vector if too small.
 *
 * @param this Memory where the vector is stored.
 * @param capacity New required capacity.
 * @param allocator Allocator that can be used to reallocate the buffers as needed.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int sparse_vec_resize(svector_t *this, uint64_t capacity, const allocator_callbacks *allocator);

/**
 * Create a Python sparse vector object based on the given sparse vector.
 *
 * @param this Vector that is to be converted.
 * @return Pointer to the object or NULL on allocation failure.
 */
INTERPLIB_INTERNAL
svec_object_t *sparse_vec_to_python(const svector_t *this);

/**
 * Performs a deep copy of a vector to another memory location.
 *
 * @param src Vector to copy from.
 * @param dst Vector to copy to.
 * @param allocator Allocator to use to potentially allocate the memory with.
 * @return Zero on success.
 */
INTERPLIB_INTERNAL
int sparse_vector_copy(const svector_t *src, svector_t *dst, const allocator_callbacks *allocator);

#endif // SVECTOR_H
