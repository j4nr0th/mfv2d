#ifndef COMMON_H
#define COMMON_H
#include "common_defines.h"

typedef struct
{
    void *(*alloc)(void *state, size_t size);
    void *(*realloc)(void *state, void *ptr, size_t new_size);
    void (*free)(void *state, void *ptr);
    void *state;
} allocator_callbacks;

MFV2D_INTERNAL
extern allocator_callbacks SYSTEM_ALLOCATOR;

MFV2D_INTERNAL
extern allocator_callbacks PYTHON_ALLOCATOR;

MFV2D_INTERNAL
extern allocator_callbacks OBJECT_ALLOCATOR;

static inline void *allocate(const allocator_callbacks *allocator, const size_t sz)
{
    return allocator->alloc(allocator->state, sz);
}

static inline void *reallocate(const allocator_callbacks *allocator, void *ptr, const size_t new_sz)
{
    return allocator->realloc(allocator->state, ptr, new_sz);
}

static inline void deallocate(const allocator_callbacks *allocator, void *ptr)
{
    return allocator->free(allocator->state, ptr);
}

#endif // COMMON_H
