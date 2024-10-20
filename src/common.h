//
// Created by jan on 19.10.2024.
//

#ifndef COMMON_H
#define COMMON_H
#include "common_defines.h"
#include "jmtx/matrix_base.h"


INTERPLIB_INTERNAL
extern jmtx_allocator_callbacks SYSTEM_ALLOCATOR;

INTERPLIB_INTERNAL
extern jmtx_allocator_callbacks PYTHON_ALLOCATOR;

static inline void* allocate(const jmtx_allocator_callbacks* allocator, const size_t sz)
{
    return allocator->alloc(allocator->state, sz);
}

static inline void deallocate(const jmtx_allocator_callbacks* allocator, void* ptr)
{
    return allocator->free(allocator->state, ptr);
}

#endif //COMMON_H
