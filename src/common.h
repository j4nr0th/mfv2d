//
// Created by jan on 19.10.2024.
//

#ifndef COMMON_H
#define COMMON_H
#include "common_defines.h"
#include <Python.h>

typedef struct
{
    void* (*alloc)(void* state, size_t size);
    void* (*realloc)(void* state, void* ptr, size_t new_size);
    void (*free)(void* state, void* ptr);
    void* state;
} allocator_callbacks;

INTERPLIB_INTERNAL
extern allocator_callbacks SYSTEM_ALLOCATOR;

INTERPLIB_INTERNAL
extern allocator_callbacks PYTHON_ALLOCATOR;

INTERPLIB_INTERNAL
extern allocator_callbacks OBJECT_ALLOCATOR;

static inline void* allocate(const allocator_callbacks* allocator, const size_t sz)
{
    return allocator->alloc(allocator->state, sz);
}

static inline void deallocate(const allocator_callbacks* allocator, void* ptr)
{
    return allocator->free(allocator->state, ptr);
}

typedef struct
{
  PyObject* basis1d_type;
  PyObject* poly1d_type;
  PyObject* spline1d_type;
} interplib_python_api;

INTERPLIB_INTERNAL
extern interplib_python_api INTERPLIB_PYTHON_API;

#endif //COMMON_H
