//
// Created by jan on 19.10.2024.
//

#include "common.h"

#include <Python.h>

//  Magic numbers meant for checking with allocators that don't need to store
//  state.
enum
{
    SYSTEM_MAGIC = 0xBadBeef,
    PYTHON_MAGIC = 0x600dBeef,
};

static void *allocate_system(void *state, size_t size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawMalloc(size);
}

static void *reallocate_system(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawRealloc(ptr, new_size);
}

static void free_system(void *state, void *ptr)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_RawFree(ptr);
}

MFV2D_INTERNAL
allocator_callbacks SYSTEM_ALLOCATOR = {
    .alloc = allocate_system,
    .free = free_system,
    .realloc = reallocate_system,
    .state = (void *)SYSTEM_MAGIC,
};

static void *allocate_python(void *state, size_t size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Malloc(size);
}

static void *reallocate_python(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Realloc(ptr, new_size);
}

static void free_python(void *state, void *ptr)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_Free(ptr);
}

MFV2D_INTERNAL
allocator_callbacks PYTHON_ALLOCATOR = {
    .alloc = allocate_python,
    .free = free_python,
    .realloc = reallocate_python,
    .state = (void *)PYTHON_MAGIC,
};

void check_memory_bounds(const size_t allocated_size, const size_t element_count, const size_t element_size,
                         const char *file, int line, const char *func)
{
#ifdef MFV2D_ASSERTS
    if (element_count * element_size >= allocated_size)
    {
        fprintf(stderr,
                "Memory bounds violation in %s:%d in function %s (Allocated %zu bytes, accessed %zu bytes at offset "
                "%zu).\n",
                file, line, func, allocated_size, element_size, element_count * element_size);
        exit(EXIT_FAILURE);
    }
#endif // MFV2D_ASSERTS
    (void)allocated_size;
    (void)element_count;
    (void)element_size;
    (void)file;
    (void)line;
    (void)func;
}
int traverse_heap_type(PyObject *op, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(op));
    return 0;
}
