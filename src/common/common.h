#ifndef COMMON_H
#define COMMON_H
#include "common_defines.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

// This must follow the numpy includes, otherwise it will shit itself
#include "../../cpyutl/src/cpyutl.h"

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

static inline void *allocate_track(const allocator_callbacks *allocator, const size_t sz, const char *file, int line,
                                   const char *func)
{
    void *const ptr = allocate(allocator, sz);
    fprintf(stderr, "Allocating %p for %zu bytes at %s:%d (%s)\n", ptr, sz, file, line, func);
    return ptr;
}

static inline void *deallocate_track(const allocator_callbacks *allocator, void *ptr, const char *file, int line,
                                     const char *func)
{
    deallocate(allocator, ptr);
    fprintf(stderr, "Deallocating %p at %s:%d (%s)\n", ptr, file, line, func);
    return ptr;
}

// #define allocate(allocator, sz) allocate_track((allocator), (sz), __FILE__, __LINE__, __func__)
// #define deallocate(allocator, sz) deallocate_track((allocator), (sz), __FILE__, __LINE__, __func__)

MFV2D_INTERNAL void check_memory_bounds(size_t allocated_size, size_t element_count, size_t element_size,
                                        const char *file, int line, const char *func);

#define CHECK_MEMORY_BOUNDS(allocated_size, offset, size)                                                              \
    check_memory_bounds((allocated_size), (offset), (size), __FILE__, __LINE__, __func__)

typedef struct
{
    unsigned i, j;
} index_2d_t;

typedef struct
{
    double x0, y0;
    double x1, y1;
    double x2, y2;
    double x3, y3;
} quad_info_t;

typedef struct
{
    // Types
    PyTypeObject *type_geoid;
    PyTypeObject *type_line;
    PyTypeObject *type_surface;
    PyTypeObject *type_man2d;
    PyTypeObject *type_mesh;
    PyTypeObject *type_int_rule;
    PyTypeObject *type_basis1d;
    PyTypeObject *type_basis2d;
    PyTypeObject *type_fem_space;
    PyTypeObject *type_form_spec;
    PyTypeObject *type_form_spec_iter;
    PyTypeObject *type_svec;
    PyTypeObject *type_crs_matrix;
    PyTypeObject *type_system;
    PyTypeObject *type_trace_vector;
    PyTypeObject *type_dense_vector;
    PyTypeObject *type_gll_cache;

    // Caches
    PyObject *cache_gll;
} mfv2d_module_state_t;

/**
 * @brief Retrieves the module state associated with a given Python type.
 *
 * This function extracts the module state associated with a Python type object.
 * It first retrieves the module corresponding to the provided type and then
 * accesses the module state. If the module cannot be resolved or the state is
 * unavailable, the function returns `NULL` and raises a Python TypeError.
 *
 *
 * @param type Pointer to the Python type object for which the module state is to be retrieved.
 * @return A pointer to the module state structure (`mfv2d_module_state_t`) if successful,
 *         or `NULL` if the module or its state could not be accessed.
 */
MFV2D_INTERNAL
const mfv2d_module_state_t *mfv2d_state_from_type(PyTypeObject *type);

MFV2D_INTERNAL
int traverse_heap_type(PyObject *op, visitproc visit, void *arg);

#endif // COMMON_H
