#ifndef COMMON_H
#define COMMON_H
#include "common_defines.h"
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>
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

/**
 * @brief Validates a NumPy array based on the specified dimensions, data type, and flags.
 *
 * This function checks several conditions for the given array, including
 * - Whether the array has the required flags.
 * - Whether the number of dimensions matches the expected value.
 * - Whether the data type matches the expected type (if specified).
 * - Whether each dimension size matches the expected size (if specified).
 *
 * If any of the conditions fail, a Python exception is raised with a descriptive error message,
 * and the function returns -1. Otherwise, the function returns 0 on success.
 *
 * @param arr Pointer to the NumPy array object to be validated.
 * @param n_dim The expected number of dimensions for the array.
 * @param dims Array of expected sizes for each dimension. Use 0 for dimensions that do not require strict matching.
 * @param dtype The expected data type of the array (e.g., NPY_DOUBLE). Use a negative value to skip this check.
 * @param flags The required flags that must be present in the array (e.g., NPY_ARRAY_C_CONTIGUOUS).
 * @param name The name of the array (for error messages).
 * @return Returns 0 if the array passes all validation checks, or -1 if any check fails.
 *         In the event of failure, a Python exception is set with an appropriate error message.
 */

MFV2D_INTERNAL int check_input_array(const PyArrayObject *arr, unsigned n_dim, const npy_intp dims[static n_dim],
                                     int dtype, int flags, const char *name);

MFV2D_INTERNAL void check_memory_bounds(size_t allocated_size, size_t element_count, size_t element_size,
                                        const char *file, int line, const char *func);

/**
 * @brief Raises a new Python exception, preserving the current exception context.
 *
 * This function raises a new Python exception with the specified type while
 * preserving the context of the current exception if one exists. It formats
 * the error message using the provided format string and additional arguments.
 * If an exception is already set, it will be attached as the cause of the new
 * exception. If no exception is set, a new exception is generated with the
 * specified type and formatted message.
 *
 * @param exception The Python exception type to raise (e.g., PyExc_RuntimeError).
 * @param format The printf-style format string to create the exception message.
 * @param ... Additional arguments to populate the format string.
 */
[[gnu::format(printf, 2, 3)]]
MFV2D_INTERNAL void raise_exception_from_current(PyObject *exception, const char *format, ...);

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
} mfv2d_module_state_t;

#endif // COMMON_H
