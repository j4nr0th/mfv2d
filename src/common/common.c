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

MFV2D_INTERNAL
int check_input_array(const PyArrayObject *const arr, const unsigned n_dim, const npy_intp dims[static n_dim],
                      const int dtype, const int flags, const char *name)
{
    if (!PyArray_Check(arr))
    {
        PyErr_Format(PyExc_TypeError, "Object %s is not a numpy array, but is %R.", name, Py_TYPE(arr));
        return -1;
    }
    if (name == NULL)
    {
        name = "UNKNOWN";
    }
    const int arr_flags = PyArray_FLAGS(arr);
    if ((arr_flags & flags) != flags)
    {
        PyErr_Format(PyExc_ValueError, "Array %s flags %u don't contain required flags %u.", name, arr_flags, flags);
        return -1;
    }

    if (n_dim && PyArray_NDIM(arr) != (npy_intp)n_dim)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of dimensions for the array %s does not match expected value (expected %u, got %u).", name,
                     n_dim, (unsigned)PyArray_NDIM(arr));
        return -1;
    }

    if (dtype >= 0 && PyArray_TYPE(arr) != dtype)
    {
        PyErr_Format(PyExc_ValueError, "Array %s does not have the expected type (expected %u, got %u).", name, dtype,
                     PyArray_TYPE(arr));
        return -1;
    }

    const npy_intp *const d = PyArray_DIMS(arr);
    for (unsigned i_dim = 0; i_dim < n_dim; ++i_dim)
    {
        if (dims[i_dim] != 0 && d[i_dim] != dims[i_dim])
        {
            PyErr_Format(PyExc_ValueError,
                         "Array %s dimension %u of the did not match expected value (expected %u, got %u).", name,
                         i_dim, dims[i_dim], d[i_dim]);
            return -1;
        }
    }

    return 0;
}
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

[[gnu::format(printf, 2, 3)]]
void raise_exception_from_current(PyObject *exception, const char *format, ...)
{
    PyObject *const original = PyErr_GetRaisedException();
    if (original)
    {
        va_list args;
        va_start(args, format);
        PyObject *const message = PyUnicode_FromFormatV(format, args);
        va_end(args);
        PyObject *new_exception = NULL;
        if (message)
        {
            new_exception = PyObject_CallFunctionObjArgs(exception, message, NULL);
            Py_DECREF(message);
        }

        if (new_exception && PyObject_SetAttrString(new_exception, "__cause__", original) == 0)
        {
            PyErr_SetObject(exception, new_exception);
            new_exception = NULL;
        }
        else
        {
            PyErr_SetRaisedException(original);
        }
        Py_XDECREF(new_exception);
    }
    else
    {
        va_list args;
        va_start(args, format);
        PyErr_FormatV(exception, format, args);
        va_end(args);
    }
}
