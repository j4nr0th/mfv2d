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

static argument_status_t validate_arg_specs(const unsigned n, const argument_t specs[const static n])
{
    // Validate input specs have all keyword args at the end
    for (unsigned i = 1; i < n; ++i)
    {
        if (specs[i - 1].kwname != NULL && specs[i].kwname == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Argument %u has a keyword argument, but argument %u does not (author is a retard).", i - 1,
                         i);
            return ARG_STATUS_BAD_SPECS;
        }
    }
    for (unsigned i = 0; i < n; ++i)
    {
        if (specs[i].kwname == NULL && specs[i].kw_only)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Argument %u was marked as keyword-only, but does not specify a keyword (author is a retard).",
                         i);
            return ARG_STATUS_BAD_SPECS;
        }
        if (specs[i].type_check != NULL && specs[i].type != ARG_TYPE_PYTHON)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Argument %u specifies a type to check, but does not specify the type as Python object "
                         "(author is a retard).",
                         i);
            return ARG_STATUS_BAD_SPECS;
        }
        if (specs[i].kwname != NULL)
        {
            for (unsigned j = i + 1; j < n; ++j)
            {
                if (specs[j].kwname != NULL && strcmp(specs[j].kwname, specs[i].kwname) == 0)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Arguments %u and %u use the same keyword \"%s\" (author is a retard).", i, j);
                    return ARG_STATUS_BAD_SPECS;
                }
            }
            if (specs[i].kwname[0] == '\0')
            {
                PyErr_Format(PyExc_RuntimeError, "Argument %u specifies a keyword with no length %u.", i);
                return ARG_STATUS_BAD_SPECS;
            }
        }
        switch (specs[i].type)
        {
        case ARG_TYPE_PYTHON:
        case ARG_TYPE_BOOL:
        case ARG_TYPE_INT:
        case ARG_TYPE_DOUBLE:
        case ARG_TYPE_STRING:
        case ARG_TYPE_NONE:
            break;
        default:
            PyErr_Format(PyExc_RuntimeError, "Argument %u has invalid type %u.", i, specs[i].type);
            return ARG_STATUS_BAD_SPECS;
        }
    }

    return ARG_STATUS_SUCCESS;
}

argument_status_t extract_argument_value(const unsigned i, PyObject *const val, argument_t *const arg)
{
    switch (arg->type)
    {
    case ARG_TYPE_PYTHON:
        if (arg->type_check && !PyObject_TypeCheck(val, arg->type_check))
        {
            PyErr_Format(PyExc_TypeError, "Argument %u is not of type %s but instead %R.", i, arg->type_check->tp_name,
                         Py_TYPE(val));
            return ARG_STATUS_INVALID;
        }
        arg->value_python = val;
        break;

    case ARG_TYPE_BOOL:
        arg->value_bool = PyObject_IsTrue(val);
        if (PyErr_Occurred())
            return ARG_STATUS_INVALID;
        break;

    case ARG_TYPE_INT:
        arg->value_int = PyLong_AsSsize_t(val);
        if (PyErr_Occurred())
            return ARG_STATUS_INVALID;
        break;

    case ARG_TYPE_DOUBLE:
        arg->value_double = PyFloat_AsDouble(val);
        if (PyErr_Occurred())
            return ARG_STATUS_INVALID;
        break;

    case ARG_TYPE_STRING:
        arg->value_string = PyUnicode_AsUTF8(val);
        if (PyErr_Occurred())
            return ARG_STATUS_INVALID;
        break;

    case ARG_TYPE_NONE:
        ASSERT(0, "Should not be reached.");
        return ARG_STATUS_BAD_SPECS;
    }
    arg->found = 1;
    return ARG_STATUS_SUCCESS;
}

MFV2D_INTERNAL
argument_status_t parse_arguments(argument_t specs[const], PyObject *const args[const], const Py_ssize_t nargs,
                                  const PyObject *const kwnames)
{
    ASSERT(args != NULL, "Pointer to positional args should not be null.");
    const unsigned nkwds = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    ASSERT(specs != NULL, "Pointer to argument specs should not be null.");
    ASSERT(nargs > 0, "Number of arguments must be a positive integer (it was %lld).", (long long int)nargs);

    unsigned n = 0;
    while (specs[n].type != ARG_TYPE_NONE)
    {
        specs[n].found = 0;
        n += 1;
    }
    if (n == 0)
    {
        // No args? Not my problem!
        return ARG_STATUS_SUCCESS;
    }

    // Validate the arguments are properly specified.
    ASSERT(validate_arg_specs(n, specs) == ARG_STATUS_SUCCESS, "Invalid argument specs.");
    ASSERT(nargs + nkwds <= n, "Number of specified arguments is less than the number of received arguments.");

    for (unsigned i = 0; i < nargs; ++i)
    {
        PyObject *const val = args[i];
        argument_t *const arg = specs + i;
        if (arg->kw_only)
        {
            PyErr_Format(PyExc_RuntimeError, "Argument %u (%s) is keyword-only, but was passed as positional argument.",
                         i, arg->kwname);
            return ARG_STATUS_KW_AS_POS;
        }

        const argument_status_t res = extract_argument_value(i, val, arg);
        if (res != ARG_STATUS_SUCCESS)
            return res;
    }

    unsigned first_kw = 0;
    while (first_kw < n && specs[first_kw].kwname == NULL)
    {
        first_kw += 1;
    }

    for (unsigned i = 0; i < nkwds; ++i)
    {
        PyObject *const val = args[nargs + i];
        PyObject *const kwname = PyTuple_GET_ITEM(kwnames, i);

        const char *kwd = PyUnicode_AsUTF8(kwname);
        if (!kwd)
            return ARG_STATUS_INVALID;

        unsigned i_arg;
        for (i_arg = first_kw; i_arg < n; ++i_arg)
        {
            if (strcmp(kwd, specs[i_arg].kwname) == 0)
                break;
        }

        if (i_arg == n)
        {
            PyErr_Format(PyExc_TypeError, "Function does not have any parameter names \"%s\".", kwd);
            return ARG_STATUS_NO_KW;
        }

        argument_t *const arg = specs + i_arg;

        if (arg->found)
        {
            PyErr_Format(PyExc_TypeError, "Parameter \"%s\" was already specified.", kwd);
            return ARG_STATUS_DUPLICATE;
        }

        const argument_status_t res = extract_argument_value(i, val, arg);
        if (res != ARG_STATUS_SUCCESS)
            return res;
    }

    for (unsigned i = 0; i < n; ++i)
    {
        const argument_t *const arg = specs + i;
        if (arg->found == 0 && arg->optional == 0)
        {
            PyErr_Format(PyExc_TypeError, "Non-optional parameter \"%s\" was not specified.", arg->kwname);
            return ARG_STATUS_MISSING;
        }
    }

    return ARG_STATUS_SUCCESS;
}

const char *arg_status_strings[] = {
    [ARG_STATUS_SUCCESS] = "Parsed correctly",
    [ARG_STATUS_MISSING] = "Argument was missing",
    [ARG_STATUS_INVALID] = "Argument had invalid value",
    [ARG_STATUS_DUPLICATE] = "Argument was found twice",
    [ARG_STATUS_BAD_SPECS] = "Specifications were incorrect",
    [ARG_STATUS_KW_AS_POS] = "Keyword argument was specified as a positional argument",
    [ARG_STATUS_NO_KW] = "No argument has this keyword",
    [ARG_STATUS_UNKNOWN] = "Unknown error",
};

const char *argument_status_str(const argument_status_t e)
{
    if ((size_t)e >= (sizeof(arg_status_strings) / sizeof(arg_status_strings[0])))
        return "UNKNOWN";
    return arg_status_strings[e];
}
