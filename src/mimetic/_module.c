//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#include "../common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Internal headers
#include "geoidobject.h"
#include "lineobject.h"
#include "surfaceobject.h"

// Manifolds
#include "manifold.h"
#include "manifold1d.h"
#include "manifold2d.h"

// Evaluation
#include "eval/allocator.h"
#include "eval/connectivity.h"
#include "eval/evaluation.h"
#include "eval/incidence.h"
#include "eval/precomp.h"

// Solver
#include "solve/givens.h"
#include "solve/lil_matrix.h"
#include "solve/svector.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static void caches_array_destroy(unsigned n, basis_precomp_t array[static n])
{
    for (unsigned i = n; i > 0; --i)
    {
        basis_precomp_destroy(array + (i - 1));
    }
}

// static PyObject *compute_element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
// {
//     PyObject *ret_val = NULL;
//     PyObject *in_form_orders;
//     PyObject *in_expressions;
//     PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
//     PyObject *element_orders;
//     PyObject *cache_contents;
//     Py_ssize_t thread_stack_size = (1 << 24);
//     if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO|n",
//                                      (char *[10]){"form_orders", "expressions", "pos_bl", "pos_br", "pos_tr",
//                                      "pos_tl",
//                                                   "element_orders", "cache_contents", "thread_stack_size", NULL},
//                                      &in_form_orders, &in_expressions, &pos_bl, &pos_br, &pos_tr, &pos_tl,
//                                      &element_orders, &cache_contents, &thread_stack_size))
//     {
//         return NULL;
//     }
//     if (thread_stack_size < 0)
//     {
//         PyErr_Format(PyExc_ValueError, "Thread stack size can not be negative (%lld).",
//                      (long long int)thread_stack_size);
//         return NULL;
//     }
//     if ((thread_stack_size & 7) != 0)
//     {
//         thread_stack_size += 8 - (thread_stack_size & 7);
//     }
//
//     // Create the system template
//     system_template_t system_template;
//     if (!system_template_create(&system_template, in_form_orders, in_expressions, &SYSTEM_ALLOCATOR))
//         return NULL;
//
//     // Create caches
//     PyObject *cache_seq = PySequence_Fast(cache_contents, "BasisCaches must be a sequence of tuples.");
//     if (!cache_seq)
//     {
//         goto after_template;
//     }
//     const unsigned n_cache = PySequence_Fast_GET_SIZE(cache_seq);
//
//     basis_precomp_t *cache_array = allocate(&SYSTEM_ALLOCATOR, sizeof *cache_array * n_cache);
//     if (!cache_array)
//     {
//         Py_DECREF(cache_seq);
//         goto after_template;
//     }
//
//     for (unsigned i = 0; i < n_cache; ++i)
//     {
//         int failed = !basis_precomp_create(PySequence_Fast_GET_ITEM(cache_seq, i), cache_array + i);
//         if (!failed)
//         {
//             for (unsigned j = 0; j < i; ++j)
//             {
//                 if (cache_array[i].order == cache_array[j].order)
//                 {
//                     PyErr_Format(PyExc_ValueError,
//                                  "Cache contains the values for order as entries with indices %u and %u.", i, j);
//                     failed = 1;
//                     break;
//                 }
//             }
//         }
//
//         if (failed)
//         {
//             caches_array_destroy(i, cache_array);
//             deallocate(&SYSTEM_ALLOCATOR, cache_array);
//             Py_DECREF(cache_seq);
//             goto after_template;
//         }
//     }
//     Py_DECREF(cache_seq);
//
//     // Convert coordinate arrays
//     PyArrayObject *const bl_array = (PyArrayObject *)PyArray_FromAny(pos_bl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
//                                                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
//                                                                      NULL);
//     PyArrayObject *const br_array = (PyArrayObject *)PyArray_FromAny(pos_br, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
//                                                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
//                                                                      NULL);
//     PyArrayObject *const tr_array = (PyArrayObject *)PyArray_FromAny(pos_tr, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
//                                                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
//                                                                      NULL);
//     PyArrayObject *const tl_array = (PyArrayObject *)PyArray_FromAny(pos_tl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
//                                                                      NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
//                                                                      NULL);
//     PyArrayObject *const orders_array = (PyArrayObject *)PyArray_FromAny(
//         element_orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
//     if (!bl_array || !br_array || !tr_array || !tl_array || !orders_array)
//     {
//         Py_XDECREF(orders_array);
//         Py_XDECREF(tl_array);
//         Py_XDECREF(tr_array);
//         Py_XDECREF(br_array);
//         Py_XDECREF(bl_array);
//         goto after_cache;
//     }
//     size_t n_elements;
//     {
//         n_elements = PyArray_DIM(orders_array, 0);
//         const npy_intp *dims_bl = PyArray_DIMS(bl_array);
//         const npy_intp *dims_br = PyArray_DIMS(br_array);
//         const npy_intp *dims_tr = PyArray_DIMS(tr_array);
//         const npy_intp *dims_tl = PyArray_DIMS(tl_array);
//         if (dims_bl[0] != n_elements || (dims_bl[0] != dims_br[0] || dims_bl[1] != dims_br[1]) ||
//             (dims_bl[0] != dims_tr[0] || dims_bl[1] != dims_tr[1]) ||
//             (dims_bl[0] != dims_tl[0] || dims_bl[1] != dims_tl[1]) || dims_bl[1] != 2)
//         {
//             PyErr_SetString(PyExc_ValueError,
//                             "All coordinate input arrays must be have same number of 2 component vectors.");
//             goto after_arrays;
//         }
//     }
//
//     // Extract C pointers
//
//     const double *restrict const coord_bl = PyArray_DATA(bl_array);
//     const double *restrict const coord_br = PyArray_DATA(br_array);
//     const double *restrict const coord_tr = PyArray_DATA(tr_array);
//     const double *restrict const coord_tl = PyArray_DATA(tl_array);
//     const unsigned *restrict const orders = PyArray_DATA(orders_array);
//
//     // Prepare output arrays
//     double **p_out = allocate(&SYSTEM_ALLOCATOR, sizeof(*p_out) * n_elements);
//     if (!p_out)
//     {
//         goto after_arrays;
//     }
//     ret_val = PyTuple_New((Py_ssize_t)n_elements);
//     if (!ret_val)
//     {
//         deallocate(&SYSTEM_ALLOCATOR, p_out);
//         goto after_arrays;
//     }
//
//     // Create an error stack for reporting issues
//
//     for (unsigned i = 0; i < n_elements; ++i)
//     {
//         size_t element_size = 0;
//         for (unsigned j = 0; j < system_template.n_forms; ++j)
//         {
//             element_size += form_degrees_of_freedom_count(system_template.form_orders[j], orders[i]);
//         }
//         const npy_intp dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
//         PyArrayObject *const a = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
//         if (!a)
//         {
//             Py_DECREF(ret_val);
//             ret_val = NULL;
//             deallocate(&SYSTEM_ALLOCATOR, p_out);
//             goto after_arrays;
//         }
//         PyTuple_SET_ITEM(ret_val, i, a);
//         p_out[i] = PyArray_DATA(a);
//         memset(p_out[i], 0, sizeof(*p_out[i]) * dims[0] * dims[1]);
//     }
//
//     eval_result_t common_res = EVAL_SUCCESS;
//     Py_BEGIN_ALLOW_THREADS
//
// #pragma omp parallel default(none) \
//     shared(SYSTEM_ALLOCATOR, system_template, stderr, common_res, n_elements, orders, cache_array, n_cache, coord_bl,
//     \
//                coord_br, coord_tr, coord_tl, p_out, thread_stack_size)
//     {
//         error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
//         // Allocate the stack through system allocator
//         matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
//         allocator_stack_t *const allocator_stack = allocator_stack_create(thread_stack_size, &SYSTEM_ALLOCATOR);
//         eval_result_t res = matrix_stack && err_stack ? EVAL_SUCCESS : EVAL_FAILED_ALLOC;
//         /* Heavy calculations here */
// #pragma omp for nowait
//         for (unsigned i_elem = 0; i_elem < n_elements; ++i_elem)
//         {
//             if (!(common_res == EVAL_SUCCESS && err_stack && matrix_stack && allocator_stack))
//             {
//                 continue;
//             }
//             allocator_stack_reset(allocator_stack);
//             const unsigned order = orders[i_elem];
//             size_t element_size = 0;
//             for (unsigned j = 0; j < system_template.n_forms; ++j)
//             {
//                 element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order);
//             }
//             precompute_t precomp;
//             unsigned i;
//             // Find cached values for the current order
//             for (i = 0; i < n_cache; ++i)
//             {
//                 if (order == cache_array[i].order)
//                 {
//                     break;
//                 }
//             }
//             if (i == n_cache)
//             {
//                 // Failed, not in cache!
//                 continue;
//             }
//             // Compute matrices for the element
//             if (!precompute_create(cache_array + i, coord_bl[2 * i_elem + 0], coord_br[2 * i_elem + 0],
//                                    coord_tr[2 * i_elem + 0], coord_tl[2 * i_elem + 0], coord_bl[2 * i_elem + 1],
//                                    coord_br[2 * i_elem + 1], coord_tr[2 * i_elem + 1], coord_tl[2 * i_elem + 1],
//                                    &precomp, &allocator_stack->base))
//             {
//                 // Failed, could not compute precomp
//                 continue;
//             }
//
//             double *restrict const output_mat = p_out[i_elem];
//
//             // Compute the individual entries
//             size_t row_offset = 0;
//             for (unsigned row = 0; row < system_template.n_forms && res == EVAL_SUCCESS; ++row)
//             {
//                 const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order);
//                 size_t col_offset = 0;
//                 for (unsigned col = 0; col < system_template.n_forms /*&& res == EVAL_SUCCESS*/; ++col)
//                 {
//                     const unsigned col_len = form_degrees_of_freedom_count(system_template.form_orders[col], order);
//                     const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
//                     if (!bytecode)
//                     {
//                         // Zero entry, we do nothing since arrays start zeroed out (I think).
//                         col_offset += col_len;
//                         continue;
//                     }
//                     matrix_full_t mat;
//                     res = evaluate_element_term(err_stack, system_template.form_orders[row], order, bytecode,
//                     &precomp,
//                                                 system_template.max_stack, matrix_stack, &allocator_stack->base,
//                                                 &mat);
//                     if (res != EVAL_SUCCESS)
//                     {
//                         EVAL_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
//                         break;
//                     }
//                     if (row_len != mat.base.rows || col_len != mat.base.cols)
//                     {
//                         EVAL_ERROR(err_stack, EVAL_DIMS_MISMATCH,
//                                    "Output matrix arrays don't match expected dims (got %u x %u when needed %u x
//                                    %u).", mat.base.rows, mat.base.cols, row_len, col_len);
//                         res = EVAL_DIMS_MISMATCH;
//                         break;
//                     }
//
//                     for (unsigned i_out = 0; i_out < row_len; ++i_out)
//                     {
//                         for (unsigned j_out = 0; j_out < col_len; ++j_out)
//                         {
//                             output_mat[(i_out + row_offset) * element_size + (j_out + col_offset)] =
//                                 mat.data[i_out * mat.base.cols + j_out];
//                         }
//                     }
//
//                     deallocate(&allocator_stack->base, mat.data);
//                     // SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, mat.data);
//                     col_offset += col_len;
//                 }
//                 row_offset += row_len;
//             }
//         }
//
//         deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
//
//         // Clean error stack
//         if (err_stack && err_stack->position != 0)
//         {
//             fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);
//
//             for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
//             {
//                 const error_message_t *msg = err_stack->messages + i_e;
//                 fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
//                         eval_result_str(msg->code), msg->message);
//                 deallocate(err_stack->allocator, msg->message);
//             }
//         }
//         if (err_stack)
//             deallocate(err_stack->allocator, err_stack);
//         if (allocator_stack)
//         {
//             deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
//             deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
//         }
// #pragma omp critical
//         {
//             if (res != EVAL_SUCCESS)
//             {
//                 common_res = res;
//             }
//         }
//     }
//
//     Py_END_ALLOW_THREADS if (common_res != EVAL_SUCCESS)
//     {
//         PyErr_Format(PyExc_ValueError, "Execution failed with error code %s.", eval_result_str(common_res));
//         // Failed allocation of matrix stack.
//         Py_DECREF(p_out);
//         p_out = NULL;
//     }
//
//     // Clean up the array of output pointers
//     deallocate(&SYSTEM_ALLOCATOR, p_out);
//
//     // Clean up the coordinate arrays
// after_arrays:
//     Py_DECREF(orders_array);
//     Py_DECREF(tl_array);
//     Py_DECREF(tr_array);
//     Py_DECREF(br_array);
//     Py_DECREF(bl_array);
//
//     // Clean up the basis caches
// after_cache:
//     caches_array_destroy(n_cache, cache_array);
//     deallocate(&SYSTEM_ALLOCATOR, cache_array);
//
//     // Clean up the template
// after_template:
//     system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
//
//     return ret_val;
// }

static PyObject *compute_element_matrices_2(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *ret_val = NULL;
    PyObject *in_form_orders;
    PyObject *in_expressions;
    PyObject *pos_bl, *pos_br, *pos_tr, *pos_tl;
    PyObject *element_orders;
    PyObject *cache_contents;
    PyTupleObject *vector_field_tuple;
    PyObject *element_offsets;
    Py_ssize_t thread_stack_size = (1 << 24);
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOOOOOO!OO|n",
            (char *[12]){"form_orders", "expressions", "pos_bl", "pos_br", "pos_tr", "pos_tl", "element_orders",
                         "vector_fields", "element_field_offsets", "cache_contents", "thread_stack_size", NULL},
            &in_form_orders, &in_expressions, &pos_bl, &pos_br, &pos_tr, &pos_tl, &element_orders, &PyTuple_Type,
            &vector_field_tuple, &element_offsets, &cache_contents, &thread_stack_size))
    {
        return NULL;
    }

    // Check that the number of vector fields is not too high
    if (PyTuple_GET_SIZE(vector_field_tuple) >= VECTOR_FIELDS_MAX)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Number of vector fields given (%zu) is over maximum supported value (VECTOR_FIELDS_MAX = %u).",
                     (size_t)PyTuple_GET_SIZE(vector_field_tuple), (unsigned)VECTOR_FIELDS_MAX);
        return NULL;
    }

    if (thread_stack_size < 0)
    {
        PyErr_Format(PyExc_ValueError, "Thread stack size can not be negative (%lld).",
                     (long long int)thread_stack_size);
        return NULL;
    }

    // Round the stack size up to the nearest 8.
    if ((thread_stack_size & 7) != 0)
    {
        thread_stack_size += 8 - (thread_stack_size & 7);
    }

    // Create the system template
    system_template_t system_template;
    if (!system_template_create(&system_template, in_form_orders, in_expressions, &SYSTEM_ALLOCATOR))
        return NULL;

    // Create caches
    PyObject *cache_seq = PySequence_Fast(cache_contents, "BasisCaches must be a sequence of tuples.");
    if (!cache_seq)
    {
        goto after_template;
    }
    const unsigned n_cache = PySequence_Fast_GET_SIZE(cache_seq);

    basis_precomp_t *cache_array = allocate(&SYSTEM_ALLOCATOR, sizeof *cache_array * n_cache);
    if (!cache_array)
    {
        Py_DECREF(cache_seq);
        goto after_template;
    }

    for (unsigned i = 0; i < n_cache; ++i)
    {
        int failed = !basis_precomp_create(PySequence_Fast_GET_ITEM(cache_seq, i), cache_array + i);
        if (!failed)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                if (cache_array[i].order == cache_array[j].order)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Cache contains the values for order as entries with indices %u and %u.", i, j);
                    failed = 1;
                    break;
                }
            }
        }

        if (failed)
        {
            caches_array_destroy(i, cache_array);
            deallocate(&SYSTEM_ALLOCATOR, cache_array);
            Py_DECREF(cache_seq);
            goto after_template;
        }
    }
    Py_DECREF(cache_seq);

    // Convert coordinate arrays
    PyArrayObject *const bl_array = (PyArrayObject *)PyArray_FromAny(pos_bl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const br_array = (PyArrayObject *)PyArray_FromAny(pos_br, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tr_array = (PyArrayObject *)PyArray_FromAny(pos_tr, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const tl_array = (PyArrayObject *)PyArray_FromAny(pos_tl, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    PyArrayObject *const orders_array = (PyArrayObject *)PyArray_FromAny(
        element_orders, PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    PyArrayObject *const offset_array = (PyArrayObject *)PyArray_FromAny(
        element_offsets, PyArray_DescrFromType(NPY_UINT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

    if (!bl_array || !br_array || !tr_array || !tl_array || !orders_array || !offset_array)
    {
        Py_XDECREF(offset_array);
        Py_XDECREF(orders_array);
        Py_XDECREF(tl_array);
        Py_XDECREF(tr_array);
        Py_XDECREF(br_array);
        Py_XDECREF(bl_array);
        goto after_cache;
    }
    size_t n_elements, n_field_points;
    {
        n_elements = PyArray_DIM(orders_array, 0);
        const npy_intp *dims_bl = PyArray_DIMS(bl_array);
        const npy_intp *dims_br = PyArray_DIMS(br_array);
        const npy_intp *dims_tr = PyArray_DIMS(tr_array);
        const npy_intp *dims_tl = PyArray_DIMS(tl_array);
        if (dims_bl[0] != n_elements || (dims_bl[0] != dims_br[0] || dims_bl[1] != dims_br[1]) ||
            (dims_bl[0] != dims_tr[0] || dims_bl[1] != dims_tr[1]) ||
            (dims_bl[0] != dims_tl[0] || dims_bl[1] != dims_tl[1]) || dims_bl[1] != 2 ||
            PyArray_SIZE(offset_array) != n_elements + 1)
        {
            PyErr_SetString(PyExc_ValueError, "All coordinate input arrays, orders array, and offset array must be "
                                              "have same number of 2 component vectors.");
            goto after_arrays;
        }
        n_field_points = ((const npy_uint64 *)PyArray_DATA(offset_array))[n_elements];
    }

    const size_t field_count = PyTuple_GET_SIZE(vector_field_tuple);
    field_information_t vector_fields = {.n_fields = field_count, .offsets = PyArray_DATA(offset_array)};
    // Check that the vector field arrays have the correct shape
    for (unsigned i = 0; i < field_count; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(vector_field_tuple, i);
        // Check type
        if (!PyArray_Check(o))
        {
            PyErr_Format(PyExc_ValueError, "Vector field tuple entry %u was not a Numpy array, but %R.", i, Py_TYPE(o));
            goto after_arrays;
        }
        PyArrayObject *const vec_field = (PyArrayObject *)o;

        // Check data type
        if (PyArray_TYPE(vec_field) != NPY_FLOAT64)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had an incorrect data type with dtype %u (should be float64 - %u).", i,
                         PyArray_TYPE(vec_field), NPY_FLOAT64);
            goto after_arrays;
        }

        // Check dim count
        if (PyArray_NDIM(vec_field) != 2)
        {
            PyErr_Format(PyExc_ValueError, "Vector field %u did not have two axis, but had %u instead.", i,
                         (unsigned)PyArray_NDIM(vec_field));
            goto after_arrays;
        }
        const npy_intp *dims = PyArray_DIMS(vec_field);

        // Check dims
        if (dims[0] != n_field_points || dims[1] != 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u had the incorrect shape (%zu, %zu). Based on the offset array it was "
                         "expected to be (%zu, %zu) instead.",
                         i, (size_t)dims[0], (size_t)dims[1], n_field_points, (size_t)2);
            goto after_arrays;
        }

        const unsigned required_flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
        const unsigned flags = PyArray_FLAGS(vec_field);
        if ((flags & required_flags) != required_flags)
        {
            PyErr_Format(PyExc_ValueError,
                         "Vector field %u did not have the required flags %04x, but instead had flags %04x.", i,
                         required_flags, flags);
            goto after_arrays;
        }

        vector_fields.fields[i] = (const double *)PyArray_DATA(vec_field);
    }

    // Extract C pointers

    const double *restrict const coord_bl = PyArray_DATA(bl_array);
    const double *restrict const coord_br = PyArray_DATA(br_array);
    const double *restrict const coord_tr = PyArray_DATA(tr_array);
    const double *restrict const coord_tl = PyArray_DATA(tl_array);
    const unsigned *restrict const orders = PyArray_DATA(orders_array);

    // Prepare output arrays
    double **p_out = allocate(&SYSTEM_ALLOCATOR, sizeof(*p_out) * n_elements);
    if (!p_out)
    {
        goto after_arrays;
    }
    ret_val = PyTuple_New((Py_ssize_t)n_elements);
    if (!ret_val)
    {
        deallocate(&SYSTEM_ALLOCATOR, p_out);
        goto after_arrays;
    }

    // Create an error stack for reporting issues

    for (unsigned i = 0; i < n_elements; ++i)
    {
        size_t element_size = 0;
        for (unsigned j = 0; j < system_template.n_forms; ++j)
        {
            element_size += form_degrees_of_freedom_count(system_template.form_orders[j], orders[i]);
        }
        const npy_intp dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
        PyArrayObject *const a = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!a)
        {
            Py_DECREF(ret_val);
            ret_val = NULL;
            deallocate(&SYSTEM_ALLOCATOR, p_out);
            goto after_arrays;
        }
        PyTuple_SET_ITEM(ret_val, i, a);
        p_out[i] = PyArray_DATA(a);
        memset(p_out[i], 0, sizeof(*p_out[i]) * dims[0] * dims[1]);
    }

    eval_result_t common_res = EVAL_SUCCESS;
    Py_BEGIN_ALLOW_THREADS

#pragma omp parallel default(none)                                                                                     \
    shared(SYSTEM_ALLOCATOR, system_template, stderr, common_res, n_elements, orders, cache_array, n_cache, coord_bl,  \
               coord_br, coord_tr, coord_tl, p_out, thread_stack_size, vector_fields)
    {
        error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
        // Allocate the stack through system allocator
        matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
        allocator_stack_t *const allocator_stack = allocator_stack_create(thread_stack_size, &SYSTEM_ALLOCATOR);
        eval_result_t res = matrix_stack && err_stack ? EVAL_SUCCESS : EVAL_FAILED_ALLOC;
        /* Heavy calculations here */
#pragma omp for nowait
        for (unsigned i_elem = 0; i_elem < n_elements; ++i_elem)
        {
            if (!(common_res == EVAL_SUCCESS && err_stack && matrix_stack && allocator_stack))
            {
                continue;
            }
            allocator_stack_reset(allocator_stack);
            const unsigned order = orders[i_elem];
            size_t element_size = 0;
            for (unsigned j = 0; j < system_template.n_forms; ++j)
            {
                element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order);
            }
            precompute_t precomp;
            unsigned i;
            // Find cached values for the current order
            for (i = 0; i < n_cache; ++i)
            {
                if (order == cache_array[i].order)
                {
                    break;
                }
            }
            if (i == n_cache)
            {
                // Failed, not in cache!
                continue;
            }
            // Compute matrices for the element
            if (!precompute_create(cache_array + i, coord_bl[2 * i_elem + 0], coord_br[2 * i_elem + 0],
                                   coord_tr[2 * i_elem + 0], coord_tl[2 * i_elem + 0], coord_bl[2 * i_elem + 1],
                                   coord_br[2 * i_elem + 1], coord_tr[2 * i_elem + 1], coord_tl[2 * i_elem + 1],
                                   &precomp, &allocator_stack->base))
            {
                // Failed, could not compute precomp
                continue;
            }

            double *restrict const output_mat = p_out[i_elem];

            // Compute the individual entries
            size_t row_offset = 0;
            for (unsigned row = 0; row < system_template.n_forms && res == EVAL_SUCCESS; ++row)
            {
                const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order);
                size_t col_offset = 0;
                for (unsigned col = 0; col < system_template.n_forms /*&& res == EVAL_SUCCESS*/; ++col)
                {
                    const unsigned col_len = form_degrees_of_freedom_count(system_template.form_orders[col], order);
                    const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
                    if (!bytecode)
                    {
                        // Zero entry, we do nothing since arrays start zeroed out (I think).
                        col_offset += col_len;
                        continue;
                    }

                    // Offset the fields to the element
                    field_information_t element_field_information = vector_fields;
                    for (unsigned idx = 0; idx < element_field_information.n_fields; ++idx)
                    {
                        element_field_information.fields[idx] += 2 * element_field_information.offsets[i_elem];
                    }

                    matrix_full_t mat;
                    res = evaluate_element_term_sibling(err_stack, system_template.form_orders[row], order, bytecode,
                                                        &precomp, &element_field_information, system_template.max_stack,
                                                        matrix_stack, &allocator_stack->base, &mat);
                    if (res != EVAL_SUCCESS)
                    {
                        EVAL_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                        break;
                    }
                    if (row_len != mat.base.rows || col_len != mat.base.cols)
                    {
                        EVAL_ERROR(err_stack, EVAL_DIMS_MISMATCH,
                                   "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                                   mat.base.rows, mat.base.cols, row_len, col_len);
                        res = EVAL_DIMS_MISMATCH;
                        break;
                    }

                    for (unsigned i_out = 0; i_out < row_len; ++i_out)
                    {
                        for (unsigned j_out = 0; j_out < col_len; ++j_out)
                        {
                            output_mat[(i_out + row_offset) * element_size + (j_out + col_offset)] =
                                mat.data[i_out * mat.base.cols + j_out];
                        }
                    }

                    deallocate(&allocator_stack->base, mat.data);
                    // SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, mat.data);
                    col_offset += col_len;
                }
                row_offset += row_len;
            }
        }

        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);

        // Clean error stack
        if (err_stack && err_stack->position != 0)
        {
            fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);

            for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
            {
                const error_message_t *msg = err_stack->messages + i_e;
                fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                        eval_result_str(msg->code), msg->message);
                deallocate(err_stack->allocator, msg->message);
            }
        }
        if (err_stack)
            deallocate(err_stack->allocator, err_stack);
        if (allocator_stack)
        {
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        }
#pragma omp critical
        {
            if (res != EVAL_SUCCESS)
            {
                common_res = res;
            }
        }
    }

    Py_END_ALLOW_THREADS if (common_res != EVAL_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Execution failed with error code %s.", eval_result_str(common_res));
        // Failed allocation of matrix stack.
        Py_DECREF(ret_val);
        ret_val = NULL;
    }

    // Clean up the array of output pointers
    deallocate(&SYSTEM_ALLOCATOR, p_out);

    // Clean up the coordinate arrays
after_arrays:
    Py_DECREF(offset_array);
    Py_DECREF(orders_array);
    Py_DECREF(tl_array);
    Py_DECREF(tr_array);
    Py_DECREF(br_array);
    Py_DECREF(bl_array);

    // Clean up the basis caches
after_cache:
    caches_array_destroy(n_cache, cache_array);
    deallocate(&SYSTEM_ALLOCATOR, cache_array);

    // Clean up the template
after_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    return ret_val;
}

static PyObject *element_matrices(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    double x0, x1, x2, x3;
    double y0, y1, y2, y3;
    PyObject *serialized;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddddddddO",
                                     (char *[10]){"x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3", "serialized", NULL},
                                     &x0, &x1, &x2, &x3, &y0, &y1, &y2, &y3, &serialized))
    {
        return NULL;
    }

    basis_precomp_t basis_precomp;
    if (!basis_precomp_create(serialized, &basis_precomp))
    {
        return NULL;
    }

    precompute_t out;
    const int res = precompute_create(&basis_precomp, x0, x1, x2, x3, y0, y1, y2, y3, &out, &SYSTEM_ALLOCATOR);

    if (!res)
    {
        return NULL;
    }
    int failed = 0;
    for (mass_mtx_indices_t t = MASS_0; t < MASS_CNT; ++t)
    {
        const matrix_full_t *m = precompute_get_matrix(&out, t, &SYSTEM_ALLOCATOR);
        if (!m)
        {
            failed = 1;
            PyErr_Format(PyExc_ValueError, "Failed allocating and crating the mass matrix %u.", (unsigned)t);
            break;
        }
    }

    PyObject *ret_val = NULL;

    if (!failed)
    {
        ret_val = PyTuple_Pack(
            6, matrix_full_to_array(out.mass_matrices + MASS_0), matrix_full_to_array(out.mass_matrices + MASS_1),
            matrix_full_to_array(out.mass_matrices + MASS_2), matrix_full_to_array(out.mass_matrices + MASS_0_I),
            matrix_full_to_array(out.mass_matrices + MASS_1_I), matrix_full_to_array(out.mass_matrices + MASS_2_I));
    }

    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.jacobian);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_0_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_1_I].data);
    SYSTEM_ALLOCATOR.free(SYSTEM_ALLOCATOR.state, out.mass_matrices[MASS_2_I].data);

    basis_precomp_destroy(&basis_precomp);
    return ret_val;
}

static PyObject *check_bytecode(PyObject *Py_UNUSED(module), PyObject *in_expression)
{
    size_t n_expr;
    bytecode_t *bytecode = NULL;
    // Convert into bytecode
    {
        PyObject *const expression = PySequence_Fast(in_expression, "Can not convert expression to sequence.");
        if (!expression)
            return NULL;

        n_expr = PySequence_Fast_GET_SIZE(expression);
        PyObject **const p_exp = PySequence_Fast_ITEMS(expression);

        bytecode = PyMem_RawMalloc(sizeof(*bytecode) * (n_expr + 1));
        if (!bytecode)
        {
            Py_DECREF(expression);
            return NULL;
        }
        unsigned unused;
        if (!convert_bytecode(n_expr, bytecode, p_exp, &unused))
        {
            PyMem_RawFree(bytecode);
            Py_DECREF(expression);
            return NULL;
        }
        Py_DECREF(expression);
    }

    PyTupleObject *out_tuple = (PyTupleObject *)PyTuple_New((Py_ssize_t)n_expr);
    for (unsigned i = 1; i <= n_expr; ++i)
    {
        switch (bytecode[i].op)
        {
        case MATOP_IDENTITY:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_IDENTITY));
            break;
        case MATOP_TRANSPOSE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_TRANSPOSE));
            break;
        case MATOP_MATMUL:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MATMUL));
            break;
        case MATOP_SCALE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SCALE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].f64));
            break;
        case MATOP_SUM:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_SUM));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyFloat_FromDouble(bytecode[i].u32));
            break;
        case MATOP_INCIDENCE:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_INCIDENCE));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_MASS:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_MASS));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            i += 1;
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromUnsignedLong(bytecode[i].u32));
            break;
        case MATOP_PUSH:
            PyTuple_SET_ITEM(out_tuple, i - 1, PyLong_FromLong(MATOP_PUSH));
            break;
        default:
            ASSERT(0, "Invalid operation.");
            break;
        }
    }

    PyMem_RawFree(bytecode);

    return (PyObject *)out_tuple;
}

static PyObject *check_incidence(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *in_x;
    unsigned order, form;
    int transpose, right;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OIIpp", (char *[6]){"", "order", "form", "transpose", "right", NULL},
                                     &in_x, &order, &form, &transpose, &right))
    {
        return NULL;
    }
    if (form > 1)
    {
        PyErr_Format(PyExc_ValueError, "Form specified is too high (%u, but only up to 1 is allowed).", order);
        return NULL;
    }
    const incidence_type_t t = ((incidence_type_t)form) + (transpose ? (INCIDENCE_TYPE_10_T - INCIDENCE_TYPE_10) : 0);
    PyArrayObject *const x = (PyArrayObject *)PyArray_FromAny(in_x, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!x)
        return NULL;

    const matrix_full_t in = {.base = {.type = MATRIX_TYPE_FULL, .rows = PyArray_DIM(x, 0), .cols = PyArray_DIM(x, 1)},
                              .data = PyArray_DATA(x)};
    matrix_full_t out;
    eval_result_t res;
    if (right)
    {
        res = apply_incidence_to_full_right(t, order, &in, &out, &SYSTEM_ALLOCATOR);
    }
    else
    {
        res = apply_incidence_to_full_left(t, order, &in, &out, &SYSTEM_ALLOCATOR);
    }
    Py_DECREF(x);
    if (res != EVAL_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not apply the incidence matrix %u (order %u) to a %u by %u matrix, reason: %s.", t, order,
                     in.base.rows, in.base.cols, eval_result_str(res));
        return NULL;
    }
    PyArrayObject *const y = matrix_full_to_array(&out);
    deallocate(&SYSTEM_ALLOCATOR, out.data);
    return (PyObject *)y;
}

static int check_input_array(const PyArrayObject *const arr, const unsigned n_dim, const npy_intp dims[static n_dim],
                             const int dtype, const int flags)
{
    const int arr_flags = PyArray_FLAGS(arr);
    if ((arr_flags & flags) != flags)
    {
        PyErr_Format(PyExc_ValueError, "Array flags %u don't contain required flags %u.", arr_flags, flags);
        return -1;
    }

    if (PyArray_NDIM(arr) != n_dim)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of dimensions for the array does not match expected value (expected %u, got %u).", n_dim,
                     (unsigned)PyArray_NDIM(arr));
        return -1;
    }

    if (dtype >= 0 && PyArray_TYPE(arr) != dtype)
    {
        PyErr_Format(PyExc_ValueError, "Array does not have the expected type (expected %u, got %u).", dtype,
                     PyArray_TYPE(arr));
        return -1;
    }

    const npy_intp *const d = PyArray_DIMS(arr);
    for (unsigned i_dim = 0; i_dim < n_dim; ++i_dim)
    {
        if (dims[i_dim] != 0 && d[i_dim] != dims[i_dim])
        {
            PyErr_Format(PyExc_ValueError, "Dimension %u of the did not match expected value (expected %u, got %u).",
                         i_dim, dims[i_dim], d[i_dim]);
            return -1;
        }
    }

    return 0;
}

static PyObject *continuity_equations(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    manifold2d_object_t *primal, *dual;
    PyArrayObject *form_orders, *element_offsets, *dof_offsets, *element_orders;
    PyTypeObject *const man_type = &manifold2d_type_object;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O!O!O!O!O!",
            (char *[7]){"primal", "dual", "form_orders", "element_offsets", "dof_offsets", "element_orders", NULL},
            man_type, &primal, man_type, &dual, &PyArray_Type, &form_orders, &PyArray_Type, &element_offsets,
            &PyArray_Type, &dof_offsets, &PyArray_Type, &element_orders))
    {
        return NULL;
    }

    // Check primal and dual match
    if (primal->n_points != dual->n_surfaces || primal->n_lines != dual->n_lines ||
        primal->n_surfaces != dual->n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Primal and dual manifolds don't match in terms of geometrical objects (%R and %R).", primal,
                     dual);
        return NULL;
    }

    if (check_input_array(form_orders, 1, (const npy_intp[1]){0}, NPY_UINT,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED) < 0 ||
        check_input_array(element_orders, 1, (const npy_intp[1]){0}, NPY_UINT,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED) < 0)
    {
        return NULL;
    }

    const npy_intp n_forms = PyArray_DIM(form_orders, 0);
    const npy_intp n_elements = PyArray_DIM(element_orders, 0);

    if (check_input_array(element_offsets, 1, (const npy_intp[1]){n_elements + 1}, NPY_UINT,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED) < 0 ||
        check_input_array(dof_offsets, 2, (const npy_intp[2]){n_forms + 1, n_elements}, NPY_UINT,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED) < 0)
    {
        return NULL;
    }

    const unsigned *const p_element_offsets = PyArray_DATA(element_offsets);
    const unsigned *const p_dof_offsets = PyArray_DATA(dof_offsets);
    const form_order_t *const p_form_orders = PyArray_DATA(form_orders);
    const unsigned *const p_element_order = PyArray_DATA(element_orders);

    for (unsigned i_form = 0; i_form < n_forms; ++i_form)
    {
        if (p_form_orders[i_form] > FORM_ORDER_2 || p_form_orders[i_form] == FORM_ORDER_UNKNOWN)
        {
            PyErr_Format(PyExc_ValueError, "For order was of unknown value %u.", (unsigned)p_form_orders[i_form]);
            return NULL;
        }
    }

    for (unsigned i_elm = 0; i_elm < n_elements; ++i_elm)
    {
        if (p_element_offsets[i_elm] >= p_element_offsets[i_elm + 1])
        {
            PyErr_Format(PyExc_ValueError, "Offsets of elements %u and %u are not strictly increasing.", i_elm,
                         i_elm + 1);
            return NULL;
        }

        if (i_elm != 0 && p_element_order[i_elm - 1] != p_element_order[i_elm])
        {
            PyErr_Format(PyExc_ValueError, "Elements %u and %u don't have the same order, which is not (yet allowed).",
                         i_elm, i_elm + 1);
            return NULL;
        }

        for (unsigned i_form = 0; i_form < n_forms; ++i_form)
        {
            const unsigned d0 = p_dof_offsets[(n_elements)*i_form + i_elm];
            const unsigned d1 = p_dof_offsets[(n_elements) * (i_form + 1) + i_elm];
            if (d0 >= d1)
            {
                PyErr_Format(PyExc_ValueError,
                             "Offsets of degrees of freedom %u and %u in the element %u are not strictly increasing "
                             "(%u and %u).",
                             i_form, i_form + 1, i_elm, d0, d1);
                return NULL;
            }
        }
    }

    error_stack_t *error_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    if (!error_stack)
    {
        return NULL;
    }
    unsigned n_equations, *equation_offsets, *equation_indices;

    const eval_result_t res = generate_connectivity_equations(
        primal, dual, n_elements, n_forms, p_form_orders, p_element_order, p_element_offsets, p_dof_offsets,
        &n_equations, &equation_offsets, &equation_indices, &SYSTEM_ALLOCATOR, error_stack);

    if (error_stack->position != 0)
    {
        for (unsigned i = 0; i < error_stack->position; ++i)
        {
            const error_message_t *msg = error_stack->messages + i;
            PySys_FormatStderr("%s:%d - %s: Error %s (%d): %s.\n", msg->file, msg->line, msg->function,
                               eval_result_str(msg->code), msg->code, msg->message);
            deallocate(&SYSTEM_ALLOCATOR, msg->message);
        }
    }
    deallocate(&SYSTEM_ALLOCATOR, error_stack);

    if (res != EVAL_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not generate connectivity equations %s (%d).", eval_result_str(res), res);
        return NULL;
    }
    // else
    // {
    //     printf("%u equations:\n", n_equations);
    //     unsigned offset = 0;
    //     for (unsigned n = 0; n < n_equations; ++n)
    //     {
    //         const unsigned len = equation_offsets[n] - offset;
    //         printf("Equation %u (len %u):", n, len);
    //         for (unsigned i = 0; i < len; ++i)
    //         {
    //             printf(" %u", equation_indices[offset + i]);
    //         }
    //         printf("\n");
    //         offset += len;
    //     }
    // }

    PyArrayObject *const arr_offsets =
        (PyArrayObject *)PyArray_SimpleNew(1, (const npy_intp[1]){n_equations}, NPY_UINT);
    PyArrayObject *const arr_indices =
        (PyArrayObject *)PyArray_SimpleNew(1, (const npy_intp[1]){equation_offsets[n_equations - 1]}, NPY_UINT);
    if (!arr_offsets || !arr_indices)
    {
        Py_XDECREF(arr_offsets);
        Py_XDECREF(arr_indices);
        return NULL;
    }

    memcpy(PyArray_DATA(arr_offsets), equation_offsets, sizeof(*equation_offsets) * n_equations);
    memcpy(PyArray_DATA(arr_indices), equation_indices, sizeof(*equation_indices) * equation_offsets[n_equations - 1]);

    deallocate(&SYSTEM_ALLOCATOR, equation_offsets);
    deallocate(&SYSTEM_ALLOCATOR, equation_indices);

    return PyTuple_Pack(2, arr_offsets, arr_indices);
}

static PyMethodDef module_methods[] = {
    // {"compute_element_matrices", (void *)compute_element_matrices, METH_VARARGS | METH_KEYWORDS,
    //  "Compute element matrices."},
    {"compute_element_matrices_2", (void *)compute_element_matrices_2, METH_VARARGS | METH_KEYWORDS,
     "Compute element matrices by sibling calls."},
    {"element_matrices", (void *)element_matrices, METH_VARARGS | METH_KEYWORDS, "Compute element matrices."},
    {"check_bytecode", check_bytecode, METH_O, "Convert bytecode to C-values, then back to Python."},
    {"check_incidence", (void *)check_incidence, METH_VARARGS | METH_KEYWORDS,
     "Apply the incidence matrix to the input matrix."},
    {"continuity", (void *)continuity_equations, METH_VARARGS | METH_KEYWORDS,
     "Create continuity equation for different forms."},
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "interplib._mimetic",
                             .m_doc = "Internal C-extension implementing mimetic related functionality",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__mimetic(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject *mod = NULL;
    if (!((mod = PyModule_Create(&module))) || PyModule_AddType(mod, &geo_id_type_object) < 0 ||
        PyModule_AddType(mod, &line_type_object) < 0 || PyModule_AddType(mod, &surface_type_object) < 0 ||
        PyModule_AddType(mod, &manifold_type_object) < 0 || PyModule_AddType(mod, &manifold1d_type_object) < 0 ||
        PyModule_AddType(mod, &manifold2d_type_object) < 0 || PyModule_AddType(mod, &svec_type_object) < 0 ||
        PyModule_AddType(mod, &givens_rotation_type_object) < 0 || PyModule_AddType(mod, &lil_mat_type_object) < 0 ||
        PyModule_AddType(mod, &givens_series_type_object) < 0)
    {
        Py_XDECREF(mod);
        return NULL;
    }

    return mod;
}
