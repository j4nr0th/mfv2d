#include "element_system.h"
#include "allocator.h"
#include "evaluation.h"
#include <numpy/ndarrayobject.h>

static inline int check_float64_array(PyObject *obj, const char *name, int ndim_expected,
                                      const npy_intp dims_expected[static ndim_expected])
{
    if (!PyArray_Check(obj) || PyArray_TYPE((PyArrayObject *)obj) != NPY_FLOAT64)
    {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy.ndarray of dtype float64", name);
        return 0;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    int ndim = PyArray_NDIM(arr);

    if (ndim_expected > 0 && ndim != ndim_expected)
    {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions (got %d)", name, ndim_expected, ndim);
        return 0;
    }
    if (ndim_expected > 0)
    {
        const npy_intp *shape = PyArray_SHAPE(arr);
        for (int i = 0; i < ndim_expected; ++i)
        {
            if (dims_expected[i] > 0 && shape[i] != dims_expected[i])
            {
                PyErr_Format(PyExc_ValueError, "%s: dimension %d must be of size %ld (got %ld)", name, i,
                             (long)dims_expected[i], (long)shape[i]);
                return 0;
            }
        }
    }
    return 1;
}

MFV2D_INTERNAL
PyObject *compute_element_matrix(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    eval_result_t res;
    PyArrayObject *return_value = NULL;
    PyObject *form_orders_obj = NULL;
    PyObject *expressions = NULL;
    PyObject *corners = NULL;
    int order_1, order_2;
    PyObject *vector_fields_obj = NULL;
    PyObject *basis_1_nodal = NULL;
    PyObject *basis_1_edge = NULL;
    PyObject *weights_1 = NULL;
    PyObject *nodes_1 = NULL;
    PyObject *basis_2_nodal = NULL;
    PyObject *basis_2_edge = NULL;
    PyObject *weights_2 = NULL;
    PyObject *nodes_2 = NULL;
    Py_ssize_t stack_memory = 1 << 24;

    static char *kwlist[16] = {"form_orders", "expressions",   "corners",       "order_1",
                               "order_2",     "vector_fields", "basis_1_nodal", "basis_1_edge",
                               "weights_1",   "nodes_1",       "basis_2_nodal", "basis_2_edge",
                               "weights_2",   "nodes_2",       "stack_memory",  NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOOOOOOO|n", kwlist,
                                     &form_orders_obj,   // Sequence[int]
                                     &expressions,       // _CompiledCodeMatrix (PyObject*)
                                     &corners,           // np.ndarray
                                     &order_1,           // int
                                     &order_2,           // int
                                     &vector_fields_obj, // Sequence[np.ndarray]
                                     &basis_1_nodal,     // np.ndarray
                                     &basis_1_edge,      // np.ndarray
                                     &weights_1,         // np.ndarray
                                     &nodes_1,           // np.ndarray
                                     &basis_2_nodal,     // np.ndarray
                                     &basis_2_edge,      // np.ndarray
                                     &weights_2,         // np.ndarray
                                     &nodes_2,           // np.ndarray
                                     &stack_memory       // int
                                     ))
    {
        return NULL;
    }
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders of the element must be strictly positive (order_1=%d, order_2=%d).",
                     order_1, order_2);
        return NULL;
    }

    if (order_1 != order_2)
    {
        PyErr_Format(PyExc_NotImplementedError,
                     "Currently element must have same order in each dimension (order_1=%d, order_2=%d).", order_1,
                     order_2);
        return NULL;
    }

    if (stack_memory <= 0 || stack_memory & 7)
    {
        PyErr_Format(PyExc_ValueError, "Stack memory size should be positive and a multiple of 8 (stack_memory=%llu).",
                     (unsigned long long)stack_memory);
        return NULL;
    }

    // Validate form_orders is a sequence
    if (!PySequence_Check(form_orders_obj))
    {
        PyErr_SetString(PyExc_TypeError, "form_orders must be a sequence of int");
        return NULL;
    }

    // Validate vector_fields is a sequence
    if (!PySequence_Check(vector_fields_obj))
    {
        PyErr_SetString(PyExc_TypeError, "vector_fields must be a sequence of np.ndarray");
        return NULL;
    }

    // Validate corners, basis, weights, stack_memory are NumPy arrays of type float64
    if (!check_float64_array(corners, "corners", 2, (const npy_intp[2]){4, 2}))
        return NULL;

    // First dimension
    if (!check_float64_array(nodes_1, "nodes_1", 1, (const npy_intp[1]){0}))
        return NULL;
    const unsigned n1 = PyArray_SIZE((const PyArrayObject *)nodes_1);
    if (!check_float64_array(basis_1_nodal, "basis_1_nodal", 1, (const npy_intp[1]){n1}))
        return NULL;
    if (!check_float64_array(basis_1_edge, "basis_1_edge", 1, (const npy_intp[1]){n1}))
        return NULL;
    if (!check_float64_array(weights_1, "weights_1", 1, (const npy_intp[1]){n1}))
        return NULL;

    // Second dimension
    if (!check_float64_array(nodes_2, "nodes_2", 1, (const npy_intp[1]){0}))
        return NULL;
    const unsigned n2 = PyArray_SIZE((const PyArrayObject *)nodes_2);
    if (!check_float64_array(basis_2_nodal, "basis_2_nodal", 1, (const npy_intp[1]){n2}))
        return NULL;
    if (!check_float64_array(basis_2_edge, "basis_2_edge", 1, (const npy_intp[1]){n2}))
        return NULL;
    if (!check_float64_array(weights_2, "weights_2", 1, (const npy_intp[1]){n2}))
        return NULL;

    // Validate each element in vector_fields is a float64 ndarray
    const Py_ssize_t n_vector_fields = PySequence_Size(vector_fields_obj);
    if (n_vector_fields > VECTOR_FIELDS_MAX)
    {
        PyErr_Format(PyExc_ValueError, "Can not have more than %d vector fields (%d were given).", VECTOR_FIELDS_MAX,
                     (int)n_vector_fields);
        return NULL;
    }
    field_information_t element_field_information = {.n_fields = n_vector_fields, .offsets = NULL, .fields = {}};
    for (Py_ssize_t i = 0; i < n_vector_fields; ++i)
    {
        PyObject *item = PySequence_GetItem(vector_fields_obj, i);
        if (!check_float64_array(item, "vector_fields element", 2, (const npy_intp[2]){n2, n1}))
        {
            Py_DECREF(item);
            return NULL;
        }
        element_field_information.fields[i] = PyArray_DATA((const PyArrayObject *)item);
        Py_DECREF(item);
    }

    // Create the system template
    system_template_t system_template;
    if (!system_template_create(&system_template, form_orders_obj, expressions, (unsigned)n_vector_fields,
                                &SYSTEM_ALLOCATOR))
        return NULL;

    size_t element_size = 0;
    for (unsigned j = 0; j < system_template.n_forms; ++j)
    {
        element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order_1, order_2);
    }
    const npy_intp output_dims[2] = {element_size, element_size};
    return_value = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_FLOAT64);
    if (!return_value)
        goto cleanup_template;

    const double (*const ptr_corners)[4][2] = (double (*)[4][2])PyArray_DATA((PyArrayObject *)corners);
    error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
    allocator_stack_t *const allocator_stack = allocator_stack_create((size_t)stack_memory, &SYSTEM_ALLOCATOR);
    if (!matrix_stack || !err_stack)
    {
        res = EVAL_FAILED_ALLOC;
        goto cleanup_template;
    }
    const unsigned order = order_1; // TODO: maybe incorporate second order

    unsigned i;

    double *restrict const output_mat = PyArray_DATA(return_value);
    unsigned row_offset = 0;
    for (unsigned row = 0; row < system_template.n_forms && res == EVAL_SUCCESS; ++row)
    {
        const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order_1, order_2);
        size_t col_offset = 0;
        for (unsigned col = 0; col < system_template.n_forms /*&& res == EVAL_SUCCESS*/; ++col)
        {
            const unsigned col_len = form_degrees_of_freedom_count(system_template.form_orders[col], order_1, order_2);
            const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
            if (!bytecode)
            {
                // Zero entry, we do nothing since arrays start zeroed out (I think).
                col_offset += col_len;
                continue;
            }

            matrix_full_t mat;
            // res = evaluate_element_term_sibling(err_stack, system_template.form_orders[row], order_1, bytecode,
            //                                     &precomp, &element_field_information, system_template.max_stack,
            //                                     matrix_stack, &allocator_stack->base, &mat, NULL);
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

    // Cleanup
cleanup_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
    if (res != EVAL_SUCCESS)
    {
        Py_XDECREF(return_value);
        PyErr_Format(PyExc_RuntimeError, "Runtime error %s was encountered.", eval_result_str(res));
        return NULL;
    }
    return (PyObject *)return_value;
}
