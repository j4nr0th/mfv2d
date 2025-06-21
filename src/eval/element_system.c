#include "element_system.h"
#include "allocator.h"
#include "basis.h"
#include "element_eval.h"
#include "fem_space.h"

#include <numpy/ndarrayobject.h>

MFV2D_INTERNAL
PyObject *compute_element_matrix(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *return_value = NULL;
    PyObject *form_orders_obj = NULL;
    PyObject *expressions = NULL;
    PyObject *corners = NULL;
    PyObject *vector_fields_obj = NULL;
    basis_2d_t *basis = NULL;
    Py_ssize_t stack_memory = 1 << 24;

    static char *kwlist[7] = {"form_orders", "expressions",    "corners", "vector_fields",
                              "basis ",      " stack_memory ", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO!|n", kwlist,
                                     &form_orders_obj,   // Sequence[int]
                                     &expressions,       // _CompiledCodeMatrix (PyObject*)
                                     &corners,           // np.ndarray
                                     &vector_fields_obj, // Sequence[np.ndarray]
                                     &basis_2d_type,     // For type checking
                                     &basis,             // Basis2D
                                     &stack_memory       // int
                                     ))
    {
        return NULL;
    }

    const unsigned order_1 = basis->basis_xi->order;
    const unsigned order_2 = basis->basis_eta->order;
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
    if (check_input_array((PyArrayObject *)corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "corners") < 0)
        return NULL;

    const fem_space_1d_t space_1 = basis_1d_as_fem_space(basis->basis_xi),
                         space_2 = basis_1d_as_fem_space(basis->basis_eta);

    const unsigned n1 = space_1.n_pts, n2 = space_2.n_pts;

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
        if (check_input_array((PyArrayObject *)item, 2, (const npy_intp[2]){n2 * n1, 2}, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "vector_fields element") < 0)
        {
            Py_DECREF(item);
            return NULL;
        }
        element_field_information.fields[i] = PyArray_DATA((const PyArrayObject *)item);
        Py_DECREF(item);
    }

    // Create the system template
    system_template_t system_template = {};
    if (!system_template_create(&system_template, form_orders_obj, expressions, (unsigned)n_vector_fields,
                                &SYSTEM_ALLOCATOR))
    {
        goto end;
    }

    size_t element_size = 0;
    for (unsigned j = 0; j < system_template.n_forms; ++j)
    {
        element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order_1, order_2);
    }
    const npy_intp output_dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
    return_value = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_FLOAT64);
    if (!return_value)
        goto cleanup_template;

    const quad_info_t *const quad = PyArray_DATA((PyArrayObject *)corners);
    fem_space_2d_t *fem_space = NULL;
    mfv2d_result_t res = fem_space_2d_create(&space_1, &space_2, quad, &fem_space, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
        goto cleanup_template;

    error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
    allocator_stack_t *const allocator_stack = allocator_stack_create((size_t)stack_memory, &SYSTEM_ALLOCATOR);
    if (!matrix_stack || !err_stack || !allocator_stack)
    {
        res = MFV2D_FAILED_ALLOC;
        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
        deallocate(&SYSTEM_ALLOCATOR, err_stack);
        if (allocator_stack)
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
        deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        goto cleanup_fem_space;
    }
    memset(matrix_stack, 0, sizeof *matrix_stack * system_template.max_stack);

    double *restrict const output_mat = PyArray_DATA(return_value);
    memset(output_mat, 0, sizeof *output_mat * element_size * element_size);
    unsigned row_offset = 0;
    for (unsigned row = 0; row < system_template.n_forms && res == MFV2D_SUCCESS; ++row)
    {
        const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order_1, order_2);
        size_t col_offset = 0;
        for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
        {
            const unsigned col_len = form_degrees_of_freedom_count(system_template.form_orders[col], order_1, order_2);
            const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
            if (!bytecode)
            {
                // Zero entry, we do nothing, since arrays start zeroed out (I think).
                col_offset += col_len;
                continue;
            }

            matrix_full_t mat;
            res = evaluate_block(err_stack, system_template.form_orders[row], order_1, bytecode, fem_space,
                                 &element_field_information, system_template.max_stack, matrix_stack,
                                 &allocator_stack->base, &mat, NULL);
            if (res != MFV2D_SUCCESS)
            {
                MFV2D_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                break;
            }
            if (row_len != mat.base.rows || col_len != mat.base.cols)
            {
                MFV2D_ERROR(err_stack, MFV2D_DIMS_MISMATCH,
                            "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                            mat.base.rows, mat.base.cols, row_len, col_len);
                res = MFV2D_DIMS_MISMATCH;
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
            col_offset += col_len;
        }
        row_offset += row_len;
    }

// Cleanup
cleanup_stacks:
    deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
    // Clean error stack
    if (err_stack->position != 0)
    {
        fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);

        for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
        {
            const error_message_t *msg = err_stack->messages + i_e;
            fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                    mfv2d_result_str(msg->code), msg->message);
            deallocate(err_stack->allocator, msg->message);
        }
    }
    deallocate(err_stack->allocator, err_stack);
    deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
    deallocate(&SYSTEM_ALLOCATOR, allocator_stack);

cleanup_fem_space:
    deallocate(&SYSTEM_ALLOCATOR, fem_space);
cleanup_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
end:
    if (res != MFV2D_SUCCESS)
    {
        Py_XDECREF(return_value);
        PyErr_Format(PyExc_RuntimeError, "Runtime error %s was encountered.", mfv2d_result_str(res));
        return NULL;
    }
    return (PyObject *)return_value;
}

MFV2D_INTERNAL
PyObject *compute_element_vector(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *return_value = NULL;
    PyObject *form_orders_obj = NULL;
    PyObject *expressions = NULL;
    PyObject *corners = NULL;
    PyObject *vector_fields_obj = NULL;
    PyArrayObject *solution = NULL;
    basis_2d_t *basis = NULL;
    Py_ssize_t stack_memory = 1 << 24;

    static char *kwlist[8] = {"form_orders", "expressions", "corners",        "vector_fields",
                              "basis ",      "solution",    " stack_memory ", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO!O|n", kwlist,
                                     &form_orders_obj,   // Sequence[int]
                                     &expressions,       // _CompiledCodeMatrix (PyObject*)
                                     &corners,           // np.ndarray
                                     &vector_fields_obj, // Sequence[np.ndarray]
                                     &basis_2d_type,     // For type checking
                                     &basis,             // Basis2D
                                     &solution,          // np.ndarray
                                     &stack_memory       // int
                                     ))
    {
        return NULL;
    }

    const unsigned order_1 = basis->basis_xi->order;
    const unsigned order_2 = basis->basis_eta->order;
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
    if (check_input_array((PyArrayObject *)corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "corners") < 0)
        return NULL;

    const fem_space_1d_t space_1 = basis_1d_as_fem_space(basis->basis_xi),
                         space_2 = basis_1d_as_fem_space(basis->basis_eta);

    const unsigned n1 = space_1.n_pts, n2 = space_2.n_pts;

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
        if (check_input_array((PyArrayObject *)item, 2, (const npy_intp[2]){n2 * n1, 2}, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "vector_fields element") < 0)
        {
            Py_DECREF(item);
            return NULL;
        }
        element_field_information.fields[i] = PyArray_DATA((const PyArrayObject *)item);
        Py_DECREF(item);
    }

    // Create the system template
    system_template_t system_template = {};
    if (!system_template_create(&system_template, form_orders_obj, expressions, (unsigned)n_vector_fields,
                                &SYSTEM_ALLOCATOR))
    {
        goto end;
    }

    size_t element_size = 0;
    for (unsigned j = 0; j < system_template.n_forms; ++j)
    {
        element_size += form_degrees_of_freedom_count(system_template.form_orders[j], order_1, order_2);
    }
    if (check_input_array((PyArrayObject *)solution, 1, (const npy_intp[1]){(npy_intp)element_size}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "solution") < 0)
    {
        goto cleanup_template;
    }
    double *restrict const solution_vec = PyArray_DATA((PyArrayObject *)solution);

    const npy_intp output_dims[1] = {(npy_intp)element_size};
    return_value = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_DOUBLE);
    if (!return_value)
        goto cleanup_template;

    const quad_info_t *const quad = PyArray_DATA((PyArrayObject *)corners);
    fem_space_2d_t *fem_space = NULL;
    mfv2d_result_t res = fem_space_2d_create(&space_1, &space_2, quad, &fem_space, &SYSTEM_ALLOCATOR);
    if (res != MFV2D_SUCCESS)
        goto cleanup_template;

    error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
    allocator_stack_t *const allocator_stack = allocator_stack_create((size_t)stack_memory, &SYSTEM_ALLOCATOR);
    if (!matrix_stack || !err_stack || !allocator_stack)
    {
        res = MFV2D_FAILED_ALLOC;
        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
        deallocate(&SYSTEM_ALLOCATOR, err_stack);
        if (allocator_stack)
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
        deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        goto cleanup_fem_space;
    }
    memset(matrix_stack, 0, sizeof *matrix_stack * system_template.max_stack);
    double *restrict const output_mat = PyArray_DATA(return_value);
    memset(output_mat, 0, sizeof *output_mat * element_size);
    unsigned row_offset = 0;
    for (unsigned row = 0; row < system_template.n_forms; ++row)
    {
        const unsigned row_len = form_degrees_of_freedom_count(system_template.form_orders[row], order_1, order_2);
        size_t col_offset = 0;
        for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
        {
            const unsigned col_len = form_degrees_of_freedom_count(system_template.form_orders[col], order_1, order_2);
            const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
            if (!bytecode)
            {
                // Zero entry, we do nothing, since arrays start zeroed out (I think).
                col_offset += col_len;
                continue;
            }

            matrix_full_t mat;
            const matrix_full_t initial = {.base = {.type = MATRIX_TYPE_FULL, .rows = col_len, .cols = 1},
                                           .data = solution_vec + col_offset};
            res = evaluate_block(err_stack, system_template.form_orders[row], order_1, bytecode, fem_space,
                                 &element_field_information, system_template.max_stack, matrix_stack,
                                 &allocator_stack->base, &mat, &initial);
            if (res != MFV2D_SUCCESS)
            {
                MFV2D_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                goto cleanup_stacks;
            }
            if (row_len != mat.base.rows || 1 != mat.base.cols)
            {
                MFV2D_ERROR(err_stack, MFV2D_DIMS_MISMATCH,
                            "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                            mat.base.rows, mat.base.cols, row_len, 1);
                res = MFV2D_DIMS_MISMATCH;
                goto cleanup_stacks;
            }

            for (unsigned i_out = 0; i_out < row_len; ++i_out)
            {
                output_mat[i_out + row_offset] += mat.data[i_out];
            }

            deallocate(&allocator_stack->base, mat.data);
            col_offset += col_len;
        }
        row_offset += row_len;
    }

// Cleanup
cleanup_stacks:
    // Clean error stack
    if (err_stack->position != 0)
    {
        fprintf(stderr, "Error stack caught %u errors.\n", err_stack->position);

        for (unsigned i_e = 0; i_e < err_stack->position; ++i_e)
        {
            const error_message_t *msg = err_stack->messages + i_e;
            fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                    mfv2d_result_str(msg->code), msg->message);
            deallocate(err_stack->allocator, msg->message);
        }
    }
    deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
    deallocate(err_stack->allocator, err_stack);
    deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
    deallocate(&SYSTEM_ALLOCATOR, allocator_stack);

cleanup_fem_space:
    deallocate(&SYSTEM_ALLOCATOR, fem_space);
cleanup_template:
    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
end:
    if (res != MFV2D_SUCCESS)
    {
        Py_XDECREF(return_value);
        PyErr_Format(PyExc_RuntimeError, "Runtime error %s was encountered.", mfv2d_result_str(res));
        return NULL;
    }
    return (PyObject *)return_value;
}
MFV2D_INTERNAL
const char compute_element_projector_docstr[] =
    "compute_element_projector(form_orders: Sequence[UnknownFormOrders],  corners: array, basis_in: Basis2D, "
    "basis_out: Basis2D) -> tuple[array]:\n"
    "Compute :math:`L^2` projection from one space to another.\n"
    "\n"
    "Projection takes DoFs from primal space of the first and takes\n"
    "them to the primal space of the other.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "form_orders : Sequence of UnknownFormOrder\n"
    "    Sequence of orders of forms which are to be projected.\n"
    "\n"
    "corners : (4, 2) array\n"
    "    Array of corner points of the element.\n"
    "\n"
    "basis_in : Basis2D\n"
    "    Basis from which the DoFs should be taken.\n"
    "\n"
    "basis_out : Basis2D\n"
    "    Basis to which the DoFs are taken.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "tuple of square arrays\n"
    "    Tuple where each entry is the respective projection matrix for that form.\n";

MFV2D_INTERNAL
PyObject *compute_element_projector(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *form_orders_obj = NULL;
    PyArrayObject *corners = NULL;
    basis_2d_t *basis_in = NULL;
    basis_2d_t *basis_out = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO!O!O!", (char *const[5]){"form_orders", "corners", "basis_in", "basis_out", NULL},
            &form_orders_obj, &PyArray_Type, &corners, &basis_2d_type, &basis_in, &basis_2d_type, &basis_out))
    {
        return NULL;
    }

    if (!PySequence_Check(form_orders_obj))
    {
        PyErr_Format(PyExc_TypeError, "form_orders must be a sequence, instead it was %R.", Py_TYPE(form_orders_obj));
        return NULL;
    }

    if (check_input_array(corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS,
                          "corners") < 0)
        return NULL;

    const unsigned n_forms = PySequence_Size(form_orders_obj);
    if (n_forms == 0)
    {
        PyErr_SetString(PyExc_ValueError, "form_orders must be non-empty.");
        return NULL;
    }
    form_order_t *orders = allocate(&SYSTEM_ALLOCATOR, sizeof *orders * n_forms);
    if (!orders)
    {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n_forms; ++i)
    {
        PyObject *item = PySequence_GetItem(form_orders_obj, i);
        if (!item)
        {
            deallocate(&SYSTEM_ALLOCATOR, orders);
            return NULL;
        }
        orders[i] = (form_order_t)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred())
        {
            deallocate(&SYSTEM_ALLOCATOR, orders);
            return NULL;
        }
        if (orders[i] < FORM_ORDER_0 || orders[i] > FORM_ORDER_2)
        {
            PyErr_Format(PyExc_ValueError, "form_orders must be in range [0, 2], got %d.", orders[i]);
            deallocate(&SYSTEM_ALLOCATOR, orders);
            return NULL;
        }
    }

    const quad_info_t *const quad = PyArray_DATA((PyArrayObject *)corners);
    fem_space_2d_t *fem_space_in = NULL, *fem_space_out = NULL;
    {
        const fem_space_1d_t space_xi = basis_1d_as_fem_space(basis_in->basis_xi);
        const fem_space_1d_t space_eta = basis_1d_as_fem_space(basis_in->basis_eta);
        const mfv2d_result_t res = fem_space_2d_create(&space_xi, &space_eta, quad, &fem_space_in, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not create FEM space for input basis.");
            deallocate(&SYSTEM_ALLOCATOR, orders);
            return NULL;
        }
    }
    {
        const fem_space_1d_t space_xi = basis_1d_as_fem_space(basis_out->basis_xi);
        const fem_space_1d_t space_eta = basis_1d_as_fem_space(basis_out->basis_eta);
        const mfv2d_result_t res = fem_space_2d_create(&space_xi, &space_eta, quad, &fem_space_out, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not create FEM space for input basis.");
            deallocate(&SYSTEM_ALLOCATOR, fem_space_in);
            deallocate(&SYSTEM_ALLOCATOR, orders);
            return NULL;
        }
    }
    mfv2d_result_t res = MFV2D_SUCCESS;
    PyTupleObject *out = (PyTupleObject *)PyTuple_New(n_forms);

    // Check integration rules' orders match
    if (fem_space_in->space_1.n_pts != fem_space_out->space_1.n_pts ||
        fem_space_in->space_2.n_pts != fem_space_out->space_2.n_pts)
    {
        PyErr_Format(PyExc_ValueError,
                     "Integration rules must have same number of points in each dimension (got %u and %u for first and "
                     "%u and %u for second).",
                     fem_space_in->space_1.n_pts, fem_space_out->space_1.n_pts, fem_space_in->space_2.n_pts,
                     fem_space_out->space_2.n_pts);
        res = MFV2D_DIMS_MISMATCH;
        goto end;
    }

    for (form_order_t order = FORM_ORDER_0; order <= FORM_ORDER_2; ++order)
    {
        unsigned order_count = 0;
        // Count how often it occurs
        for (unsigned i = 0; i < n_forms; ++i)
        {
            if (orders[i] == order)
                ++order_count;
        }
        if (order_count == 0)
            continue;

        // Compute mixed mass matrix.
        matrix_full_t mat;
        switch (order)
        {
        case FORM_ORDER_0:
            res = compute_mass_matrix_node_double(fem_space_in, fem_space_out, &mat, &SYSTEM_ALLOCATOR);
            break;

        case FORM_ORDER_1:
            res = compute_mass_matrix_edge_double(fem_space_in, fem_space_out, &mat, &SYSTEM_ALLOCATOR);
            break;

        case FORM_ORDER_2:
            res = compute_mass_matrix_surf_double(fem_space_in, fem_space_out, &mat, &SYSTEM_ALLOCATOR);
            break;

        default:
            PyErr_Format(PyExc_ValueError, "Invalid form order %d.", order);
            res = MFV2D_BAD_ENUM;
            goto end;
        }

        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not compute mixed mass matrix for order %d, error code %s.", order,
                         mfv2d_result_str(res));
            goto end;
        }

        // Compute the inverse mass matrix
        const bytecode_t bytecode[4] = {
            [0] = {.u32 = 3},
            [1] = {.op = MATOP_MASS},
            [2] = {.u32 = order - 1},
            [3] = {.u32 = 1},
        };

        matrix_full_t out_mat;
        error_stack_t *error_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
        if (!error_stack)
        {
            res = MFV2D_FAILED_ALLOC;
            deallocate(&SYSTEM_ALLOCATOR, mat.data);
            goto end;
        }
        enum
        {
            MATRIX_STACK_SIZE = 3
        };
        matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * MATRIX_STACK_SIZE);
        if (!matrix_stack)
        {
            res = MFV2D_FAILED_ALLOC;
            deallocate(&SYSTEM_ALLOCATOR, error_stack);
            deallocate(&SYSTEM_ALLOCATOR, mat.data);
            goto end;
        }
        memset(matrix_stack, 0, sizeof *matrix_stack * MATRIX_STACK_SIZE);
        res = evaluate_block(error_stack, order, -1, bytecode, fem_space_out, NULL, MATRIX_STACK_SIZE, matrix_stack,
                             &SYSTEM_ALLOCATOR, &out_mat, &mat);
        deallocate(&SYSTEM_ALLOCATOR, mat.data);
        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
        if (error_stack->position != 0)
        {
            for (unsigned i = 0; i < error_stack->position; ++i)
            {
                error_message_t *const msg = error_stack->messages + i;
                fprintf(stderr, "%s:%d in %s: (%s) - %s\n", msg->file, msg->line, msg->function,
                        mfv2d_result_str(msg->code), msg->message);
                deallocate(error_stack->allocator, msg);
            }
        }
        deallocate(&SYSTEM_ALLOCATOR, error_stack);

        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not compute mixed mass matrix for order %d, error code %s.", order,
                         mfv2d_result_str(res));
            goto end;
        }

        PyArrayObject *array = matrix_full_to_array(&out_mat);
        deallocate(&SYSTEM_ALLOCATOR, out_mat.data);
        if (!array)
        {
            res = MFV2D_FAILED_ALLOC;
            goto end;
        }
        for (unsigned i = 0; i < n_forms; ++i)
        {
            if (orders[i] != order)
                continue;

            PyTuple_SET_ITEM(out, i, array);
            Py_INCREF(array);
        }

        Py_DECREF(array);
    }

end:
    if (res != MFV2D_SUCCESS)
    {
        Py_DECREF(out);
        out = NULL;
    }

    deallocate(&SYSTEM_ALLOCATOR, fem_space_out);
    deallocate(&SYSTEM_ALLOCATOR, fem_space_in);
    deallocate(&SYSTEM_ALLOCATOR, orders);

    return (PyObject *)out;
}

MFV2D_INTERNAL
const char compute_element_mass_matrix_docstr[] =
    "compute_element_mass_matrix(form_orders: Sequence[UnknownFormOrders], corners: array, basis: Basis2D, inverse: "
    "bool = False) -> array\n"
    "Compute mass matrix for a given element.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "form_order : UnknownFormOrder\n"
    "    Order of the form for which the mass matrix should be computed.\n"
    "\n"
    "corners : (4, 2) array\n"
    "    Array of corner points of the element.\n"
    "\n"
    "basis : Basis2D\n"
    "    Basis used for the test and sample space.\n"
    "\n"
    "inverse : bool, default: False\n"
    "    Should the inverse of the matrix be computed instead of its value directly.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Mass matrix (or its inverse if specified) for the appropriate form.\n";

PyObject *compute_element_mass_matrix(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    form_order_t order;
    basis_2d_t *basis = NULL;
    PyArrayObject *corners = NULL;
    int inverse = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!|p",
                                     (char *const[5]){"order", "corners", "basis", "inverse", NULL}, &order,
                                     &PyArray_Type, &corners, &basis_2d_type, &basis, &inverse))
    {
        return NULL;
    }
    if (order < FORM_ORDER_0 || order > FORM_ORDER_2)
    {
        PyErr_Format(PyExc_ValueError, "order must be in range [1, 3], got %d.", order);
    }

    if (check_input_array(corners, 2, (const npy_intp[2]){4, 2}, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS,
                          "corners") < 0)
        return NULL;

    const quad_info_t *const quad = PyArray_DATA((PyArrayObject *)corners);
    fem_space_2d_t *fem_space = NULL;
    {
        const fem_space_1d_t space_xi = basis_1d_as_fem_space(basis->basis_xi);
        const fem_space_1d_t space_eta = basis_1d_as_fem_space(basis->basis_eta);
        const mfv2d_result_t res = fem_space_2d_create(&space_xi, &space_eta, quad, &fem_space, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not create FEM space for input basis.");
            return NULL;
        }
    }

    PyArrayObject *return_value = NULL;
    mfv2d_result_t res = MFV2D_SUCCESS;
    matrix_full_t mat;
    switch (order)
    {
    case FORM_ORDER_0:
        res = compute_mass_matrix_node(fem_space, &mat, &SYSTEM_ALLOCATOR);
        break;

    case FORM_ORDER_1:
        res = compute_mass_matrix_edge(fem_space, &mat, &SYSTEM_ALLOCATOR);
        break;

    case FORM_ORDER_2:
        res = compute_mass_matrix_surf(fem_space, &mat, &SYSTEM_ALLOCATOR);
        break;

    default:
        PyErr_Format(PyExc_ValueError, "Invalid form order %d.", order);
        goto end;
    }

    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not compute mass matrix for order %d, error code %s.", order,
                     mfv2d_result_str(res));
        goto end;
    }

    if (inverse)
    {
        matrix_full_t tmp;
        res = matrix_full_invert(&mat, &tmp, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not invert mass matrix for order %d, error code %s.", order,
                         mfv2d_result_str(res));
            goto end;
        }
        // Free memory and swap back
        deallocate(&SYSTEM_ALLOCATOR, mat.data);
        mat = tmp;
    }

    return_value = matrix_full_to_array(&mat);
    deallocate(&SYSTEM_ALLOCATOR, mat.data);

end:
    deallocate(&SYSTEM_ALLOCATOR, fem_space);
    return (PyObject *)return_value;
}
