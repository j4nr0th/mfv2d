#include "element_system.h"
#include "../common/allocator.h"
#include "../fem_space/basis.h"
#include "../fem_space/element_fem_space.h"
#include "../fem_space/fem_space.h"
#include "element_eval.h"

#include <numpy/ndarrayobject.h>

#include "forms.h"

MFV2D_INTERNAL
PyObject *compute_element_matrix(PyObject *mod, PyObject *args, PyObject *kwargs)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    PyArrayObject *return_value = NULL;
    element_form_spec_t *form_specs = NULL;
    PyObject *expressions = NULL;
    element_fem_space_2d_t *element_fem_space = NULL;
    PyArrayObject *py_degrees_of_freedom = NULL;
    Py_ssize_t stack_memory = 1 << 24;

    static char *kwlist[6] = {"form_specs",         "expressions",  "element_fem_space",
                              "degrees_of_freedom", "stack_memory", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OO!|O!n", kwlist,
                                     state->type_form_spec,  // for type checking
                                     &form_specs,            // ElementFormsSpecs
                                     &expressions,           // _CompiledCodeMatrix (PyObject*)
                                     state->type_fem_space,  // for type checking
                                     &element_fem_space,     // element cache
                                     &PyArray_Type,          // for type checking
                                     &py_degrees_of_freedom, // array
                                     &stack_memory           // int
                                     ))
    {
        return NULL;
    }

    const unsigned order_1 = element_fem_space->basis_xi->order;
    const unsigned order_2 = element_fem_space->basis_eta->order;
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

    // Check the degrees of freedom if any were passed.
    const unsigned element_size = element_form_specs_total_count(form_specs, order_1, order_2);

    if (py_degrees_of_freedom &&
        check_input_array(py_degrees_of_freedom, 1, (const npy_intp[1]){(npy_intp)element_size}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "degrees_of_freedom") < 0)
    {
        return NULL;
    }

    double *restrict degrees_of_freedom;
    if (py_degrees_of_freedom)
    {
        degrees_of_freedom = PyArray_DATA(py_degrees_of_freedom);
    }
    else
    {
        degrees_of_freedom = allocate(&SYSTEM_ALLOCATOR, sizeof *degrees_of_freedom * element_size);
        if (!degrees_of_freedom)
        {
            return NULL;
        }
        memset(degrees_of_freedom, 0, sizeof *degrees_of_freedom * element_size);
    }

    // Create the system template
    system_template_t system_template = {};
    mfv2d_result_t res = MFV2D_SUCCESS;

    res = system_template_create(&system_template, form_specs, element_fem_space, expressions, &SYSTEM_ALLOCATOR,
                                 degrees_of_freedom);

    if (!py_degrees_of_freedom)
    {
        deallocate(&SYSTEM_ALLOCATOR, degrees_of_freedom);
    }
    degrees_of_freedom = NULL;

    if (res != MFV2D_SUCCESS)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Could not parse bytecode into system template, reason: %s",
                                     mfv2d_result_str(res));
        return NULL;
    }

    const npy_intp output_dims[2] = {(npy_intp)element_size, (npy_intp)element_size};
    return_value = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_FLOAT64);
    if (!return_value)
    {
        system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
        return NULL;
    }

    error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
    allocator_stack_t *const allocator_stack = allocator_stack_create((size_t)stack_memory, &SYSTEM_ALLOCATOR);
    if (!matrix_stack || !err_stack || !allocator_stack)
    {
        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
        deallocate(&SYSTEM_ALLOCATOR, err_stack);
        if (allocator_stack)
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
        deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        Py_DECREF(return_value);
        system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
        return NULL;
    }
    memset(matrix_stack, 0, sizeof *matrix_stack * system_template.max_stack);

    double *restrict const output_mat = PyArray_DATA(return_value);
    memset(output_mat, 0, sizeof *output_mat * element_size * element_size);
    unsigned row_offset = 0;

    Py_BEGIN_ALLOW_THREADS;

    for (unsigned row = 0; row < system_template.n_forms && res == MFV2D_SUCCESS; ++row)
    {
        const unsigned row_len = form_degrees_of_freedom_count(form_specs->forms[row].order, order_1, order_2);
        size_t col_offset = 0;
        for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
        {
            const unsigned col_len = form_degrees_of_freedom_count(form_specs->forms[col].order, order_1, order_2);
            const bytecode_t *bytecode = system_template.bytecodes[row * system_template.n_forms + col];
            if (!bytecode)
            {
                // Zero entry, we do nothing, since arrays start zeroed out (I think).
                col_offset += col_len;
                continue;
            }

            matrix_full_t mat;
            res = evaluate_block(err_stack, form_specs->forms[row].order, order_1, bytecode, element_fem_space,
                                 &system_template.fields, system_template.max_stack, matrix_stack,
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

    Py_END_ALLOW_THREADS;

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

    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    if (res != MFV2D_SUCCESS)
    {
        Py_XDECREF(return_value);
        raise_exception_from_current(PyExc_RuntimeError, "Runtime error %s was encountered.", mfv2d_result_str(res));
        return NULL;
    }
    return (PyObject *)return_value;
}

MFV2D_INTERNAL
const char compute_element_matrix_docstr[] =
    "compute_element_matrix(form_specs: _ElementFormsSpecs, expressions: mfv2d.eval._CompiledCodeMatrix, "
    "fem_space: ElementFemSpace2D, degrees_of_freedom: array | None = None, stack_memory: int = 1 << 24,"
    ") -> NDArray\n"
    "Compute a single element matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "form_specs : _ElementFormsSpecs\n"
    "    Specification of differential forms that make up the system.\n"
    "\n"
    "expressions : mfv2d.eval._CompiledCodeMatrix\n"
    "    Instructions to execute for each block in the system.\n"
    "\n"
    "fem_space : ElementFemSpace2D\n"
    "    Element's FEM space.\n"
    "\n"
    "degrees_of_freedom : array, optional\n"
    "   Degrees of freedom for the element. If specified, these are used for non-linear\n"
    "   interior product terms.\n"
    "\n"
    "stack_memory : int, default: 1 << 24\n"
    "    Amount of memory to use for the evaluation stack.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Element matrix for the specified system.\n";

MFV2D_INTERNAL
PyObject *compute_element_vector(PyObject *mod, PyObject *args, PyObject *kwargs)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    PyArrayObject *return_value = NULL;
    element_form_spec_t *form_specs = NULL;
    PyObject *expressions = NULL;
    element_fem_space_2d_t *element_fem_space = NULL;
    PyArrayObject *solution = NULL;
    Py_ssize_t stack_memory = 1 << 24;

    static char *kwlist[6] = {"form_specs",         "expressions",   "element_fem_space",
                              "degrees_of_freedom", "stack_memory ", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OO!O!|n", kwlist,
                                     state->type_form_spec, // for type checking
                                     &form_specs,           // _ElementFormsSpecs
                                     &expressions,          // (PyObject*)
                                     state->type_fem_space, // for type checking
                                     &element_fem_space,    // element cache
                                     &PyArray_Type,         // For type checking
                                     &solution,             // np.ndarray
                                     &stack_memory          // int
                                     ))
    {
        return NULL;
    }

    const unsigned order_1 = element_fem_space->basis_xi->order;
    const unsigned order_2 = element_fem_space->basis_eta->order;
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

    const unsigned element_size = element_form_specs_total_count(form_specs, order_1, order_2);
    if (check_input_array((PyArrayObject *)solution, 1, (const npy_intp[1]){(npy_intp)element_size}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "solution") < 0)
    {
        return NULL;
    }
    double *restrict const solution_vec = PyArray_DATA((PyArrayObject *)solution);

    // Create the system template
    system_template_t system_template = {};
    mfv2d_result_t res = MFV2D_SUCCESS;
    if ((res = system_template_create(&system_template, form_specs, element_fem_space, expressions, &SYSTEM_ALLOCATOR,
                                      solution_vec)) != MFV2D_SUCCESS)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Could not parse bytecode into system template, reason: %s",
                                     mfv2d_result_str(res));
        return NULL;
    }

    const npy_intp output_dims[1] = {(npy_intp)element_size};
    return_value = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_DOUBLE);
    if (!return_value)
    {
        system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
        return NULL;
    }

    error_stack_t *const err_stack = error_stack_create(32, &SYSTEM_ALLOCATOR);
    matrix_t *matrix_stack = allocate(&SYSTEM_ALLOCATOR, sizeof *matrix_stack * system_template.max_stack);
    allocator_stack_t *const allocator_stack = allocator_stack_create((size_t)stack_memory, &SYSTEM_ALLOCATOR);
    if (!matrix_stack || !err_stack || !allocator_stack)
    {
        deallocate(&SYSTEM_ALLOCATOR, matrix_stack);
        deallocate(&SYSTEM_ALLOCATOR, err_stack);
        if (allocator_stack)
            deallocate(&SYSTEM_ALLOCATOR, allocator_stack->memory);
        deallocate(&SYSTEM_ALLOCATOR, allocator_stack);
        Py_DECREF(return_value);

        system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);
        return NULL;
    }
    memset(matrix_stack, 0, sizeof *matrix_stack * system_template.max_stack);
    double *restrict const output_mat = PyArray_DATA(return_value);
    memset(output_mat, 0, sizeof *output_mat * element_size);
    unsigned row_offset = 0;
    for (unsigned row = 0; row < system_template.n_forms && res == MFV2D_SUCCESS; ++row)
    {
        const unsigned row_len = form_degrees_of_freedom_count(form_specs->forms[row].order, order_1, order_2);
        size_t col_offset = 0;
        for (unsigned col = 0; col < system_template.n_forms /*&& res == MFV2D_SUCCESS*/; ++col)
        {
            const unsigned col_len = form_degrees_of_freedom_count(form_specs->forms[col].order, order_1, order_2);
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
            res = evaluate_block(err_stack, form_specs->forms[row].order, order_1, bytecode, element_fem_space,
                                 &system_template.fields, system_template.max_stack, matrix_stack,
                                 &allocator_stack->base, &mat, &initial);
            if (res != MFV2D_SUCCESS)
            {
                MFV2D_ERROR(err_stack, res, "Could not evaluate term for block (%u, %u).", row, col);
                break;
            }
            if (row_len != mat.base.rows || 1 != mat.base.cols)
            {
                MFV2D_ERROR(err_stack, MFV2D_DIMS_MISMATCH,
                            "Output matrix arrays don't match expected dims (got %u x %u when needed %u x %u).",
                            mat.base.rows, mat.base.cols, row_len, 1);
                res = MFV2D_DIMS_MISMATCH;
                break;
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

    system_template_destroy(&system_template, &SYSTEM_ALLOCATOR);

    if (res != MFV2D_SUCCESS)
    {
        Py_XDECREF(return_value);
        raise_exception_from_current(PyExc_RuntimeError, "Runtime error %s was encountered.", mfv2d_result_str(res));
        return NULL;
    }
    return (PyObject *)return_value;
}

MFV2D_INTERNAL
const char compute_element_vector_docstr[] =
    "compute_element_vector(form_orders: Sequence[int], expressions: _CompiledCodeMatrix, corners: array, "
    "vector_fields: Sequence[array], basis: Basis2D, solution: array, stack_memory: int = 1 << 24) -> array\n"
    "Compute a single element forcing.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "form_orders : Sequence of int\n"
    "    Orders of differential forms for the degrees of freedom. Must be between 0 and 2.\n"
    "\n"
    "expressions\n"
    "    Compiled bytecode to execute.\n"
    "\n"
    "corners : (4, 2) array\n"
    "    Array of corners of the element.\n"
    "\n"
    "vector_fields : Sequence of arrays\n"
    "    Vector field arrays as required for interior product evaluations.\n"
    "\n"
    "basis : Basis2D\n"
    "    Basis functions with integration rules to use.\n"
    "\n"
    "solution : array\n"
    "    Array with degrees of freedom for the element.\n"
    "\n"
    "stack_memory : int, default: 1 << 24\n"
    "    Amount of memory to use for the evaluation stack.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Element vector for the specified system.\n";

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
    "dual : bool\n"
    "    Should the projection be to dual space of the output space instead\n"
    "    of the primal space.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "tuple of square arrays\n"
    "    Tuple where each entry is the respective projection matrix for that form.\n";

MFV2D_INTERNAL
PyObject *compute_element_projector(PyObject *mod, PyObject *args, PyObject *kwds)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    element_form_spec_t *form_specs;
    basis_2d_t *basis_in;
    element_fem_space_2d_t *space_out;
    int dual = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!|p",
                                     (char *const[]){"form_orders", "basis_in", "space_out", "dual", NULL},
                                     state->type_form_spec, &form_specs, state->type_basis2d, &basis_in,
                                     state->type_fem_space, &space_out, &dual))
    {
        return NULL;
    }

    const unsigned n_forms = Py_SIZE(form_specs);

    const quad_info_t *const quad = &space_out->corners;
    fem_space_2d_t *fem_space_in = NULL;
    const fem_space_2d_t *const fem_space_out = space_out->fem_space;
    {
        const fem_space_1d_t space_xi = basis_1d_as_fem_space(basis_in->basis_xi);
        const fem_space_1d_t space_eta = basis_1d_as_fem_space(basis_in->basis_eta);
        const mfv2d_result_t res = fem_space_2d_create(&space_xi, &space_eta, quad, &fem_space_in, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not create FEM space for input basis.");
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
            if (form_specs->forms[i].order == order)
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

        matrix_full_t out_mat;
        if (!dual)
        {
            const matrix_full_t *mass = NULL;
            switch (order)
            {
            case FORM_ORDER_0:
                mass = element_mass_cache_get_node_inv(space_out);
                break;

            case FORM_ORDER_1:
                mass = element_mass_cache_get_edge_inv(space_out);
                break;

            case FORM_ORDER_2:
                mass = element_mass_cache_get_surf_inv(space_out);
                break;

            default:
                ASSERT(0, "Should never happen.");
            }

            if (mass == NULL)
            {
                PyErr_Format(PyExc_RuntimeError, "Could not compute mass matrix for order %d, error code %s.", order,
                             mfv2d_result_str(res));
                deallocate(&SYSTEM_ALLOCATOR, mat.data);
                goto end;
            }

            res = matrix_full_multiply(mass, &mat, &out_mat, &SYSTEM_ALLOCATOR);
            deallocate(&SYSTEM_ALLOCATOR, mat.data);
            if (res != MFV2D_SUCCESS)
            {
                PyErr_Format(PyExc_RuntimeError, "Could not compute projection matrix of order %d, error code %s.",
                             order, mfv2d_result_str(res));
                goto end;
            }
        }
        else
        {
            out_mat = mat;
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
            if (form_specs->forms[i].order != order)
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

    deallocate(&SYSTEM_ALLOCATOR, fem_space_in);

    return (PyObject *)out;
}

MFV2D_INTERNAL
const char compute_element_mass_matrix_docstr[] =
    "compute_element_mass_matrix(form_orders: _ElementFormSpecification, corners: array, basis: Basis2D, inverse: "
    "bool = False) -> array\n"
    "Compute mass matrix for a given element.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "form_orders : _ElementFormSpecification\n"
    "    Specification of forms in the element.\n"
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

PyObject *compute_element_mass_matrix(PyObject *mod, PyObject *args, PyObject *kwds)
{
    const mfv2d_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    form_order_t order;
    basis_2d_t *basis = NULL;
    PyArrayObject *corners = NULL;
    int inverse = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!|p",
                                     (char *const[5]){"order", "corners", "basis", "inverse", NULL}, &order,
                                     &PyArray_Type, &corners, state->type_basis2d, &basis, &inverse))
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
