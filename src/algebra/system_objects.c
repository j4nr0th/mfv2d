#include "system_objects.h"
#include "crs_matrix.h"

static mfv2d_result_t system_fill_out(system_object_t *const this, const unsigned num_blocks,
                                      const unsigned n_trace_vars, const unsigned offsets[static num_blocks + 1],
                                      const unsigned n_trace_indices, const unsigned indices[static n_trace_indices],
                                      const allocator_callbacks *allocator, PyTupleObject *tuple_blocks)
{
    this->system = (system_t){}; // Zero it out first
    this->system.n_blocks = num_blocks;
    this->system.blocks = allocate(allocator, sizeof *this->system.blocks * num_blocks);
    if (!this->system.blocks)
    {
        return MFV2D_FAILED_ALLOC;
    }
    this->system.trace_offsets = allocate(allocator, sizeof *this->system.trace_offsets * (n_trace_vars + 1));
    if (!this->system.trace_offsets)
    {
        return MFV2D_FAILED_ALLOC;
    }
    this->system.trace_values = allocate(allocator, sizeof *this->system.trace_values * n_trace_indices);
    if (!this->system.trace_values)
    {
        return MFV2D_FAILED_ALLOC;
    }

    memcpy(this->system.trace_offsets, offsets, sizeof *this->system.trace_offsets * (n_trace_vars + 1));
    memcpy(this->system.trace_values, indices, sizeof *this->system.trace_values * n_trace_indices);
    memset(this->system.blocks, 0, sizeof *this->system.blocks * num_blocks);
    unsigned total_dense = 0;
    for (unsigned i = 0; i < num_blocks; ++i)
    {
        PyObject *const block_info = PyTuple_GET_ITEM(tuple_blocks, i);
        const PyArrayObject *const dense_part = (PyArrayObject *)PyTuple_GET_ITEM(block_info, 0);
        const crs_matrix_t *const constraints = (crs_matrix_t *)PyTuple_GET_ITEM(block_info, 1);

        system_block_t *const block = this->system.blocks + i;
        block->n = PyArray_DIM(dense_part, 0);
        block->diagonal_block = allocate(allocator, sizeof *block->diagonal_block * block->n * block->n);
        if (!block->diagonal_block)
        {
            return MFV2D_FAILED_ALLOC;
        }
        memcpy(block->diagonal_block, PyArray_DATA(dense_part), sizeof *block->diagonal_block * block->n * block->n);
        if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_copy(constraints->matrix, &block->constraints, &JMTX_ALLOCATOR)))
        {
            return MFV2D_FAILED_ALLOC;
        }
        block->diagonal_lu = allocate(&SYSTEM_ALLOCATOR, sizeof *block->diagonal_lu * block->n * block->n);
        if (!block->diagonal_lu)
        {
            return MFV2D_FAILED_ALLOC;
        }
        block->pivots = allocate(&SYSTEM_ALLOCATOR, sizeof *block->pivots * block->n);
        if (!block->pivots)
        {
            return MFV2D_FAILED_ALLOC;
        }
        const mfv2d_result_t decomp_res =
            decompose_pivoted_lu(block->n, block->diagonal_block, block->diagonal_lu, block->pivots, 1e-12);
        if (decomp_res != MFV2D_SUCCESS)
        {
            return decomp_res;
        }
        total_dense += block->n;
    }
    this->total_dense = total_dense;
    this->system.n_trace = n_trace_vars;

    return MFV2D_SUCCESS;
}

static PyObject *system_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(type);
    if (!state)
        return NULL;

    PyTupleObject *tuple_blocks;
    PyArrayObject *array_offsets;
    PyArrayObject *array_indices;

    // Parse the args
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", (char *[]){"blocks", "offsets", "indices", NULL},
                                     &PyTuple_Type, &tuple_blocks, &PyArray_Type, &array_offsets, &PyArray_Type,
                                     &array_indices))
    {
        return NULL;
    }

    // Check the inputs
    const unsigned num_blocks = PyTuple_GET_SIZE(tuple_blocks);
    if (num_blocks == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Linear system must have at least one block");
        return NULL;
    }

    if (check_input_array(array_offsets, 1, (const npy_intp[1]){0}, NPY_UINT32,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "offsets array"))
    {
        return NULL;
    }

    // Check offsets are sorted
    const unsigned n_trace_vars = PyArray_DIM(array_offsets, 0) - 1;
    const npy_uint32 *const offsets = PyArray_DATA(array_offsets);
    for (unsigned i = 0; i < n_trace_vars; ++i)
    {
        if (offsets[i] > offsets[i + 1])
        {
            PyErr_Format(PyExc_ValueError, "Offsets array must be sorted (entry %u was %u, but entry %u was %u).", i,
                         offsets[i], i + 1, offsets[i + 1]);
            return NULL;
        }
    }

    // const unsigned n_trace = offsets[num_blocks];
    const unsigned n_trace_indices = offsets[n_trace_vars];
    if (check_input_array(array_indices, 1, (const npy_intp[1]){n_trace_indices}, NPY_UINT32,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "indices array"))
    {
        return NULL;
    }

    // Check trace indices are in range
    const uint32_t *const indices = (uint32_t *)PyArray_DATA(array_indices);
    for (unsigned i = 0; i < n_trace_indices; ++i)
    {
        if (indices[i] >= num_blocks)
        {
            PyErr_SetString(PyExc_ValueError, "Indices array must be in range [0, num_blocks)");
            return NULL;
        }
    }

    // Check block values are correct.
    unsigned total_dense = 0;
    for (unsigned i = 0; i < num_blocks; ++i)
    {
        PyObject *const block_info = PyTuple_GET_ITEM(tuple_blocks, i);
        if (!PyTuple_Check(block_info) || PyTuple_GET_SIZE(block_info) != 2)
        {
            PyErr_Format(PyExc_TypeError, "Each block must be a tuple (dense_part, constraints), got %R for block %u",
                         block_info, i);
            return NULL;
        }
        const PyArrayObject *const dense_part = (PyArrayObject *)PyTuple_GET_ITEM(block_info, 0);
        if (check_input_array(dense_part, 2, (const npy_intp[2]){0, 0}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "dense part"))
        {
            return NULL;
        }
        const unsigned block_dim = PyArray_DIM(dense_part, 0);
        total_dense += block_dim;
        if (block_dim != PyArray_DIM(dense_part, 1))
        {
            PyErr_Format(PyExc_ValueError, "Dense part must be square, but block %u had the shape (%u, %u).", i,
                         (unsigned)block_dim, (unsigned)PyArray_DIM(dense_part, 1));
            return NULL;
        }
        const crs_matrix_t *const constraints = (crs_matrix_t *)PyTuple_GET_ITEM(block_info, 1);
        if (!PyObject_TypeCheck(constraints, state->type_crs_matrix))
        {
            PyErr_Format(PyExc_TypeError, "Constraints must be a crs_matrix, got %R for block %u", Py_TYPE(constraints),
                         i);
            return NULL;
        }

        if (constraints->built_rows != constraints->matrix->base.rows)
        {
            PyErr_Format(PyExc_RuntimeError, "Constraint matrix for block %u has not been built for all rows", i);
            return NULL;
        }

        if (constraints->matrix->base.rows != n_trace_vars || constraints->matrix->base.cols != block_dim)
        {
            PyErr_Format(PyExc_ValueError,
                         "Constraint matrix for block %u has the wrong shape (expected (%u, %u), but got (%u, %u)).", i,
                         n_trace_vars, block_dim, constraints->matrix->base.rows, constraints->matrix->base.cols);
            return NULL;
        }
    }

    system_object_t *const this = (system_object_t *)type->tp_alloc(type, 0);
    if (!this)
    {
        return NULL;
    }

    const mfv2d_result_t res = system_fill_out(this, num_blocks, n_trace_vars, offsets, n_trace_indices, indices,
                                               &SYSTEM_ALLOCATOR, tuple_blocks);
    if (res != MFV2D_SUCCESS)
    {
        Py_DECREF(this);
        return NULL;
    }
    return (PyObject *)this;
}

static void system_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    system_object_t *const this = (system_object_t *)self;
    if (this->system.blocks)
    {
        for (unsigned i = 0; i < this->system.n_blocks; ++i)
        {
            deallocate(&SYSTEM_ALLOCATOR, this->system.blocks[i].diagonal_block);
            deallocate(&SYSTEM_ALLOCATOR, this->system.blocks[i].diagonal_lu);
            deallocate(&SYSTEM_ALLOCATOR, this->system.blocks[i].pivots);
            if (this->system.blocks[i].constraints)
                jmtxd_matrix_crs_destroy(this->system.blocks[i].constraints);
            this->system.blocks[i] = (system_block_t){};
        }
        deallocate(&SYSTEM_ALLOCATOR, this->system.blocks);
        this->system.blocks = NULL;
    }
    deallocate(&SYSTEM_ALLOCATOR, this->system.trace_offsets);
    deallocate(&SYSTEM_ALLOCATOR, this->system.trace_values);
    this->system = (system_t){};
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *system_str(PyObject *self)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;

    if (!PyObject_TypeCheck(self, state->type_system))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", state->type_system->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }

    const system_object_t *const this = (const system_object_t *)self;
    return PyUnicode_FromFormat("<LinearSystem (%u DoFs, %u Const.) with %u blocks>", this->total_dense,
                                this->system.n_trace, this->system.n_blocks);
}

PyDoc_STRVAR(system_get_dense_blocks_docstring,
             "get_system_dense_blocks() -> tuple[numpy.typing.NDArray[numpy.float64], ...]\n"
             "Get the dense blocks of the system.\n");

static PyObject *system_get_dense_blocks(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                         const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "Expected no arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;
    PyTupleObject *const res = (PyTupleObject *)PyTuple_New(n_blocks);
    if (!res)
    {
        return NULL;
    }

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const system_block_t *const block = system->system.blocks + i;
        const unsigned n = block->n;
        const npy_intp dims[2] = {n, n};
        PyArrayObject *const dense_part = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!dense_part)
        {
            Py_DECREF(res);
            return NULL;
        }
        npy_double *const data = PyArray_DATA(dense_part);
        for (unsigned j = 0; j < n; ++j)
        {
            for (unsigned k = 0; k < n; ++k)
            {
                data[j * n + k] = block->diagonal_block[j * n + k];
            }
        }
        PyTuple_SET_ITEM(res, i, (PyObject *)dense_part);
    }

    return (PyObject *)res;
}

PyDoc_STRVAR(system_get_constraint_blocks_docstring, "get_system_constraint_blocks() -> tuple[MatrixCRS, ...]\n"
                                                     "Get the constraint blocks of the system.\n");

static PyObject *system_get_constraint_blocks(PyObject *self, PyTypeObject *defining_class,
                                              PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                              PyObject *Py_UNUSED(kwnames))
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "Expected no arguments, got %zd", nargs);
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    const system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;
    PyTupleObject *const res = (PyTupleObject *)PyTuple_New(n_blocks);
    if (!res)
    {
        return NULL;
    }

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const system_block_t *const block = system->system.blocks + i;
        jmtxd_matrix_crs *copy;
        if (!JMTX_SUCCEEDED(jmtxd_matrix_crs_copy(block->constraints, &copy, &JMTX_ALLOCATOR)))
        {
            Py_DECREF(res);
            return NULL;
        }
        crs_matrix_t *const crs = (crs_matrix_t *)state->type_crs_matrix->tp_alloc(state->type_crs_matrix, 0);
        if (!crs)
        {
            jmtxd_matrix_crs_destroy(copy);
            Py_DECREF(res);
            return NULL;
        }
        crs->matrix = copy;
        crs->built_rows = copy->base.rows;
        PyTuple_SET_ITEM(res, i, (PyObject *)crs);
    }

    return (PyObject *)res;
}

PyDoc_STRVAR(system_apply_diagonal_docstring, "apply_diagonal(x: DenseVector, out: DenseVector, /) -> None\n"
                                              "Apply multiplication by the diagonal part of the system.\n"
                                              "\n"
                                              "Parameters\n"
                                              "----------\n"
                                              "DenseVector\n"
                                              "    Dense vector to which this is applied.\n"
                                              "\n"
                                              "DenseVector\n"
                                              "    Dense vector to which the output is returned.\n");

static PyObject *system_apply_diagonal(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, PyObject *kwnames)
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "apply_diagonal() takes no keyword arguments");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected second argument to be a DenseVector, got %s",
                     Py_TYPE(args[1])->tp_name);
        return NULL;
    }
    const dense_vector_object_t *const vec_in = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const vec_out = (dense_vector_object_t *)args[1];
    if (vec_in->parent != system || vec_out->parent != system)
    {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be from the same LinearSystem");
        return NULL;
    }
    mfv2d_result_t res;
    Py_BEGIN_ALLOW_THREADS;

    const dense_vector_t vec_in_alias = dense_vector_object_as_dense_vector(vec_in);
    const dense_vector_t vec_out_alias = dense_vector_object_as_dense_vector(vec_out);
    res = sparse_system_apply_diagonal(system_object_as_system(system), &vec_in_alias, &vec_out_alias);

    Py_END_ALLOW_THREADS;

    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to apply diagonal block: reason %s", mfv2d_result_str(res));
        return NULL;
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(system_apply_diagonal_inverse_docstring,
             "apply_diagonal_inverse(x: DenseVector, out: DenseVector, /) -> None\n"
             "Apply inverse of the diagonal part of the system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "DenseVector\n"
             "    Dense vector to which this is applied.\n"
             "\n"
             "DenseVector\n"
             "    Dense vector to which the output is returned.\n");

static PyObject *system_apply_diagonal_inverse(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, PyObject *kwnames)
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "apply_diagonal_inverse() takes no keyword arguments");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected second argument to be a DenseVector, got %s",
                     Py_TYPE(args[1])->tp_name);
        return NULL;
    }
    const dense_vector_object_t *const vec_in = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const vec_out = (dense_vector_object_t *)args[1];
    if (vec_in->parent != system || vec_out->parent != system)
    {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be from the same LinearSystem");
        return NULL;
    }
    mfv2d_result_t res;
    Py_BEGIN_ALLOW_THREADS;

    const dense_vector_t vec_in_alias = dense_vector_object_as_dense_vector(vec_in);
    const dense_vector_t vec_out_alias = dense_vector_object_as_dense_vector(vec_out);
    res = sparse_system_apply_diagonal_inverse(system_object_as_system(system), &vec_in_alias, &vec_out_alias);

    Py_END_ALLOW_THREADS;

    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to apply diagonal inverse: reason %s", mfv2d_result_str(res));
        return NULL;
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(system_apply_trace_docstring, "apply_trace(x: DenseVector, out: TraceVector, /) -> None\n"
                                           "Apply the trace constraints to the dense vector.\n"
                                           "\n"
                                           "Parameters\n"
                                           "----------\n"
                                           "DenseVector\n"
                                           "    Dense vector to which this is applied.\n"
                                           "\n"
                                           "TraceVector\n"
                                           "    Trace vector to which the output is returned.\n");

static PyObject *system_apply_trace(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                    const Py_ssize_t nargs, PyObject *kwnames)
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "apply_trace() takes no keyword arguments");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a TraceVector, got %s", Py_TYPE(args[1])->tp_name);
        return NULL;
    }
    const dense_vector_object_t *const vec_in = (dense_vector_object_t *)args[0];
    const trace_vector_object_t *const vec_out = (trace_vector_object_t *)args[1];
    if (vec_in->parent != system || vec_out->parent != system)
    {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be from the same LinearSystem");
        return NULL;
    }
    mfv2d_result_t res;
    Py_BEGIN_ALLOW_THREADS;

    const dense_vector_t vec_in_alias = dense_vector_object_as_dense_vector(vec_in);
    const trace_vector_t vec_out_alias = trace_vector_object_as_trace_vector(vec_out);
    res = sparse_system_apply_trace(system_object_as_system(system), &vec_in_alias, &vec_out_alias);

    Py_END_ALLOW_THREADS;

    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to apply trace: reason %s", mfv2d_result_str(res));
        return NULL;
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(system_apply_trace_transpose_docstring,
             "apply_trace_transpose(x: TraceVector, out: DenseVector, /) -> None\n"
             "Apply the transpose of the constraints to the trace vector.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "TraceVector\n"
             "    Trace vector to which this is applied.\n"
             "\n"
             "DenseVector\n"
             "    Dense vector to which the output is returned.\n");

static PyObject *system_apply_trace_transpose(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, PyObject *kwnames)
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "apply_trace_transpose() takes no keyword arguments");
        return NULL;
    }

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a TraceVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[1])->tp_name);
        return NULL;
    }
    const trace_vector_object_t *const vec_in = (trace_vector_object_t *)args[0];
    const dense_vector_object_t *const vec_out = (dense_vector_object_t *)args[1];
    if (vec_in->parent != system || vec_out->parent != system)
    {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be from the same LinearSystem");
        return NULL;
    }
    mfv2d_result_t res;
    Py_BEGIN_ALLOW_THREADS;

    const trace_vector_t vec_in_alias = trace_vector_object_as_trace_vector(vec_in);
    const dense_vector_t vec_out_alias = dense_vector_object_as_dense_vector(vec_out);
    res = sparse_system_apply_trace_transpose(system_object_as_system(system), &vec_in_alias, &vec_out_alias);

    Py_END_ALLOW_THREADS;

    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to apply trace transpose: reason %s", mfv2d_result_str(res));
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef system_methods[] = {
    {
        .ml_name = "get_dense_blocks",
        .ml_meth = (void *)system_get_dense_blocks,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_get_dense_blocks_docstring,
    },
    {
        .ml_name = "get_constraint_blocks",
        .ml_meth = (void *)system_get_constraint_blocks,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_get_constraint_blocks_docstring,
    },
    {
        .ml_name = "apply_diagonal",
        .ml_meth = (void *)system_apply_diagonal,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_apply_diagonal_docstring,
    },
    {
        .ml_name = "apply_diagonal_inverse",
        .ml_meth = (void *)system_apply_diagonal_inverse,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_apply_diagonal_inverse_docstring,
    },
    {
        .ml_name = "apply_trace",
        .ml_meth = (void *)system_apply_trace,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_apply_trace_docstring,
    },
    {
        .ml_name = "apply_trace_transpose",
        .ml_meth = (void *)system_apply_trace_transpose,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = system_apply_trace_transpose_docstring,
    },
    {}, // Sentinel
};

PyDoc_STRVAR(system_object_docstring,
             "LinearSystem(blocks: tuple[tuple[numpy.typing.NDArray[numpy.float64], MatrixCRS], ...], offsets: "
             "numpy.typing.NDArray[numpy.uint32], indices:  numpy.typing.NDArray[numpy.uint32])\n"
             "Class used to represent a linear system with element equations and constraints.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "blocks : tuple of (array, MatrixCRS)\n"
             "    Tuple of blocks that contain the (square) dense matrix diagonal block and the\n"
             "    matrix representing the constraint block.\n"
             "\n"
             "offsets : array\n"
             "    Array of offsets for the array of element indices for each trace variable.\n"
             "\n"
             "indices : array\n"
             "    Packed array of element indices for each trace variable, denoting to which elements\n"
             "    the variable must be added to.\n");

// PyTypeObject system_object_type = {
//     PyVarObject_HEAD_INIT(NULL, 0) //
//         .tp_name = "mfv2d._mfv2d.LinearSystem",
//     .tp_basicsize = sizeof(system_object_t),
//     .tp_itemsize = 0,
//     .tp_str = system_str,
//     .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
//     .tp_doc = system_object_docstring,
//     .tp_methods = system_methods,
//     // .tp_getset = ,
//     .tp_new = system_new,
//     .tp_dealloc = system_dealloc,
// };

static PyType_Slot system_object_slots[] = {
    {.slot = Py_tp_str, .pfunc = system_str},
    {.slot = Py_tp_doc, .pfunc = (void *)system_object_docstring},
    {.slot = Py_tp_methods, .pfunc = system_methods},
    {.slot = Py_tp_new, .pfunc = system_new},
    {.slot = Py_tp_dealloc, .pfunc = system_dealloc},
    {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
    {}, // sentinel
};

PyType_Spec system_object_spec = {
    .name = "mfv2d._mfv2d.LinearSystem",
    .basicsize = sizeof(system_object_t),
    .itemsize = 0,
    .flags =
        Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots = system_object_slots,
};

static PyObject *dense_vector_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(type);
    if (!state)
        return NULL;

    if (kwargs != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector does not accept keyword arguments");
        return NULL;
    }
    if (args == NULL || PyTuple_GET_SIZE(args) == 0)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector requires at least one argument");
    }

    system_object_t *const system = (system_object_t *)PyTuple_GET_ITEM(args, 0);
    if (!PyObject_TypeCheck(system, state->type_system))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a LinearSystem, got %s", Py_TYPE(system)->tp_name);
        return NULL;
    }
    const unsigned n_args = PyTuple_GET_SIZE(args) - 1; // First is the system
    const unsigned n_blocks = system->system.n_blocks;

    if (n_args != 0 && n_args != n_blocks)
    {
        PyErr_Format(PyExc_TypeError,
                     "Same number of arguments must be given as there are blocks (%u), but only %u were given.",
                     n_blocks, (unsigned)n_args);
        return NULL;
    }

    // Check args are 1D numpy arrays of doubles with the same size as blocks.
    if (n_args != 0)
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            const system_block_t *const block = system->system.blocks + i;
            if (check_input_array((PyArrayObject *)PyTuple_GET_ITEM(args, i + 1), 1, (const npy_intp[1]){block->n},
                                  NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "block value"))
            {
                raise_exception_from_current(PyExc_ValueError, "Block %u was not a contiguous matrix of %u doubles", i,
                                             block->n);
                return NULL;
            }
        }
    }

    // Allocate all the necessary memory
    dense_vector_object_t *const this = (dense_vector_object_t *)type->tp_alloc(type, 0);
    if (!this)
    {
        return NULL;
    }
    this->offsets = NULL;
    this->values = NULL;
    this->parent = system;
    Py_INCREF(system);
    this->offsets = allocate(&SYSTEM_ALLOCATOR, sizeof *this->offsets * (n_blocks + 1));
    if (!this->offsets)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->values = allocate(&SYSTEM_ALLOCATOR, sizeof *this->values * system->total_dense);
    if (!this->values)
    {
        Py_DECREF(this);
        return NULL;
    }

    // Fill the data in
    unsigned offset = 0;
    if (n_args != 0)
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            this->offsets[i] = offset;
            const system_block_t *const block = system->system.blocks + i;
            const npy_double *const data = PyArray_DATA((PyArrayObject *)PyTuple_GET_ITEM(args, i + 1));
            for (unsigned j = 0; j < block->n; ++j)
            {
                this->values[offset + j] = data[j];
            }
            offset += block->n;
        }
    }
    else
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            this->offsets[i] = offset;
            const system_block_t *const block = system->system.blocks + i;
            for (unsigned j = 0; j < block->n; ++j)
            {
                this->values[offset + j] = 0;
            }
            offset += block->n;
        }
    }

    this->offsets[n_blocks] = offset;

    return (PyObject *)this;
}

static void dense_vector_destroy(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    dense_vector_object_t *const this = (dense_vector_object_t *)self;
    deallocate(&SYSTEM_ALLOCATOR, this->offsets);
    this->offsets = NULL;
    deallocate(&SYSTEM_ALLOCATOR, this->values);
    this->values = NULL;
    Py_DECREF(this->parent);
    this->parent = NULL;
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *dense_vector_str(PyObject *self)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", state->type_dense_vector->tp_name,
                     Py_TYPE(self)->tp_name);
        return NULL;
    }
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    return PyUnicode_FromFormat("<DenseVector with parent at %p>", this->parent);
}

PyDoc_STRVAR(dense_vector_as_merged_docstring, "as_merged() -> numpy.typing.NDArray[numpy.float64]\n"
                                               "Return the dense vector as a single merged array.\n");

static PyObject *dense_vector_as_merged(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                        const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }
    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "DenseVector.as_merged() takes no arguments");
        return NULL;
    }
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    const npy_intp dim = this->parent->total_dense;
    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!res)
        return NULL;

    npy_double *const data = PyArray_DATA(res);
    for (unsigned i = 0; i < dim; ++i)
    {
        data[i] = this->values[i];
    }
    return (PyObject *)res;
}

PyDoc_STRVAR(dense_vector_as_split_docstring, "as_split() -> tuple[numpy.typing.NDArray[numpy.float64], ...]\n"
                                              "Return the dense vector as a tuple of individual block arrays.\n");

static PyObject *dense_vector_as_split(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                       const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s", defining_class->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }

    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "DenseVector.as_merged() takes no arguments");
        return NULL;
    }
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    PyTupleObject *const res = (PyTupleObject *)PyTuple_New(this->parent->system.n_blocks);
    if (!res)
        return NULL;

    const dense_vector_t dense_vec = dense_vector_object_as_dense_vector(this);
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        unsigned count;
        double *vals;
        element_dense_vector(&dense_vec, i, &count, &vals);
        const npy_intp dim = count;
        PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
        if (!array)
        {
            Py_DECREF(res);
            return NULL;
        }
        npy_double *const ptr = PyArray_DATA(array);
        for (unsigned j = 0; j < count; ++j)
        {
            ptr[j] = vals[j];
        }
        PyTuple_SET_ITEM(res, i, (PyObject *)array);
    }

    return (PyObject *)res;
}

PyDoc_STRVAR(dense_vector_dot_docstring, "dot(v1: DenseVector, v2: DenseVector, /) -> float\n"
                                         "Compute the dot product of two dense vectors.\n");

static PyObject *dense_vector_dot(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector.dot() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "dot() takes exactly 2 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "dot() first argument must be a DenseVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "dot() second argument must be a DenseVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }

    const dense_vector_object_t *const lhs = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const rhs = (dense_vector_object_t *)args[1];
    if (lhs->parent != rhs->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot perform dot product between vectors from different systems.");
        return NULL;
    }

    double res = 0.0;
    Py_BEGIN_ALLOW_THREADS;

    const unsigned n = lhs->parent->total_dense;

#pragma omp simd reduction(+ : res)
    for (unsigned i = 0; i < n; ++i)
    {
        res += lhs->values[i] * rhs->values[i];
    }

    Py_END_ALLOW_THREADS;

    return PyFloat_FromDouble(res);
}

PyDoc_STRVAR(dense_vector_add_docstring,
             "add(v1: DenseVector, v2: DenseVector, out: DenseVector, k: float, /) -> DenseVector\n"
             "Add two dense vectors, potentially scaling one.\n"
             "\n"
             "Result obtained is :math:`\\vec{v}_1 + k \\cdot \\vec{v}_2`. The result is written\n"
             "to the ``out`` vector, which is also returned by the function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "DenseVector\n"
             "    The vector to which the result is written.\n");

static PyObject *dense_vector_add(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector.add() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "add() takes exactly 4 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "First add() argument must be a DenseVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Second add() argument must be a DenseVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[2], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third add() argument must be a DenseVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[3]);
    if (PyErr_Occurred())
        return NULL;

    const dense_vector_object_t *const v1 = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const v2 = (dense_vector_object_t *)args[1];
    const dense_vector_object_t *const out = (dense_vector_object_t *)args[2];

    if (v1->parent != v2->parent || v1->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n = v1->parent->total_dense;

#pragma omp simd
    for (unsigned i = 0; i < n; ++i)
    {
        out->values[i] = v1->values[i] + k * v2->values[i];
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(dense_vector_subtract_docstring,
             "subtract(v1: DenseVector, v2: DenseVector, out: DenseVector, k: float, /) -> DenseVector\n"
             "Subtract two dense vectors, scaling the second one.\n"
             "\n"
             "Result obtained is :math:`\\vec{v}_1 - k \\cdot \\vec{v}_2`. The result is written\n"
             "to the ``out`` vector, which is also returned by the function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "DenseVector\n"
             "    The vector to which the result is written.\n");

static PyObject *dense_vector_subtract(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, const PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector.subtract() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "subtract() takes exactly 4 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "First subtract() argument must be a DenseVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Second subtract() argument must be a DenseVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[2], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third subtract() argument must be a DenseVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[3]);
    if (PyErr_Occurred())
        return NULL;

    const dense_vector_object_t *const v1 = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const v2 = (dense_vector_object_t *)args[1];
    const dense_vector_object_t *const out = (dense_vector_object_t *)args[2];

    if (v1->parent != v2->parent || v1->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n = v1->parent->total_dense;

#pragma omp simd
    for (unsigned i = 0; i < n; ++i)
    {
        out->values[i] = v1->values[i] - k * v2->values[i];
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(dense_vector_scale_docstring, "scale(v: DenseVector, x: float, out: DenseVector, /) -> DenseVector\n"
                                           "Scale the vector by a value.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "DenseVector\n"
                                           "The vector to which the result is written.\n");

static PyObject *dense_vector_scale(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                    const Py_ssize_t nargs, const PyObject *kwnames)
{

    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector.scale() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 3)
    {
        PyErr_Format(PyExc_TypeError, "scale() takes exactly 3 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "First scale() argument must be a DenseVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[1]);
    if (PyErr_Occurred())
        return NULL;
    if (!PyObject_TypeCheck(args[2], state->type_dense_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third scale() argument must be a DenseVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }

    const dense_vector_object_t *const v1 = (dense_vector_object_t *)args[0];
    const dense_vector_object_t *const out = (dense_vector_object_t *)args[2];

    if (v1->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n = v1->parent->total_dense;

#pragma omp simd
    for (unsigned i = 0; i < n; ++i)
    {
        out->values[i] = k * v1->values[i];
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(dense_vector_copy_docstring, "copy() -> DenseVector\n"
                                          "Create a copy of itself.");

static PyObject *dense_vector_copy(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                   const Py_ssize_t nargs, const PyObject *Py_UNUSED(kwnames))
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "set() argument must be called on a TraceVector, not %R", Py_TYPE(self));
        return NULL;
    }

    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "DenseVector.copy() takes no arguments");
        return NULL;
    }
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;

    dense_vector_object_t *const res = (dense_vector_object_t *)defining_class->tp_alloc(defining_class, 0);
    if (!res)
        return NULL;
    res->parent = this->parent;
    Py_INCREF(this->parent);
    res->values = NULL;
    res->offsets = NULL;
    res->values = allocate(&SYSTEM_ALLOCATOR, sizeof *res->values * this->parent->total_dense);
    if (!res->values)
    {
        Py_DECREF(res);
        return NULL;
    }
    const unsigned n_offsets = this->parent->system.n_blocks + 1;
    res->offsets = allocate(&SYSTEM_ALLOCATOR, sizeof *res->offsets * n_offsets);
    if (!res->offsets)
    {
        Py_DECREF(res);
        return NULL;
    }
    memcpy(res->offsets, this->offsets, sizeof *res->offsets * n_offsets);
    memcpy(res->values, this->values, sizeof *res->values * this->parent->total_dense);
    return (PyObject *)res;
}

PyDoc_STRVAR(dense_vector_set_from_docstring, "set(other: DenseVector) -> None\n"
                                              "Sets the value of the vector from another.");

static PyObject *dense_vector_set_from(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, const PyObject *kwnames)
{
    if (nargs != 1)
    {
        PyErr_Format(PyExc_TypeError, "DenseVector.set() takes exactly 1 argument (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "DenseVector.set() does not accept keyword arguments");
        return NULL;
    }
    PyObject *const other = args[0];

    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "set() argument must be called on a TraceVector, not %R", Py_TYPE(self));
        return NULL;
    }
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    if (!PyObject_TypeCheck(other, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "set() argument must be a TraceVector, not %R", Py_TYPE(other));
        return NULL;
    }
    const dense_vector_object_t *const that = (dense_vector_object_t *)other;
    if (this->parent != that->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot set vectors from different systems.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n = this->parent->total_dense;
#pragma omp simd
    for (unsigned i = 0; i < n; ++i)
    {
        this->values[i] = that->values[i];
    }

    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyMethodDef dense_vector_methods[] = {
    {
        .ml_name = "as_merged",
        .ml_meth = (void *)dense_vector_as_merged,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = dense_vector_as_merged_docstring,
    },
    {
        .ml_name = "as_split",
        .ml_meth = (void *)dense_vector_as_split,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = dense_vector_as_split_docstring,
    },
    {
        .ml_name = "dot",
        .ml_meth = (void *)dense_vector_dot,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = dense_vector_dot_docstring,
    },
    {
        .ml_name = "add",
        .ml_meth = (void *)dense_vector_add,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = dense_vector_add_docstring,
    },
    {
        .ml_name = "subtract",
        .ml_meth = (void *)dense_vector_subtract,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = dense_vector_subtract_docstring,
    },
    {
        .ml_name = "scale",
        .ml_meth = (void *)dense_vector_scale,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = dense_vector_scale_docstring,
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)dense_vector_copy,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = dense_vector_copy_docstring,
    },
    {
        .ml_name = "set_from",
        .ml_meth = (void *)dense_vector_set_from,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = dense_vector_set_from_docstring,
    },
    {}, // sentinel
};

static PyObject *dense_vector_get_parent(const dense_vector_object_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->parent);
    return (PyObject *)self->parent;
}

static PyGetSetDef dense_vector_getset[] = {
    {
        .name = "parent",
        .get = (getter)dense_vector_get_parent,
        .doc = "LinearSystem : The LinearSystem object that this DenseVector belongs to.",
    },
    {}, // Sentinel
};

PyDoc_STRVAR(dense_vector_object_type_docstring,
             "Type used to represent the \"dense\" system variables associated with one element.\n");

// PyTypeObject dense_vector_object_type = {
//     PyVarObject_HEAD_INIT(NULL, 0) //
//         .tp_name = "mfv2d._mfv2d.DenseVector",
//     .tp_basicsize = sizeof(dense_vector_object_t),
//     .tp_itemsize = 0,
//     .tp_str = dense_vector_str,
//     .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_BASETYPE,
//     .tp_doc = dense_vector_object_type_docstring,
//     .tp_methods = dense_vector_methods,
//     .tp_getset = dense_vector_getset,
//     .tp_dealloc = dense_vector_destroy,
//     .tp_new = dense_vector_new,
// };

static PyType_Slot dense_vector_slots[] = {
    {.slot = Py_tp_str, .pfunc = dense_vector_str},
    {.slot = Py_tp_doc, .pfunc = (void *)dense_vector_object_type_docstring},
    {.slot = Py_tp_methods, .pfunc = dense_vector_methods},
    {.slot = Py_tp_getset, .pfunc = dense_vector_getset},
    {.slot = Py_tp_dealloc, .pfunc = dense_vector_destroy},
    {.slot = Py_tp_new, .pfunc = dense_vector_new},
    {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
    {}, // sentinel
};

PyType_Spec dense_vector_object_type_spec = {
    .name = "mfv2d._mfv2d.DenseVector",
    .basicsize = sizeof(dense_vector_object_t),
    .itemsize = 0,
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots = dense_vector_slots,
};

static PyObject *trace_vector_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(type);
    if (!state)
        return NULL;

    if (kwargs != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector does not accept keyword arguments");
        return NULL;
    }
    if (args == NULL || (PyTuple_GET_SIZE(args) != 1 && PyTuple_GET_SIZE(args) != 2))
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector requires one or two arguments");
    }

    system_object_t *const system = (system_object_t *)PyTuple_GET_ITEM(args, 0);
    if (!PyObject_TypeCheck(system, state->type_system))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a LinearSystem, got %s", Py_TYPE(system)->tp_name);
        return NULL;
    }

    const unsigned n_blocks = system->system.n_blocks;
    const npy_double *values = NULL;
    if (PyTuple_GET_SIZE(args) == 2)
    {
        const PyArrayObject *const array = (PyArrayObject *)PyTuple_GET_ITEM(args, 1);
        if (check_input_array(array, 1, (const npy_intp[1]){system->system.n_trace}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "trace values"))
        {
            return NULL;
        }
        values = PyArray_DATA(array);
    }

    trace_vector_object_t *const this = (trace_vector_object_t *)type->tp_alloc(type, 0);
    if (!this)
    {
        return NULL;
    }
    this->values = NULL;
    this->parent = system;
    Py_INCREF(system);
    this->values = allocate(&SYSTEM_ALLOCATOR, sizeof *this->values * n_blocks);
    if (!this->values)
    {
        Py_DECREF(this);
        return NULL;
    }
    // Zero it out to make the state valid
    memset(this->values, 0, sizeof *this->values * n_blocks);

    // Count up all the unknowns
    for (unsigned i = 0; i < system->system.n_trace; ++i)
    {
        unsigned count;
        unsigned *indices;
        trace_element_indices(&system->system, i, &count, &indices);
        for (unsigned j = 0; j < count; ++j)
        {
            this->values[indices[j]].count += 1;
        }
    }

    // Allocate all the vectors
    for (unsigned i = 0; i < system->system.n_blocks; ++i)
    {
        svector_t *const vec = this->values + i;
        const mfv2d_result_t res = sparse_vector_new(vec, system->system.n_trace, vec->count, &SYSTEM_ALLOCATOR);
        if (res != MFV2D_SUCCESS)
        {
            Py_DECREF(this);
            return NULL;
        }
    }

    // Fill all the vectors
    if (values != NULL)
    {
        for (unsigned i = 0; i < system->system.n_trace; ++i)
        {
            unsigned count;
            unsigned *indices;
            trace_element_indices(&system->system, i, &count, &indices);

            for (unsigned j = 0; j < count; ++j)
            {
                svector_t *const vec = this->values + indices[j];
                vec->entries[vec->count].value = values[i];
                vec->entries[vec->count].index = i;
                vec->count += 1;
            }
        }
    }
    else
    {

        for (unsigned i = 0; i < system->system.n_trace; ++i)
        {
            unsigned count;
            unsigned *indices;
            trace_element_indices(&system->system, i, &count, &indices);

            for (unsigned j = 0; j < count; ++j)
            {
                svector_t *const vec = this->values + indices[j];
                vec->entries[vec->count].value = 0;
                vec->entries[vec->count].index = i;
                vec->count += 1;
            }
        }
    }

    return (PyObject *)this;
}

static PyObject *trace_vector_str(PyObject *self)
{
    const mfv2d_module_state_t *const state = mfv2d_state_from_type(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Expected a TraceVector, got %s", Py_TYPE(self)->tp_name);
        return NULL;
    }
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    return PyUnicode_FromFormat("<TraceVector with parent at %p>", this->parent);
}

static void trace_vector_destroy(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    trace_vector_object_t *const this = (trace_vector_object_t *)self;
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        sparse_vector_del(this->values + i, &SYSTEM_ALLOCATOR);
        this->values[i] = (svector_t){};
    }
    deallocate(&SYSTEM_ALLOCATOR, this->values);
    Py_DECREF(this->parent);
    this->parent = NULL;
    Py_TYPE(this)->tp_free((PyObject *)this);
}

PyDoc_STRVAR(trace_vector_as_merged_docstring, "as_merged() -> numpy.typing.NDArray[numpy.float64]\n"
                                               "Return the trace vector as a single merged array.");

static PyObject *trace_vector_as_merged(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                        const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "TraceVector.as_merged() must be called on a TraceVector, not %R", Py_TYPE(self));
        return NULL;
    }

    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "TraceVector.as_merged() takes no arguments");
        return NULL;
    }

    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    const npy_intp dim = this->parent->system.n_trace;
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!array)
        return NULL;

    npy_double *const ptr = PyArray_DATA(array);
    memset(ptr, 0, sizeof *ptr * dim);
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        const svector_t *const vec = this->values + i;
        for (unsigned j = 0; j < vec->count; ++j)
        {
            ptr[vec->entries[j].index] = vec->entries[j].value;
        }
    }

    return (PyObject *)array;
}

PyDoc_STRVAR(trace_vector_as_split_docstring, "as_split() -> tuple[SparseVector, ...]\n"
                                              "Return the trace vector as a tuple of individual block traces.\n");

static PyObject *trace_vector_as_split(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                       const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    if (!PyObject_TypeCheck(self, defining_class))
    {
        PyErr_Format(PyExc_TypeError, "TraceVector.as_split() argument must be called on a TraceVector, not %R",
                     Py_TYPE(self));
        return NULL;
    }
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);

    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "TraceVector.as_split() takes no arguments");
        return NULL;
    }

    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    PyTupleObject *const res = (PyTupleObject *)PyTuple_New(this->parent->system.n_blocks);
    if (!res)
        return NULL;
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        svec_object_t *const svec = sparse_vector_to_python(state->type_svec, this->values + i);
        if (!svec)
        {
            Py_DECREF(res);
            return NULL;
        }

        PyTuple_SET_ITEM(res, i, (PyObject *)svec);
    }

    return (PyObject *)res;
}

PyDoc_STRVAR(trace_vector_dot_docstring, "dot(v1: TraceVector, v2: TraceVector, /) -> float\n"
                                         "Compute the dot product of two trace vectors.\n");

static PyObject *trace_vector_dot(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector.dot() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "dot() takes exactly 2 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "dot() first argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "dot() second argument must be a TraceVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }

    const trace_vector_object_t *const lhs = (trace_vector_object_t *)args[0];
    const trace_vector_object_t *const rhs = (trace_vector_object_t *)args[1];
    if (lhs->parent != rhs->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot perform dot product between vectors from different systems.");
        return NULL;
    }

    double res = 0.0;
    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = lhs->parent->system.n_blocks;
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = lhs->values + i;
        const svector_t *const rhs_vec = rhs->values + i;
        ASSERT(lhs_vec->count == rhs_vec->count, "Trace vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)rhs_vec->count);

#pragma omp simd reduction(+ : res)
        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            const double lhs_value = lhs_vec->entries[j].value;
            const double rhs_value = rhs_vec->entries[j].value;
            const unsigned trace_idx = lhs_vec->entries[j].index;
            ASSERT(trace_idx == rhs_vec->entries[j].index,
                   "Trace vector blocks %u don't have matching indices %u (%u vs %u)", i, j, (unsigned)trace_idx,
                   (unsigned)rhs_vec->entries[j].index);
            const unsigned cnt = // NOTE: consider caching this inverse
                lhs->parent->system.trace_offsets[trace_idx + 1] - lhs->parent->system.trace_offsets[trace_idx];
            // If the unknown is in multiple elements, it will be added to the result multiple times, so divide that out
            res += lhs_value * rhs_value / (double)cnt;
        }
    }

    Py_END_ALLOW_THREADS;
    return PyFloat_FromDouble(res);
}

PyDoc_STRVAR(trace_vector_add_docstring,
             "add(v1: TraceVector, v2: TraceVector, out: TraceVector, k: float, /) -> TraceVector\n"
             "Add two trace vectors, potentially scaling one.\n"
             "\n"
             "Result obtained is :math:`\\vec{v}_1 + k \\cdot \\vec{v}_2`. The result is written\n"
             "to the ``out`` vector, which is also returned by the function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "TraceVector\n"
             "    The vector to which the result is written.\n");

static PyObject *trace_vector_add(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector.add() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "add() takes exactly 4 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "First add() argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Second add() argument must be a TraceVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[2], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third add() argument must be a TraceVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[3]);
    if (PyErr_Occurred())
        return NULL;

    const trace_vector_object_t *const v1 = (trace_vector_object_t *)args[0];
    const trace_vector_object_t *const v2 = (trace_vector_object_t *)args[1];
    const trace_vector_object_t *const out = (trace_vector_object_t *)args[2];

    if (v1->parent != v2->parent || v1->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = v1->parent->system.n_blocks;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = v1->values + i;
        const svector_t *const rhs_vec = v2->values + i;
        const svector_t *const out_vec = out->values + i;
        ASSERT(lhs_vec->count == rhs_vec->count, "Trace vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)rhs_vec->count);
        ASSERT(lhs_vec->count == out_vec->count, "Output vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)out_vec->count);

#pragma omp simd
        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            ASSERT(lhs_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Trace vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)lhs_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            ASSERT(out_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Output vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)out_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            out_vec->entries[j].value = lhs_vec->entries[j].value + k * rhs_vec->entries[j].value;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(trace_vector_subtract_docstring,
             "subtract(v1: TraceVector, v2: TraceVector, out: TraceVector, k: float, /) ->TraceVector\n"
             "Subtract two trace vectors, scaling the second one.\n"
             "\n"
             "Result obtained is :math:`\\vec{v}_1 - k \\cdot \\vec{v}_2`. The result is written\n"
             "to the ``out`` vector, which is also returned by the function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "TraceVector\n"
             "    The vector to which the result is written.\n");

static PyObject *trace_vector_subtract(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector.subtract() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "subtract() takes exactly 4 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "First subtract() argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Second subtract() argument must be a TraceVector, not %R", Py_TYPE(args[1]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[2], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third subtract() argument must be a TraceVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[3]);
    if (PyErr_Occurred())
        return NULL;

    const trace_vector_object_t *const v1 = (trace_vector_object_t *)args[0];
    const trace_vector_object_t *const v2 = (trace_vector_object_t *)args[1];
    const trace_vector_object_t *const out = (trace_vector_object_t *)args[2];

    if (v1->parent != v2->parent || v1->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = v1->parent->system.n_blocks;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = v1->values + i;
        const svector_t *const rhs_vec = v2->values + i;
        const svector_t *const out_vec = out->values + i;
        ASSERT(lhs_vec->count == rhs_vec->count, "Trace vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)rhs_vec->count);
        ASSERT(lhs_vec->count == out_vec->count, "Output vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)out_vec->count);

#pragma omp simd
        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            ASSERT(lhs_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Trace vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)lhs_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            ASSERT(out_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Output vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)out_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            out_vec->entries[j].value = lhs_vec->entries[j].value - k * rhs_vec->entries[j].value;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(trace_vector_scale_docstring, "scale(v1: TraceVector, x: float, out: TraceVector, /) -> TraceVector\n"
                                           "Scale the vector by a value.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "TraceVector\n"
                                           "    The vector to which the result is written.\n");

static PyObject *trace_vector_scale(PyObject *Py_UNUSED(self), PyTypeObject *defining_class, PyObject *const *args,
                                    const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "TraceVector.scale() does not accept keyword arguments");
        return NULL;
    }

    if (nargs != 3)
    {
        PyErr_Format(PyExc_TypeError, "scale() takes exactly 3 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "First scale() argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    const double k = PyFloat_AsDouble(args[1]);
    if (PyErr_Occurred())
        return NULL;
    if (!PyObject_TypeCheck(args[2], state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "Third scale() argument must be a TraceVector, not %R", Py_TYPE(args[2]));
        return NULL;
    }

    const trace_vector_object_t *const this = (trace_vector_object_t *)args[0];
    const trace_vector_object_t *const out = (trace_vector_object_t *)args[2];

    if (this->parent != out->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Both input and output vectors must have the same parent system.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = this->parent->system.n_blocks;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = this->values + i;
        const svector_t *const out_vec = out->values + i;
#pragma omp simd
        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            ASSERT(out_vec->entries[j].index == lhs_vec->entries[j].index,
                   "Output vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)out_vec->entries[j].index, (unsigned)lhs_vec->entries[j].index);
            out_vec->entries[j].value = lhs_vec->entries[j].value * k;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_INCREF(out);
    return (PyObject *)out;
}

PyDoc_STRVAR(trace_vector_copy_docstring, "copy() -> TraceVector\n"
                                          "Create a copy of itself.");

static PyObject *trace_vector_copy(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                   const Py_ssize_t nargs, PyObject *Py_UNUSED(kwnames))
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (nargs != 0)
    {
        PyErr_Format(PyExc_TypeError, "copy() takes no arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "copy() argument must be a TraceVector, not %R", Py_TYPE(self));
        return NULL;
    }

    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    PyTypeObject *const type = state->type_trace_vector;
    trace_vector_object_t *const res = (trace_vector_object_t *)type->tp_alloc(type, 0);
    if (!res)
        return NULL;
    res->parent = this->parent;
    res->values = NULL;
    Py_INCREF(this->parent);
    res->values = allocate(&SYSTEM_ALLOCATOR, this->parent->system.n_blocks * sizeof(svector_t));
    if (!res->values)
    {
        Py_DECREF(res);
    }
    memset(res->values, 0, this->parent->system.n_blocks * sizeof(svector_t));
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        const svector_t *const src = this->values + i;
        svector_t *const dst = res->values + i;
        const mfv2d_result_t mfv2d_result = sparse_vector_copy(src, dst, &SYSTEM_ALLOCATOR);
        if (mfv2d_result != MFV2D_SUCCESS)
        {
            Py_DECREF(res);
            return NULL;
        }
    }

    return (PyObject *)res;
}

PyDoc_STRVAR(trace_vector_set_from_docstring, "set(other: TraceVector) -> None\n"
                                              "Sets the value of the vector from another.");

static PyObject *trace_vector_set_from(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, PyObject *kwnames)
{
    const mfv2d_module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;
    if (nargs != 1)
    {
        PyErr_Format(PyExc_TypeError, "set() takes exactly 1 argument (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "set() does not accept keyword arguments");
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->type_trace_vector))
    {
        PyErr_Format(PyExc_TypeError, "set() argument must be a TraceVector, not %R", Py_TYPE(self));
        return NULL;
    }
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;

    if (!PyObject_TypeCheck(args[0], Py_TYPE(this)))
    {
        PyErr_Format(PyExc_TypeError, "set() argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    const trace_vector_object_t *const that = (trace_vector_object_t *)args[0];
    if (this->parent != that->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot set vectors from different systems.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = this->parent->system.n_blocks;
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = this->values + i;
        const svector_t *const rhs_vec = that->values + i;
        ASSERT(lhs_vec->count == rhs_vec->count, "Trace vector blocks %u don't have matching sizes (%u vs %u)", i,
               (unsigned)lhs_vec->count, (unsigned)rhs_vec->count);

        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            ASSERT(lhs_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Trace vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)lhs_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            lhs_vec->entries[j].value = rhs_vec->entries[j].value;
        }
    }

    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyMethodDef trace_vector_methods[] = {
    {
        .ml_name = "as_merged",
        .ml_meth = (void *)trace_vector_as_merged,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = trace_vector_as_merged_docstring,
    },
    {
        .ml_name = "as_split",
        .ml_meth = (void *)trace_vector_as_split,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = trace_vector_as_split_docstring,
    },
    {
        .ml_name = "dot",
        .ml_meth = (void *)trace_vector_dot,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = trace_vector_dot_docstring,
    },
    {
        .ml_name = "add",
        .ml_meth = (void *)trace_vector_add,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = trace_vector_add_docstring,
    },
    {
        .ml_name = "subtract",
        .ml_meth = (void *)trace_vector_subtract,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = trace_vector_subtract_docstring,
    },
    {
        .ml_name = "scale",
        .ml_meth = (void *)trace_vector_scale,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = trace_vector_scale_docstring,
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)trace_vector_copy,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = trace_vector_copy_docstring,
    },
    {
        .ml_name = "set_from",
        .ml_meth = (void *)trace_vector_set_from,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = trace_vector_set_from_docstring,
    },
    {}, // sentinel
};

static PyObject *trace_vector_get_parent(const trace_vector_object_t *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->parent);
    return (PyObject *)self->parent;
}

static PyGetSetDef trace_vector_getset[] = {
    {
        .name = "parent",
        .get = (getter)trace_vector_get_parent,
        .doc = "LinearSystem : The LinearSystem object that this TraceVector belongs to.",
    },
    {}, // Sentinel
};

PyDoc_STRVAR(trace_vector_object_type_docstring,
             "Type used to represent the \"trace\" system variables associated with constraints.\n");

static PyType_Slot trace_vector_slots[] = {
    {.slot = Py_tp_str, .pfunc = trace_vector_str},
    {.slot = Py_tp_doc, .pfunc = (void *)trace_vector_object_type_docstring},
    {.slot = Py_tp_methods, .pfunc = trace_vector_methods},
    {.slot = Py_tp_getset, .pfunc = trace_vector_getset},
    {.slot = Py_tp_new, .pfunc = trace_vector_new},
    {.slot = Py_tp_dealloc, .pfunc = trace_vector_destroy},
    {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
    {.slot = Py_tp_traverse, .pfunc = traverse_heap_type},
    {}, // sentinel
};

PyType_Spec trace_vector_type_spec = {
    .name = "mfv2d._mfv2d.TraceVector",
    .basicsize = sizeof(trace_vector_object_t),
    .itemsize = 0,
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots = trace_vector_slots,
};
