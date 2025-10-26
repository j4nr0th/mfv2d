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
        PyArrayObject *const dense_part = (PyArrayObject *)PyTuple_GET_ITEM(block_info, 0);
        crs_matrix_t *const constraints = (crs_matrix_t *)PyTuple_GET_ITEM(block_info, 1);

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
        if (!PyObject_TypeCheck(constraints, &crs_matrix_type_object))
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
    const system_object_t *const this = (const system_object_t *)self;
    return PyUnicode_FromFormat("<LinearSystem (%u DoFs, %u Const.) with %u blocks>", this->total_dense,
                                this->system.n_trace, this->system.n_blocks);
}

PyDoc_STRVAR(system_create_empty_dense_vector_docstring, "create_empty_dense_vector() -> DenseVector\n"
                                                         "Create an empty dense vector associated with the system.\n"
                                                         "\n"
                                                         "Returns\n"
                                                         "-------\n"
                                                         "DenseVector\n"
                                                         "    Dense vector containing all zeros.\n");

static PyObject *system_create_empty_dense_vector(PyObject *self, PyObject *Py_UNUSED(args))
{
    system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;

    // Allocate all the necessary memory
    dense_vector_object_t *const this =
        (dense_vector_object_t *)dense_vector_object_type.tp_alloc(&dense_vector_object_type, 0);
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
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        this->offsets[i] = offset;
        const system_block_t *const block = system->system.blocks + i;
        for (unsigned j = 0; j < block->n; ++j)
        {
            this->values[offset + j] = 0.0;
        }
        offset += block->n;
    }
    this->offsets[n_blocks] = offset;

    return (PyObject *)this;
}

PyDoc_STRVAR(system_create_dense_vector_docstring,
             "create_dense_vector(*blocks: numpy.typing.NDArray[numpy.float64]) -> DenseVector\n"
             "Create a dense vector associated with the system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*blocks : array\n"
             "    Arrays with values of the vector for each block. These must have the\n"
             "    same size as each of the blocks.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "DenseVector\n"
             "    Dense vector representation of the values.\n");

static PyObject *system_create_dense_vector(PyObject *self, PyObject *const *args, const Py_ssize_t n_args)
{
    system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;

    if (n_args != n_blocks)
    {
        PyErr_Format(PyExc_TypeError,
                     "Same number of arguments must be given as there are blocks (%u), but only %u were given.",
                     n_blocks, (unsigned)n_args);
        return NULL;
    }

    // Check args are 1D numpy arrays of doubles with the same size as blocks.
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const system_block_t *const block = system->system.blocks + i;
        if (check_input_array((PyArrayObject *)(args[i]), 1, (const npy_intp[1]){block->n}, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "block value"))
        {
            raise_exception_from_current(PyExc_ValueError, "Block %u was not a contiguous matrix of %u doubles", i,
                                         block->n);
            return NULL;
        }
    }

    // Allocate all the necessary memory
    dense_vector_object_t *const this =
        (dense_vector_object_t *)dense_vector_object_type.tp_alloc(&dense_vector_object_type, 0);
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
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        this->offsets[i] = offset;
        const system_block_t *const block = system->system.blocks + i;
        const npy_double *const data = PyArray_DATA((PyArrayObject *)(args[i]));
        for (unsigned j = 0; j < block->n; ++j)
        {
            this->values[offset + j] = data[j];
        }
        offset += block->n;
    }
    this->offsets[n_blocks] = offset;

    return (PyObject *)this;
}

PyDoc_STRVAR(system_create_trace_vector_docstring,
             "create_trace_vector(value: numpy.typing.NDArray[numpy.float64], /) -> TraceVector\n"
             "Create a trace vector associated with the system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "value : array\n"
             "    Array with values of all trace variables.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "TraceVector\n"
             "    Trace vector representation of the the trace of the system.\n");

static PyObject *system_create_trace_vector(PyObject *self, PyObject *arg)
{
    system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;

    const PyArrayObject *const array = (PyArrayObject *)arg;
    if (check_input_array(array, 1, (const npy_intp[1]){system->system.n_trace}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "trace values"))
    {
        return NULL;
    }

    trace_vector_object_t *const this =
        (trace_vector_object_t *)trace_vector_object_type.tp_alloc(&trace_vector_object_type, 0);
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

    const npy_double *const values = PyArray_DATA(array);

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

    return (PyObject *)this;
}

PyDoc_STRVAR(system_create_empty_trace_vector_docstring,
             "create_empty_trace_vector(value: numpy.typing.NDArray[numpy.float64], /) -> TraceVector\n"
             "Create an empty trace vector associated with the system.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "TraceVector\n"
             "    Trace vector of all zeros.\n");

static PyObject *system_create_empty_trace_vector(PyObject *self, PyObject *Py_UNUSED(args))
{
    system_object_t *const system = (system_object_t *)self;
    const unsigned n_blocks = system->system.n_blocks;

    trace_vector_object_t *const this =
        (trace_vector_object_t *)trace_vector_object_type.tp_alloc(&trace_vector_object_type, 0);
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
    for (unsigned i = 0; i < system->system.n_trace; ++i)
    {
        unsigned count;
        unsigned *indices;
        trace_element_indices(&system->system, i, &count, &indices);

        for (unsigned j = 0; j < count; ++j)
        {
            svector_t *const vec = this->values + indices[j];
            vec->entries[vec->count].value = 0.0;
            vec->entries[vec->count].index = i;
            vec->count += 1;
        }
    }

    return (PyObject *)this;
}

PyDoc_STRVAR(system_get_dense_blocks_docstring,
             "get_system_dense_blocks() -> tuple[numpy.typing.NDArray[numpy.float64], ...]\n"
             "Get the dense blocks of the system.\n");

static PyObject *system_get_dense_blocks(PyObject *self, PyObject *Py_UNUSED(args))
{
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

static PyObject *system_get_constraint_blocks(PyObject *self, PyObject *Py_UNUSED(args))
{
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
        crs_matrix_t *const crs = (crs_matrix_t *)crs_matrix_type_object.tp_alloc(&crs_matrix_type_object, 0);
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

static PyObject *system_apply_diagonal(PyObject *self, PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], &dense_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &dense_vector_object_type))
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

static PyObject *system_apply_diagonal_inverse(PyObject *self, PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], &dense_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &dense_vector_object_type))
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

static PyObject *system_apply_trace(PyObject *self, PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
        return NULL;
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], &dense_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a DenseVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &trace_vector_object_type))
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

static PyObject *system_apply_trace_transpose(PyObject *self, PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
    }

    const system_object_t *const system = (system_object_t *)self;
    if (!PyObject_TypeCheck(args[0], &trace_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected first argument to be a TraceVector, got %s", Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &dense_vector_object_type))
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
        .ml_name = "create_empty_dense_vector",
        .ml_meth = (void *)system_create_empty_dense_vector,
        .ml_flags = METH_NOARGS,
        .ml_doc = system_create_empty_dense_vector_docstring,
    },
    {
        .ml_name = "create_dense_vector",
        .ml_meth = (void *)system_create_dense_vector,
        .ml_flags = METH_FASTCALL,
        .ml_doc = system_create_dense_vector_docstring,
    },
    {
        .ml_name = "create_trace_vector",
        .ml_meth = (void *)system_create_trace_vector,
        .ml_flags = METH_O,
        .ml_doc = system_create_trace_vector_docstring,
    },
    {
        .ml_name = "create_empty_trace_vector",
        .ml_meth = (void *)system_create_empty_trace_vector,
        .ml_flags = METH_NOARGS,
        .ml_doc = system_create_empty_trace_vector_docstring,
    },
    {
        .ml_name = "get_dense_blocks",
        .ml_meth = (void *)system_get_dense_blocks,
        .ml_flags = METH_NOARGS,
        .ml_doc = system_get_dense_blocks_docstring,
    },
    {
        .ml_name = "get_constraint_blocks",
        .ml_meth = (void *)system_get_constraint_blocks,
        .ml_flags = METH_NOARGS,
        .ml_doc = system_get_constraint_blocks_docstring,
    },
    {
        .ml_name = "apply_diagonal",
        .ml_meth = (void *)system_apply_diagonal,
        .ml_flags = METH_FASTCALL,
        .ml_doc = system_apply_diagonal_docstring,
    },
    {
        .ml_name = "apply_diagonal_inverse",
        .ml_meth = (void *)system_apply_diagonal_inverse,
        .ml_flags = METH_FASTCALL,
        .ml_doc = system_apply_diagonal_inverse_docstring,
    },
    {
        .ml_name = "apply_trace",
        .ml_meth = (void *)system_apply_trace,
        .ml_flags = METH_FASTCALL,
        .ml_doc = system_apply_trace_docstring,
    },
    {
        .ml_name = "apply_trace_transpose",
        .ml_meth = (void *)system_apply_trace_transpose,
        .ml_flags = METH_FASTCALL,
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

PyTypeObject system_object_type = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "mfv2d._mfv2d.LinearSystem",
    .tp_basicsize = sizeof(system_object_t),
    .tp_itemsize = 0,
    .tp_str = system_str,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = system_object_docstring,
    .tp_methods = system_methods,
    // .tp_getset = ,
    .tp_new = system_new,
    .tp_dealloc = system_dealloc,
};

static void dense_vector_destroy(PyObject *self)
{
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
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    return PyUnicode_FromFormat("<DenseVector with parent at %p>", this->parent);
}

PyDoc_STRVAR(dense_vector_as_merged_docstring, "as_merged() -> numpy.typing.NDArray[numpy.float64]\n"
                                               "Return the dense vector as a single merged array.\n");

static PyObject *dense_vector_as_merged(PyObject *self, PyObject *Py_UNUSED(args))
{
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

static PyObject *dense_vector_as_split(PyObject *self, PyObject *Py_UNUSED(args))
{
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

PyDoc_STRVAR(dense_vector_subtract_from_docstring, "subtract_from(other: DenseVector, /) -> None\n"
                                                   "Subtract another dense vector from itself.");

static PyObject *dense_vector_subtract_from(PyObject *self, PyObject *other)
{
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    const dense_vector_object_t *const that = (dense_vector_object_t *)other;
    if (this->parent != that->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot subtract vectors from different systems.");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    const unsigned count = this->parent->total_dense;

    for (unsigned i = 0; i < count; ++i)
    {
        this->values[i] -= that->values[i];
    }

    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

PyDoc_STRVAR(dense_vector_copy_docstring, "copy() -> DenseVector\n"
                                          "Create a copy of itself.");

static PyObject *dense_vector_copy(PyObject *self, PyObject *Py_UNUSED(other))
{
    const dense_vector_object_t *const this = (dense_vector_object_t *)self;
    dense_vector_object_t *const res =
        (dense_vector_object_t *)dense_vector_object_type.tp_alloc(&dense_vector_object_type, 0);
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
    const unsigned n_offsets = (this->parent->system.n_blocks + 1);
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

static PyMethodDef dense_vector_methods[] = {
    {
        .ml_name = "as_merged",
        .ml_meth = (void *)dense_vector_as_merged,
        .ml_flags = METH_NOARGS,
        .ml_doc = dense_vector_as_merged_docstring,
    },
    {
        .ml_name = "as_split",
        .ml_meth = (void *)dense_vector_as_split,
        .ml_flags = METH_NOARGS,
        .ml_doc = dense_vector_as_split_docstring,
    },
    {
        .ml_name = "subtract_from",
        .ml_meth = (void *)dense_vector_subtract_from,
        .ml_flags = METH_O,
        .ml_doc = dense_vector_subtract_from_docstring,
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)dense_vector_copy,
        .ml_flags = METH_NOARGS,
        .ml_doc = dense_vector_copy_docstring,
    },
    {}, // sentinel
};

PyDoc_STRVAR(dense_vector_object_type_docstring,
             "Type used to represent the \"dense\" system variables associated with one element.\n");

PyTypeObject dense_vector_object_type = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "mfv2d._mfv2d.DenseVector",
    .tp_basicsize = sizeof(dense_vector_object_t),
    .tp_itemsize = 0,
    .tp_str = dense_vector_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = dense_vector_object_type_docstring,
    .tp_methods = dense_vector_methods,
    // .tp_getset = ,
    .tp_dealloc = dense_vector_destroy,
};

static PyObject *trace_vector_str(PyObject *self)
{
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    return PyUnicode_FromFormat("<TraceVector with parent at %p>", this->parent);
}

static void trace_vector_destroy(PyObject *self)
{
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

static PyObject *trace_vector_as_merged(PyObject *self, PyObject *Py_UNUSED(args))
{
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

static PyObject *trace_vector_as_split(PyObject *self, PyObject *Py_UNUSED(args))
{
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    PyTupleObject *const res = (PyTupleObject *)PyTuple_New(this->parent->system.n_blocks);
    if (!res)
        return NULL;
    for (unsigned i = 0; i < this->parent->system.n_blocks; ++i)
    {
        svec_object_t *const svec = sparse_vector_to_python(this->values + i);
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

static PyObject *trace_vector_dot(PyObject *Py_UNUSED(mod), PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "dot() takes exactly 2 arguments (%u given)", (unsigned)nargs);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &trace_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "dot() first argument must be a TraceVector, not %R", Py_TYPE(args[0]));
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &trace_vector_object_type))
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

        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            const double lhs_value = lhs_vec->entries[j].value;
            const double rhs_value = rhs_vec->entries[j].value;
            ASSERT(lhs_vec->entries[j].index == rhs_vec->entries[j].index,
                   "Trace vector blocks %u don't have matching indices %u (%u vs %u)", i, j,
                   (unsigned)lhs_vec->entries[j].index, (unsigned)rhs_vec->entries[j].index);
            res += lhs_value * rhs_value;
        }
    }

    Py_END_ALLOW_THREADS;
    return PyFloat_FromDouble(res);
}

PyDoc_STRVAR(trace_vector_add_to_docstring, "add_to(other: TraceVector, /) -> None\n"
                                            "Add another trace vector to itself.\n");

static PyObject *trace_vector_add_to(PyObject *self, PyObject *other)
{
    if (!PyObject_TypeCheck(other, &trace_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "add_to() argument must be a TraceVector, not %R", Py_TYPE(other));
        return NULL;
    }
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    const trace_vector_object_t *const that = (trace_vector_object_t *)other;

    if (this->parent != that->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot add vectors from different systems.");
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
            lhs_vec->entries[j].value += rhs_vec->entries[j].value;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(trace_vector_subtract_from_docstring, "subtract_from(other: TraceVector, /) -> None\n"
                                                   "Subtract another trace vector from itself.");

static PyObject *trace_vector_subtract_from(PyObject *self, PyObject *other)
{
    if (!PyObject_TypeCheck(other, &trace_vector_object_type))
    {
        PyErr_Format(PyExc_TypeError, "subtract_from() argument must be a TraceVector, not %R", Py_TYPE(other));
        return NULL;
    }
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    const trace_vector_object_t *const that = (trace_vector_object_t *)other;

    if (this->parent != that->parent)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot subtract vectors from different systems.");
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
            lhs_vec->entries[j].value -= rhs_vec->entries[j].value;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(trace_vector_scale_by_docstring, "scale_by(x: float, /) -> None\n"
                                              "Scale the vector by a value.");

static PyObject *trace_vector_scale_by(PyObject *self, PyObject *other)
{
    const double k = PyFloat_AsDouble(other);
    if (PyErr_Occurred())
        return NULL;

    const trace_vector_object_t *const this = (trace_vector_object_t *)self;

    Py_BEGIN_ALLOW_THREADS;

    const unsigned n_blocks = this->parent->system.n_blocks;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const svector_t *const lhs_vec = this->values + i;

        for (unsigned j = 0; j < lhs_vec->count; ++j)
        {
            lhs_vec->entries[j].value *= k;
        }
    }

    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(trace_vector_copy_docstring, "copy() -> TraceVector\n"
                                          "Create a copy of itself.");

static PyObject *trace_vector_copy(PyObject *self, PyObject *Py_UNUSED(other))
{
    const trace_vector_object_t *const this = (trace_vector_object_t *)self;
    trace_vector_object_t *const res =
        (trace_vector_object_t *)trace_vector_object_type.tp_alloc(&trace_vector_object_type, 0);
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

static PyMethodDef trace_vector_methods[] = {
    {
        .ml_name = "as_merged",
        .ml_meth = (void *)trace_vector_as_merged,
        .ml_flags = METH_NOARGS,
        .ml_doc = trace_vector_as_merged_docstring,
    },
    {
        .ml_name = "as_split",
        .ml_meth = (void *)trace_vector_as_split,
        .ml_flags = METH_NOARGS,
        .ml_doc = trace_vector_as_split_docstring,
    },
    {
        .ml_name = "dot",
        .ml_meth = (void *)trace_vector_dot,
        .ml_flags = METH_FASTCALL | METH_STATIC,
        .ml_doc = trace_vector_dot_docstring,
    },
    {
        .ml_name = "add_to",
        .ml_meth = (void *)trace_vector_add_to,
        .ml_flags = METH_O,
        .ml_doc = trace_vector_add_to_docstring,
    },
    {
        .ml_name = "subtract_from",
        .ml_meth = (void *)trace_vector_subtract_from,
        .ml_flags = METH_O,
        .ml_doc = trace_vector_subtract_from_docstring,
    },
    {
        .ml_name = "scale_by",
        .ml_meth = (void *)trace_vector_scale_by,
        .ml_flags = METH_O,
        .ml_doc = trace_vector_scale_by_docstring,
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)trace_vector_copy,
        .ml_flags = METH_NOARGS,
        .ml_doc = trace_vector_copy_docstring,
    },
    {}, // sentinel
};

PyDoc_STRVAR(trace_vector_object_type_docstring,
             "Type used to represent the \"trace\" system variables associated with constraints.\n");

PyTypeObject trace_vector_object_type = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "mfv2d._mfv2d.TraceVector",
    .tp_basicsize = sizeof(trace_vector_object_t),
    .tp_itemsize = 0,
    .tp_str = trace_vector_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = trace_vector_object_type_docstring,
    .tp_methods = trace_vector_methods,
    // .tp_getset = ,
    .tp_dealloc = trace_vector_destroy,
};
