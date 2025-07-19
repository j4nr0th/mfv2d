#include "mesh.h"

enum
{
    PARENT_IDX_INVALID = ~0
};

static mfv2d_result_t element_mesh_create(element_mesh_t *this, const allocator_callbacks *allocator,
                                          const unsigned n_elements, const quad_info_t quads[static n_elements],
                                          const index_2d_t orders[static n_elements])
{
    this->allocator = allocator;
    this->count = n_elements;
    this->capacity = n_elements;
    this->leaf_count = n_elements;
    this->elements = allocate(allocator, sizeof *this->elements * this->capacity);
    if (!this->elements)
    {
        return MFV2D_FAILED_ALLOC;
    }

    for (unsigned i = 0; i < n_elements; ++i)
    {
        element_leaf_t *const elem = &this->elements[i].leaf;

        *elem = (element_leaf_t){
            .base = (element_base_t){.type = ELEMENT_TYPE_LEAF, .parent = PARENT_IDX_INVALID},
            .data = {.orders = orders[i], .corners = quads[i]},
        };
    }

    return MFV2D_SUCCESS;
}

static void element_mesh_destroy(element_mesh_t *const this)
{
    deallocate(this->allocator, this->elements);
    *this = (element_mesh_t){};
}

static mfv2d_result_t element_mesh_split_element(element_mesh_t *this, const unsigned index,
                                                 const index_2d_t orders_bottom_left,
                                                 const index_2d_t orders_bottom_right,
                                                 const index_2d_t orders_top_right, const index_2d_t orders_top_left)
{
    // Check index and type make sense
    if (this->count <= index)
    {
        return MFV2D_INDEX_OUT_OF_RANGE;
    }

    if (this->elements[index].base.type != ELEMENT_TYPE_LEAF)
    {
        return MFV2D_NOT_A_LEAF;
    }

    // Check if we have enough space to split the element
    if (this->capacity <= this->count + 3)
    {
        const unsigned new_capacity = (this->capacity + 4) * 2;
        element_t *const new_elements =
            reallocate(this->allocator, this->elements, sizeof *new_elements * new_capacity);
        if (!new_elements)
        {
            return MFV2D_FAILED_ALLOC;
        }
        this->elements = new_elements;
        this->capacity = new_capacity;
    }

    // First update the element itself
    const element_leaf_t base_element = this->elements[index].leaf;
    const quad_info_t base_quad = base_element.data.corners;
    this->elements[index].node = (element_node_t){
        .base =
            (element_base_t){
                .type = ELEMENT_TYPE_NODE,
                .parent = base_element.base.parent,
            },
        .child_bottom_left = this->count + 0,
        .child_bottom_right = this->count + 1,
        .child_top_right = this->count + 2,
        .child_top_left = this->count + 3,
    };

    const double x_bottom_middle = (base_quad.x0 + base_quad.x1) / 2;
    const double y_bottom_middle = (base_quad.y0 + base_quad.y1) / 2;

    const double x_top_middle = (base_quad.x2 + base_quad.x3) / 2;
    const double y_top_middle = (base_quad.y2 + base_quad.y3) / 2;

    const double x_left_middle = (base_quad.x0 + base_quad.x3) / 2;
    const double y_left_middle = (base_quad.y0 + base_quad.y3) / 2;

    const double x_right_middle = (base_quad.x1 + base_quad.x2) / 2;
    const double y_right_middle = (base_quad.y1 + base_quad.y2) / 2;

    const double x_middle = (base_quad.x0 + base_quad.x1 + base_quad.x2 + base_quad.x3) / 4;
    const double y_middle = (base_quad.y0 + base_quad.y1 + base_quad.y2 + base_quad.y3) / 4;

    // Prepare the child elements
    this->elements[this->count + 0].leaf = (element_leaf_t){
        .base =
            (element_base_t){
                .type = ELEMENT_TYPE_LEAF,
                .parent = index,
            },
        .data =
            (leaf_data){
                .orders = orders_bottom_left,
                .corners =
                    (quad_info_t){
                        .x0 = base_quad.x0,
                        .y0 = base_quad.y0,
                        .x1 = x_bottom_middle,
                        .y1 = y_bottom_middle,
                        .x2 = x_middle,
                        .y2 = y_middle,
                        .x3 = x_left_middle,
                        .y3 = y_left_middle,
                    },
            },
    };

    this->elements[this->count + 1].leaf = (element_leaf_t){
        .base =
            (element_base_t){
                .type = ELEMENT_TYPE_LEAF,
                .parent = index,
            },
        .data =
            (leaf_data){
                .orders = orders_bottom_right,
                .corners =
                    (quad_info_t){
                        .x0 = x_bottom_middle,
                        .y0 = y_bottom_middle,
                        .x1 = base_quad.x1,
                        .y1 = base_quad.y1,
                        .x2 = x_right_middle,
                        .y2 = y_right_middle,
                        .x3 = x_middle,
                        .y3 = y_middle,
                    },
            },
    };

    this->elements[this->count + 2].leaf = (element_leaf_t){
        .base =
            (element_base_t){
                .type = ELEMENT_TYPE_LEAF,
                .parent = index,
            },
        .data =
            {
                .orders = orders_top_right,
                .corners =
                    (quad_info_t){
                        .x0 = x_middle,
                        .y0 = y_middle,
                        .x1 = x_right_middle,
                        .y1 = y_right_middle,
                        .x2 = base_quad.x2,
                        .y2 = base_quad.y2,
                        .x3 = x_top_middle,
                        .y3 = y_top_middle,
                    },
            },
    };

    this->elements[this->count + 3].leaf = (element_leaf_t){
        .base =
            (element_base_t){
                .type = ELEMENT_TYPE_LEAF,
                .parent = index,
            },
        .data =
            {
                .orders = orders_top_left,
                .corners =
                    (quad_info_t){
                        .x0 = x_left_middle,
                        .y0 = y_left_middle,
                        .x1 = x_middle,
                        .y1 = y_middle,
                        .x2 = x_top_middle,
                        .y2 = y_top_middle,
                        .x3 = base_quad.x3,
                        .y3 = base_quad.y3,
                    },
            },
    };

    this->leaf_count += 3;
    this->count += 4;

    return MFV2D_SUCCESS;
}

static mfv2d_result_t element_mesh_copy_element(const element_mesh_t *const src, element_mesh_t *const dst)
{
    element_t *const elements = allocate(&SYSTEM_ALLOCATOR, src->count * sizeof *elements);
    if (elements == NULL)
        return MFV2D_FAILED_ALLOC;

    memcpy(elements, src->elements, src->count * sizeof *elements);
    *dst = *src;
    dst->elements = elements;
    return MFV2D_SUCCESS;
}

static PyObject *mesh_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    manifold2d_object_t *primal, *dual;
    PyArrayObject *corners;
    PyArrayObject *orders;
    PyArrayObject *boundary;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!",
                                     (char *[6]){"primal", "dual", "corners", "orders", "boundary", NULL},
                                     &manifold2d_type_object, &primal, &manifold2d_type_object, &dual, &PyArray_Type,
                                     &corners, &PyArray_Type, &orders, &PyArray_Type, &boundary))
    {
        return NULL;
    }

    if (primal->n_points != dual->n_surfaces || primal->n_lines != dual->n_lines ||
        primal->n_surfaces != dual->n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Primal and dual manifolds must have the same number of points (%u) / surfaces (%u), lines (%u) / "
                     "lines (%u), and surfaces (%u) / points (%u).",
                     primal->n_points, dual->n_surfaces, primal->n_lines, dual->n_lines, primal->n_surfaces,
                     dual->n_points);
        return NULL;
    }

    if (check_input_array(corners, 3, (const npy_intp[3]){0, 4, 2}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "corners") < 0)
    {
        return NULL;
    }
    const unsigned n_elements = PyArray_DIM(corners, 0);

    if (check_input_array(orders, 2, (const npy_intp[2]){n_elements, 2}, NPY_UINT,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "orders") < 0)
    {
        return NULL;
    }

    if (check_input_array(boundary, 1, (const npy_intp[1]){0}, NPY_UINT, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                          "boundary") < 0)
    {
        return NULL;
    }

    const quad_info_t *const quads = (const quad_info_t *)PyArray_DATA(corners);
    const index_2d_t *const orders_ptr = (const index_2d_t *)PyArray_DATA(orders);

    mesh_t *const this = (mesh_t *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    const unsigned boundary_len = PyArray_DIM(boundary, 0);
    const npy_uint *const boundary_data = (const npy_uint *)PyArray_DATA(boundary);
    unsigned *const boundary_indices = allocate(&SYSTEM_ALLOCATOR, boundary_len * sizeof *boundary_indices);
    if (!boundary_indices)
        return NULL;

    const mfv2d_result_t res =
        element_mesh_create(&this->element_mesh, &SYSTEM_ALLOCATOR, n_elements, quads, orders_ptr);
    if (res != MFV2D_SUCCESS)
    {
        deallocate(&SYSTEM_ALLOCATOR, boundary_indices);
        Py_DECREF(this);
        return NULL;
    }

    memcpy(boundary_indices, boundary_data, boundary_len * sizeof *boundary_indices);
    this->boundary_indices = boundary_indices;
    this->boundary_count = boundary_len;
    this->primal = primal;
    this->dual = dual;
    Py_INCREF(primal);
    Py_INCREF(dual);

    return (PyObject *)this;
}

static void mesh_dealloc(mesh_t *const this)
{
    element_mesh_destroy(&this->element_mesh);
    deallocate(&SYSTEM_ALLOCATOR, this->boundary_indices);
    Py_DECREF(this->primal);
    Py_DECREF(this->dual);
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *mesh_get_primal(const mesh_t *const this, void *Py_UNUSED(closure))
{
    Py_INCREF(this->primal);
    return (PyObject *)this->primal;
}

static PyObject *mesh_get_dual(const mesh_t *const this, void *Py_UNUSED(closure))
{
    Py_INCREF(this->dual);
    return (PyObject *)this->dual;
}

static PyObject *mesh_get_element_count(const mesh_t *const this, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(this->element_mesh.count);
}

static PyObject *mesh_get_leaf_count(const mesh_t *const this, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(this->element_mesh.leaf_count);
}

static PyObject *mesh_get_boundary_indices(const mesh_t *const this, void *Py_UNUSED(closure))
{
    const npy_intp shape[1] = {this->boundary_count};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, shape, NPY_UINT);
    if (!out)
        return NULL;
    memcpy(PyArray_DATA(out), this->boundary_indices, this->boundary_count * sizeof *this->boundary_indices);
    return (PyObject *)out;
}

static PyGetSetDef mesh_getset[] = {
    {
        .name = "primal",
        .get = (void *)mesh_get_primal,
        .set = NULL,
        .doc = "mfv2d._mfv2d.Manifold2d : Primal manifold topology.",
        .closure = NULL,
    },
    {
        .name = "dual",
        .get = (void *)mesh_get_dual,
        .set = NULL,
        .doc = "mfv2d._mfv2d.Manifold2d : Dual manifold topology.",
        .closure = NULL,
    },
    {
        .name = "element_count",
        .get = (void *)mesh_get_element_count,
        .set = NULL,
        .doc = "int : Number of elements in the mesh.",
        .closure = NULL,
    },
    {
        .name = "leaf_count",
        .get = (void *)mesh_get_leaf_count,
        .set = NULL,
        .doc = "int : Number of leaf elements in the mesh.",
        .closure = NULL,
    },
    {
        .name = "boundary_indices",
        .get = (void *)mesh_get_boundary_indices,
        .set = NULL,
        .doc = "array : Indices of the boundary elements.",
        .closure = NULL,
    },
    {},
};

static PyObject *mesh_get_element_parent(const mesh_t *const this, PyObject *index)
{
    long index_long = PyLong_AsLong(index);
    if (index_long == -1 && PyErr_Occurred())
    {
        return NULL;
    }

    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    const element_base_t *const elem = &this->element_mesh.elements[index_long].base;
    if (elem->parent == PARENT_IDX_INVALID)
    {
        Py_RETURN_NONE;
    }

    return PyLong_FromLong(elem->parent);
}

static PyObject *mesh_split_element(mesh_t *const this, PyObject *args, PyObject *kwds)
{
    long index_long;
    long orders_bottom_left[2];
    long orders_bottom_right[2];
    long orders_top_right[2];
    long orders_top_left[2];
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "l(ll)(ll)(ll)(ll)",
            (char *[6]){"", "orders_bottom_left", "orders_bottom_right", "orders_top_right", "orders_top_left", NULL},
            &index_long, orders_bottom_left + 0, orders_bottom_left + 1, orders_bottom_right + 0,
            orders_bottom_right + 1, orders_top_right + 0, orders_top_right + 1, orders_top_left + 0,
            orders_top_left + 1))
    {
        return NULL;
    }

    if (orders_bottom_left[0] <= 0 || orders_bottom_left[1] <= 0 || orders_bottom_right[0] <= 0 ||
        orders_bottom_right[1] <= 0 || orders_top_right[0] <= 0 || orders_top_right[1] <= 0 ||
        orders_top_left[0] <= 0 || orders_top_left[1] <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive (got (%l, %l), (%l, %l), (%l, %l), (%l, %l)).",
                     orders_bottom_left[0], orders_bottom_left[1], orders_bottom_right[0], orders_bottom_right[1],
                     orders_top_right[0], orders_top_right[1], orders_top_left[0], orders_top_left[1]);
        return NULL;
    }

    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    const mfv2d_result_t res = element_mesh_split_element(
        &this->element_mesh, index_long, (index_2d_t){orders_bottom_left[0], orders_bottom_left[1]},
        (index_2d_t){orders_bottom_right[0], orders_bottom_right[1]},
        (index_2d_t){orders_top_right[0], orders_top_right[1]}, (index_2d_t){orders_top_left[0], orders_top_left[1]});
    if (res != MFV2D_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not split element at index %ld: %s", index_long, mfv2d_result_str(res));
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *mesh_get_element_children(const mesh_t *const this, PyObject *index)
{
    long index_long = PyLong_AsLong(index);
    if (index_long == -1 && PyErr_Occurred())
    {
        return NULL;
    }

    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    const element_t *const elem = &this->element_mesh.elements[index_long];
    if (elem->base.type != ELEMENT_TYPE_NODE)
        Py_RETURN_NONE;

    const element_node_t *const node = &elem->node;
    return Py_BuildValue("IIII", node->child_bottom_left, node->child_bottom_right, node->child_top_right,
                         node->child_top_left);
}

static PyObject *mesh_get_leaf_corners(const mesh_t *const this, PyObject *index)
{
    long index_long = PyLong_AsLong(index);
    if (index_long == -1 && PyErr_Occurred())
    {
        return NULL;
    }
    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
    }

    const element_t *const elem = &this->element_mesh.elements[index_long];
    if (elem->base.type != ELEMENT_TYPE_LEAF)
    {
        PyErr_Format(PyExc_ValueError, "Element at index %ld is not a leaf.", index_long);
        return NULL;
    }

    const element_leaf_t *const leaf = &elem->leaf;
    const npy_intp shape[2] = {4, 2};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    if (!out)
        return NULL;

    double *const out_data = (double *)PyArray_DATA(out);
    memcpy(out_data, &leaf->data.corners, sizeof leaf->data.corners);
    return (PyObject *)out;
}

static PyObject *mesh_get_leaf_orders(const mesh_t *const this, PyObject *index)
{
    long index_long = PyLong_AsLong(index);
    if (index_long == -1 && PyErr_Occurred())
    {
        return NULL;
    }
    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    const element_t *const elem = &this->element_mesh.elements[index_long];
    if (elem->base.type != ELEMENT_TYPE_LEAF)
    {
        PyErr_Format(PyExc_ValueError, "Element at index %ld is not a leaf.", index_long);
        return NULL;
    }

    const element_leaf_t *const leaf = &elem->leaf;
    return Py_BuildValue("II", leaf->data.orders.i, leaf->data.orders.j);
}

static PyObject *mesh_get_leaf_indices(const mesh_t *const this, PyObject *Py_UNUSED(args))
{
    const npy_intp cnt = this->element_mesh.leaf_count;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &cnt, NPY_UINT);
    npy_uint *const out_data = (npy_uint *)PyArray_DATA(out);
    // TODO: reverse so that we can end as soon as we count up to cnt
    for (unsigned i = 0, j = 0; i < this->element_mesh.count; ++i)
    {
        if (this->element_mesh.elements[i].base.type == ELEMENT_TYPE_LEAF)
        {
            out_data[j++] = i;
        }
    }

    return (PyObject *)out;
}

static PyObject *mesh_copy(const mesh_t *const this, PyObject *Py_UNUSED(args))
{
    mesh_t *const that = (mesh_t *)mesh_type_object.tp_alloc(&mesh_type_object, 0);
    if (!that)
        return NULL;

    that->boundary_count = this->boundary_count;
    that->boundary_indices = allocate(&SYSTEM_ALLOCATOR, this->boundary_count * sizeof *this->boundary_indices);
    if (!that->boundary_indices)
    {
        Py_DECREF(that);
        return NULL;
    }
    const mfv2d_result_t res = element_mesh_copy_element(&this->element_mesh, &that->element_mesh);
    if (res != MFV2D_SUCCESS)
    {
        Py_DECREF(that);
        PyErr_Format(PyExc_RuntimeError, "Could not copy element mesh: %s", mfv2d_result_str(res));
        return NULL;
    }

    memcpy(that->boundary_indices, this->boundary_indices, this->boundary_count * sizeof *this->boundary_indices);

    that->primal = this->primal;
    Py_INCREF(that->primal);
    that->dual = this->dual;
    Py_INCREF(that->dual);

    return (PyObject *)that;
}

static PyObject *mesh_get_element_depth(const mesh_t *const this, PyObject *index)
{
    long index_long = PyLong_AsLong(index);
    if (index_long == -1 && PyErr_Occurred())
    {
        return NULL;
    }
    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    unsigned depth = 0;
    const element_t *elem = &this->element_mesh.elements[index_long];
    while (elem->base.parent != PARENT_IDX_INVALID)
    {
        ++depth;
        elem = &this->element_mesh.elements[elem->base.parent];
    }

    return PyLong_FromUnsignedLong(depth);
}

static PyObject *mesh_set_leaf_orders(const mesh_t *const this, PyObject *args, PyObject *kwds)
{
    long index_long;
    long orders_i;
    long orders_j;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l(ll)", (char *[4]){"", "order_1", "order_2", NULL}, &index_long,
                                     &orders_i, &orders_j))
    {
        return NULL;
    }

    if (index_long < 0 || index_long >= this->element_mesh.count)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range for mesh with %u elements.", index_long,
                     this->element_mesh.count);
        return NULL;
    }

    if (orders_i <= 0 || orders_j <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive (got (%ld, %ld)).", orders_i, orders_j);
        return NULL;
    }

    element_t *const elem = &this->element_mesh.elements[index_long];
    if (elem->base.type != ELEMENT_TYPE_LEAF)
    {
        PyErr_Format(PyExc_ValueError, "Element at index %ld is not a leaf.", index_long);
        return NULL;
    }

    element_leaf_t *const leaf = &elem->leaf;
    leaf->data.orders.i = orders_i;
    leaf->data.orders.j = orders_j;

    Py_RETURN_NONE;
}

static mfv2d_result_t refine_element_depth_first(element_mesh_t *mesh, unsigned index, unsigned depth,
                                                 unsigned max_depth, PyObject *predicate, PyObject *forwarded_args,
                                                 PyObject *forwarded_kwargs)
{
    if (depth >= max_depth)
        return MFV2D_SUCCESS;

    const element_t *const elem = &mesh->elements[index];
    if (elem->base.type != ELEMENT_TYPE_LEAF)
    {
        // Should never happen.
        return MFV2D_UNSPECIFIED_ERROR;
    }

    PyObject *index_val = PyLong_FromUnsignedLong(index);
    if (!index_val)
        return MFV2D_UNSPECIFIED_ERROR;
    PyTuple_SET_ITEM(forwarded_args, 1, index_val);
    PyObject *result = PyObject_Call(predicate, forwarded_args, forwarded_kwargs);
    Py_DECREF(index_val);
    PyTuple_SET_ITEM(forwarded_args, 1, NULL);
    if (!result)
    {
        return MFV2D_FAILED_CALLBACK;
    }
    if (Py_IsNone(result))
    {
        Py_DECREF(result);
        return MFV2D_SUCCESS;
    }

    long order_bottom_left_1, order_bottom_left_2;
    long order_bottom_right_1, order_bottom_right_2;
    long order_top_right_1, order_top_right_2;
    long order_top_left_1, order_top_left_2;
    const int parsed_orders = PyArg_ParseTuple(result, "(ll)(ll)(ll)(ll)", &order_bottom_left_1, &order_bottom_left_2,
                                               &order_bottom_right_1, &order_bottom_right_2, &order_top_right_1,
                                               &order_top_right_2, &order_top_left_1, &order_top_left_2);
    Py_DECREF(result);
    if (!parsed_orders)
    {
        return MFV2D_FAILED_CALLBACK;
    }

    if (order_bottom_left_1 <= 0 || order_bottom_left_2 <= 0 || order_bottom_right_1 <= 0 ||
        order_bottom_right_2 <= 0 || order_top_right_1 <= 0 || order_top_right_2 <= 0 || order_top_left_1 <= 0 ||
        order_top_left_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError,
                     "Orders must be positive (got (%ld, %ld), (%ld, %ld), (%ld, %ld), (%ld, %ld), "
                     "(%ld, %ld), (%ld, %ld)).",
                     order_bottom_left_1, order_bottom_left_2, order_bottom_right_1, order_bottom_right_2,
                     order_top_right_1, order_top_right_2, order_top_left_1, order_top_left_2);
        return MFV2D_FAILED_CALLBACK;
    }

    const unsigned children_begin = mesh->count;
    const mfv2d_result_t res = element_mesh_split_element(
        mesh, index, (index_2d_t){order_bottom_left_1, order_bottom_left_2},
        (index_2d_t){order_bottom_right_1, order_bottom_right_2}, (index_2d_t){order_top_right_1, order_top_right_2},
        (index_2d_t){order_top_left_1, order_top_left_2});
    if (res != MFV2D_SUCCESS)
    {
        return res;
    }

    mfv2d_result_t res_children = MFV2D_SUCCESS;
    for (unsigned i = children_begin; i < children_begin + 4; ++i)
    {
        if ((res_children = refine_element_depth_first(mesh, i, depth + 1, max_depth, predicate, forwarded_args,
                                                       forwarded_kwargs)) != MFV2D_SUCCESS)
            break;
    }

    return res_children;
}

[[gnu::format(printf, 2, 3)]]
static void raise_exception_from(PyObject *exception, const char *format, ...)
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

static mesh_t *mesh_split_prepare_arguments(const mesh_t *const this, PyObject *args, PyObject **p_predicate,
                                            PyObject **p_forwarded_args, long *p_max_depth)
{
    if (PyTuple_GET_SIZE(args) < 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at least two arguments.");
        return NULL;
    }
    const long maximum_depth = PyLong_AsLong(PyTuple_GET_ITEM(args, 0));
    if (maximum_depth == -1 && PyErr_Occurred())
        return NULL;

    if (maximum_depth <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Maximum depth must be positive (got %ld).", maximum_depth);
        return NULL;
    }

    PyObject *const predicate = PyTuple_GET_ITEM(args, 1);
    if (!PyCallable_Check(predicate))
    {
        PyErr_Format(PyExc_TypeError, "Predicate must be callable (got %s).", Py_TYPE(predicate)->tp_name);
        return NULL;
    }

    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    PyObject *forwarded_args = PyTuple_New(nargs);
    if (!forwarded_args)
    {
        return NULL;
    }

    if (nargs > 2)
    {
        for (unsigned i = 2; i < nargs; ++i)
        {
            PyObject *arg = PyTuple_GET_ITEM(args, i);
            Py_INCREF(arg);
            PyTuple_SET_ITEM(forwarded_args, i, arg);
        }
    }

    mesh_t *output_mesh = (mesh_t *)mesh_copy(this, NULL);
    if (!output_mesh)
    {
        Py_DECREF(forwarded_args);
        return NULL;
    }
    Py_INCREF(output_mesh);
    PyTuple_SET_ITEM(forwarded_args, 0, (PyObject *)output_mesh);

    *p_predicate = predicate;
    *p_forwarded_args = forwarded_args;
    *p_max_depth = maximum_depth;

    return output_mesh;
}
static PyObject *mesh_split_depth_first(const mesh_t *const this, PyObject *args, PyObject *kwds)
{
    long maximum_depth;
    PyObject *predicate, *forwarded_args;
    mesh_t *const output_mesh = mesh_split_prepare_arguments(this, args, &predicate, &forwarded_args, &maximum_depth);
    if (!output_mesh)
    {
        return NULL;
    }

    for (unsigned i = 0; i < this->element_mesh.count; ++i)
    {
        if (this->element_mesh.elements[i].base.type != ELEMENT_TYPE_LEAF)
            continue;

        const mfv2d_result_t res = refine_element_depth_first(&output_mesh->element_mesh, i, 0, maximum_depth,
                                                              predicate, forwarded_args, kwds);
        if (res != MFV2D_SUCCESS)
        {
            raise_exception_from(PyExc_RuntimeError, "Could not split element %u, reason: %s", i,
                                 mfv2d_result_str(res));
            Py_DECREF(output_mesh);
            Py_DECREF(forwarded_args);
            return (PyObject *)output_mesh;
        }
    }

    Py_DECREF(forwarded_args);
    return (PyObject *)output_mesh;
}

static mfv2d_result_t refine_element_breath_first(element_mesh_t *mesh, unsigned index, PyObject *predicate,
                                                  PyObject *forwarded_args, PyObject *forwarded_kwargs)
{
    const element_t *const elem = &mesh->elements[index];
    if (elem->base.type != ELEMENT_TYPE_LEAF)
    {
        // Should never happen.
        return MFV2D_UNSPECIFIED_ERROR;
    }

    PyObject *index_val = PyLong_FromUnsignedLong(index);
    if (!index_val)
        return MFV2D_UNSPECIFIED_ERROR;
    PyTuple_SET_ITEM(forwarded_args, 1, index_val);
    PyObject *result = PyObject_Call(predicate, forwarded_args, forwarded_kwargs);
    Py_DECREF(index_val);
    PyTuple_SET_ITEM(forwarded_args, 1, NULL);
    if (!result)
    {
        return MFV2D_FAILED_CALLBACK;
    }
    if (Py_IsNone(result))
    {
        Py_DECREF(result);
        return MFV2D_SUCCESS;
    }

    long order_bottom_left_1, order_bottom_left_2;
    long order_bottom_right_1, order_bottom_right_2;
    long order_top_right_1, order_top_right_2;
    long order_top_left_1, order_top_left_2;
    const int parsed_orders = PyArg_ParseTuple(result, "(ll)(ll)(ll)(ll)", &order_bottom_left_1, &order_bottom_left_2,
                                               &order_bottom_right_1, &order_bottom_right_2, &order_top_right_1,
                                               &order_top_right_2, &order_top_left_1, &order_top_left_2);
    Py_DECREF(result);
    if (!parsed_orders)
    {
        return MFV2D_FAILED_CALLBACK;
    }

    if (order_bottom_left_1 <= 0 || order_bottom_left_2 <= 0 || order_bottom_right_1 <= 0 ||
        order_bottom_right_2 <= 0 || order_top_right_1 <= 0 || order_top_right_2 <= 0 || order_top_left_1 <= 0 ||
        order_top_left_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError,
                     "Orders must be positive (got (%ld, %ld), (%ld, %ld), (%ld, %ld), (%ld, %ld), "
                     "(%ld, %ld), (%ld, %ld)).",
                     order_bottom_left_1, order_bottom_left_2, order_bottom_right_1, order_bottom_right_2,
                     order_top_right_1, order_top_right_2, order_top_left_1, order_top_left_2);
        return MFV2D_FAILED_CALLBACK;
    }

    const mfv2d_result_t res = element_mesh_split_element(
        mesh, index, (index_2d_t){order_bottom_left_1, order_bottom_left_2},
        (index_2d_t){order_bottom_right_1, order_bottom_right_2}, (index_2d_t){order_top_right_1, order_top_right_2},
        (index_2d_t){order_top_left_1, order_top_left_2});

    return res;
}

static PyObject *mesh_split_breath_first(const mesh_t *const this, PyObject *args, PyObject *kwds)
{
    long maximum_depth;
    PyObject *predicate, *forwarded_args;
    mesh_t *const output_mesh = mesh_split_prepare_arguments(this, args, &predicate, &forwarded_args, &maximum_depth);
    if (!output_mesh)
    {
        return NULL;
    }

    unsigned offset = 0;
    for (unsigned depth = 0; depth < maximum_depth; ++depth)
    {
        const unsigned count = output_mesh->element_mesh.count;
        for (unsigned i = offset; i < count; ++i)
        {
            if (output_mesh->element_mesh.elements[i].base.type != ELEMENT_TYPE_LEAF)
                continue;

            const mfv2d_result_t res =
                refine_element_breath_first(&output_mesh->element_mesh, i, predicate, forwarded_args, kwds);
            if (res != MFV2D_SUCCESS)
            {
                raise_exception_from(PyExc_RuntimeError, "Could not split element %u, reason: %s", i,
                                     mfv2d_result_str(res));
                Py_DECREF(output_mesh);
                Py_DECREF(forwarded_args);
                return NULL;
            }
        }
        // Only deal with newly created elements.
        offset = count;
    }

    Py_DECREF(forwarded_args);
    return (PyObject *)output_mesh;
}

PyDoc_STRVAR(mesh_get_element_parent_docstr,
             "get_element_parent(idx: typing.SupportsIndex, /) -> int | None\n"
             "Get the index of the element's parent or ``None`` if it is a root element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : SupportsIndex\n"
             "    Index of the element to get the parent from.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int or None\n"
             "    If the element has a parent, its index is returned. If the element is a\n"
             "    root element and has no parent, ``None`` is returned instead.\n");
PyDoc_STRVAR(mesh_split_element_docstr,
             "split_element(idx: typing.SupportsIndex, /, orders_bottom_left: tuple[int, int], orders_bottom_right: "
             "tuple[int, int], orders_top_right: tuple[int, int], orders_top_left: tuple[int, int]) -> None\n"
             "Split a leaf element into four child elements.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : SupportsIndex\n"
             "    Index of the element to split. Must be a leaf.\n"
             "\n"
             "orders_bottom_left : (int, int)\n"
             "    Orders of the newly created bottom left elements.\n"
             "\n"
             "orders_bottom_right : (int, int)\n"
             "    Orders of the newly created bottom right elements.\n"
             "\n"
             "orders_top_right : (int, int)\n"
             "    Orders of the newly created top right elements..\n"
             "\n"
             "orders_top_left : (int, int)\n"
             "    Orders of the newly created top left elements.\n");
PyDoc_STRVAR(mesh_get_element_children_docstr,
             "get_element_children(idx: SupportsIndex, /) -> tuple[int, int, int, int] | None\n"
             "Get indices of element's children.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : SupportsIndex\n"
             "    Index of the element to get the children for.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "(int, int, int, int) or None\n"
             "    If the element has children, their indices are returned in the order bottom\n"
             "    left, bottom right, top right, and top left. If the element is a leaf element\n"
             "    and has no parents, ``None`` is returned.\n");
PyDoc_STRVAR(mesh_get_leaf_corners_docstr, "get_leaf_corners(idx: typing.SupportsIndex, /) -> npt.NDArray[np.double]\n"
                                           "Get corners of the leaf element.\n"
                                           "\n"
                                           "Parameters\n"
                                           "----------\n"
                                           "idx : SupportsIndex\n"
                                           "    Index of the leaf element to get the orders for.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "(4, 2) array\n"
                                           "    Corners of the element in the counter-clockwise order, starting at\n"
                                           "    the bottom left corner.\n");
PyDoc_STRVAR(mesh_get_leaf_orders_docstr, "get_leaf_orders(idx: typing.SupportsIndex, /) -> tuple[int, int]\n"
                                          "Get orders of the leaf element.\n"
                                          "\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "idx : SupportsIndex\n"
                                          "    Index of the leaf element to get the orders for.\n"
                                          "\n"
                                          "Returns\n"
                                          "-------\n"
                                          "(int, int)\n"
                                          "    Orders of the leaf element in the first and second direction.\n");

PyDoc_STRVAR(mesh_get_leaf_indices_docstr, "get_leaf_indices() -> npt.NDArray[np.uintc]\n"
                                           "Get indices of leaf elements.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "(N,) array\n"
                                           "    Indices of leaf elements.\n");

PyDoc_STRVAR(mesh_copy_docstr, "copy() -> Mesh\n"
                               "Create a copy of the mesh.\n"
                               "\n"
                               "Returns\n"
                               "-------\n"
                               "Mesh\n"
                               "    Copy of the mesh.\n");
PyDoc_STRVAR(mesh_get_element_depth_docstr, "get_element_depth(idx: typing.SupportsIndex, /) -> int\n"
                                            "Check how deep the element is in the hierarchy.\n"
                                            "\n"
                                            "Parameters\n"
                                            "----------\n"
                                            "idx : SupportsIndex\n"
                                            "    Index of the element element to check.\n"
                                            "\n"
                                            "Returns\n"
                                            "-------\n"
                                            "int\n"
                                            "    Number of ancestors of the element.\n");
PyDoc_STRVAR(mesh_set_leaf_orders_docstr,
             "set_leaf_orders(idx: typing.SupportsIndex, /, order_1: int, order_2: int) -> None\n"
             "Set orders of a leaf element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : SupportsIndex\n"
             "    Index of the leaf element to set.\n"
             "\n"
             "order_1 : int\n"
             "    New order of the leaf in the first dimension.\n"
             "\n"
             "order_2 : int\n"
             "    New order of the leaf in the second dimension.\n");
PyDoc_STRVAR(mesh_split_depth_first_docstr,
             "split_depth_first(maximum_depth: int, predicate: Callable, *args, **kwargs) -> Mesh\n"
             "Split leaf elements based on a predicate in a depth-first approach.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "maximum_depth : int\n"
             "    Maximum number of levels of division allowed.\n"
             "\n"
             "predicate : Callable\n"
             "    Predicate to use for determining if the element should be split.\n"
             "    It will always be given the mesh as the first argument and\n"
             "    the ID of the element as its second argument. If the element should\n"
             "    not be split, the function should return ``None``, otherwise it should\n"
             "    return orders for all four newly created elements.\n"
             "\n"
             "*args\n"
             "    Arguments passed to the predicate function.\n"
             "\n"
             "**kwargs\n"
             "    Keyword arguments passed to the predicate function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Mesh\n"
             "    Mesh with refined elements.\n");
PyDoc_STRVAR(mesh_split_breath_first_docstr,
             "split_breath_first(maximum_depth: int, predicate: Callable, *args, **kwargs) -> Mesh\n"
             "Split leaf elements based on a predicate in a breath-first approach.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "maximum_depth : int\n"
             "    Maximum number of levels of division allowed.\n"
             "\n"
             "predicate : Callable\n"
             "    Predicate to use for determining if the element should be split.\n"
             "    It will always be given the mesh as the first argument and\n"
             "    the ID of the element as its second argument. If the element should\n"
             "    not be split, the function should return ``None``, otherwise it should\n"
             "    return orders for all four newly created elements.\n"
             "\n"
             "*args\n"
             "    Arguments passed to the predicate function.\n"
             "\n"
             "**kwargs\n"
             "    Keyword arguments passed to the predicate function.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Mesh\n"
             "    Mesh with refined elements.\n");

static PyMethodDef mesh_methods[] = {
    {
        .ml_name = "get_element_parent",
        .ml_meth = (void *)mesh_get_element_parent,
        .ml_flags = METH_O,
        .ml_doc = mesh_get_element_parent_docstr,
    },
    {
        .ml_name = "split_element",
        .ml_meth = (void *)mesh_split_element,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = mesh_split_element_docstr,
    },
    {
        .ml_name = "get_element_children",
        .ml_meth = (void *)mesh_get_element_children,
        .ml_flags = METH_O,
        .ml_doc = mesh_get_element_children_docstr,
    },
    {
        .ml_name = "get_leaf_corners",
        .ml_meth = (void *)mesh_get_leaf_corners,
        .ml_flags = METH_O,
        .ml_doc = mesh_get_leaf_corners_docstr,
    },
    {
        .ml_name = "get_leaf_orders",
        .ml_meth = (void *)mesh_get_leaf_orders,
        .ml_flags = METH_O,
        .ml_doc = mesh_get_leaf_orders_docstr,
    },
    {
        .ml_name = "get_leaf_indices",
        .ml_meth = (void *)mesh_get_leaf_indices,
        .ml_flags = METH_NOARGS,
        .ml_doc = mesh_get_leaf_indices_docstr,
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)mesh_copy,
        .ml_flags = METH_NOARGS,
        .ml_doc = mesh_copy_docstr,
    },
    {
        .ml_name = "get_element_depth",
        .ml_meth = (void *)mesh_get_element_depth,
        .ml_flags = METH_O,
        .ml_doc = mesh_get_element_depth_docstr,
    },
    {
        .ml_name = "set_leaf_orders",
        .ml_meth = (void *)mesh_set_leaf_orders,
        .ml_flags = METH_KEYWORDS | METH_VARARGS,
        .ml_doc = mesh_set_leaf_orders_docstr,
    },
    {
        .ml_name = "split_depth_first",
        .ml_meth = (void *)mesh_split_depth_first,
        .ml_flags = METH_KEYWORDS | METH_VARARGS,
        .ml_doc = mesh_split_depth_first_docstr,
    },
    {
        .ml_name = "split_breath_first",
        .ml_meth = (void *)mesh_split_breath_first,
        .ml_flags = METH_KEYWORDS | METH_VARARGS,
        .ml_doc = mesh_split_breath_first_docstr,
    },
    {},
};

PyDoc_STRVAR(mesh_type_docstr,
             "Mesh(primal : Manifold2D, dual : Manifold2D, corners: array, orders: array, boundary: array)\n"
             "Mesh containing topology, geometry, and discretization information.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "primal : Manifold2D\n"
             "    Primal topology manifold.\n"
             "\n"
             "dual : Manifold2D\n"
             "    Dual topology manifold.\n"
             "\n"
             "corners : (N, 4, 2) array\n"
             "    Array of element corners.\n"
             "\n"
             "orders : (N, 2) array\n"
             "    Array of element orders.\n"
             "\n"
             "boundary : (N,) array\n"
             "    Array of boundary edge indices.\n");

PyTypeObject mesh_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d.Mesh",
    .tp_new = mesh_new,
    .tp_dealloc = (destructor)mesh_dealloc,
    .tp_getset = mesh_getset,
    .tp_methods = mesh_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_basicsize = sizeof(mesh_t),
    .tp_doc = mesh_type_docstr,
};
