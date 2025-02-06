//
// Created by jan on 24.11.2024.
//

#include "manifold2d.h"

#include <numpy/arrayobject.h>

#include "geoidobject.h"
#include "lineobject.h"
#include "surfaceobject.h"

static PyObject *manifold2d_str(PyObject *self)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyUnicode_FromFormat("Manifold2D(%u points, %u lines, %u surfaces)", this->n_points, this->n_lines,
                                this->n_surfaces);
}

PyDoc_STRVAR(manifold2d_type_docstring, "Manifold2D(...)\n"
                                        "Two dimensional manifold consisting of surfaces made of lines.\n");
// TODO
// static PyObject *manifold2d_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
// {
//     manifold2d_object_t *this = NULL;
//     unsigned n_elements = 0;
//     unsigned *per_element = NULL;
//     unsigned *flat_points = NULL;
//     PyObject *seq = NULL;
//
//     PyObject *root;
//     unsigned n_points;
//     if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IO", (char *[3]){"n_points", "connectivity", NULL}, &n_points,
//                                      &root))
//     {
//         return NULL;
//     }
//
//     // Load element data
//     {
//         seq = PySequence_Fast(root, "Second parameter must be a sequence of sequences");
//         if (!seq)
//         {
//             goto end;
//         }
//         n_elements = PySequence_Fast_GET_SIZE(seq);
//         per_element = PyMem_Malloc(sizeof(*per_element) * n_elements);
//         if (!per_element)
//         {
//             goto end;
//         }
//         unsigned total_pts = 0;
//         for (unsigned i = 0; i < n_elements; ++i)
//         {
//             const Py_ssize_t len = PySequence_Size(PySequence_Fast_GET_ITEM(seq, i));
//             if (len < 0)
//             {
//                 PyErr_Format(PyExc_TypeError, "Element indices for element %u were not a sequence.", i);
//                 goto end;
//             }
//             if (len < 3)
//             {
//                 PyErr_Format(PyExc_ValueError, "Element %u had only %u indices given (at least 3 are needed).", i,
//                              (unsigned)len);
//                 goto end;
//             }
//             total_pts += (unsigned)len;
//             per_element[i] = (unsigned)len;
//         }
//         flat_points = PyMem_Malloc(sizeof(*flat_points) * total_pts);
//         if (!flat_points)
//         {
//             goto end;
//         }
//         for (unsigned i = 0, j = 0; i < n_elements; ++i)
//         {
//             const PyArrayObject *const idx = (PyArrayObject *)PyArray_FromAny(
//                 PySequence_Fast_GET_ITEM(seq, i), PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS,
//                 NULL);
//             if (!idx)
//             {
//                 goto end;
//             }
//
//             const unsigned *data = PyArray_DATA(idx);
//             for (unsigned k = 0; k < per_element[i]; ++k)
//             {
//                 const unsigned v = data[k];
//                 if (v > n_points)
//                 {
//                     PyErr_Format(PyExc_ValueError,
//                                  "Element %u had specified a point with index %u as its %u"
//                                  " point, while only %u points were given.",
//                                  i, v, k, n_points);
//                     Py_DECREF(idx);
//                     goto end;
//                 }
//                 flat_points[j + k] = v;
//             }
//             j += per_element[i];
//
//             Py_DECREF(idx);
//         }
//         this = (PyVL_MeshObject *)type->tp_alloc(type, 0);
//         if (!this)
//         {
//             goto end;
//         }
//         const int status = mesh_from_elements(&this->mesh, n_elements, per_element, flat_points, &CVL_OBJ_ALLOCATOR);
//         if (status)
//         {
//             PyErr_Format(PyExc_RuntimeError, "Failed creating a mesh from given indices.");
//             goto end;
//         }
//     }
//     this->mesh.n_points = n_points;
//
// end:
//     PyMem_Free(flat_points);
//     PyMem_Free(per_element);
//     Py_XDECREF(seq);
//     return (PyObject *)this;
// }

static void manifold2d_dealloc(PyObject *self)
{
    manifold2d_object_t *this = (manifold2d_object_t *)self;

    PyObject_Free(this->lines);
    PyObject_Free(this->surf_counts);
    PyObject_Free(this->surf_lines);

    Py_TYPE(this)->tp_free(this);
}

static PyObject *manifold2d_get_dimension(PyObject *Py_UNUSED(self), void *Py_UNUSED(closure))
{
    return PyLong_FromLong(2);
}

static PyObject *manifold2d_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->n_points);
}

static PyObject *manifold2d_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->n_lines);
}

static PyObject *manifold2d_get_n_surfaces(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->n_surfaces);
}

static PyGetSetDef manifold2d_getset[] = {
    {.name = "dimension",
     .get = manifold2d_get_dimension,
     .set = NULL,
     .doc = "int : Dimension of the manifold.",
     .closure = NULL},
    {.name = "n_points",
     .get = manifold2d_get_n_points,
     .set = NULL,
     .doc = "Number of points in the mesh.",
     .closure = NULL},
    {.name = "n_lines",
     .get = manifold2d_get_n_lines,
     .set = NULL,
     .doc = "Number of lines in the mesh.",
     .closure = NULL},
    {.name = "n_surfaces",
     .get = manifold2d_get_n_surfaces,
     .set = NULL,
     .doc = "Number of surfaces in the mesh.",
     .closure = NULL},
    {0},
};

static PyObject *manifold2d_get_line(PyObject *self, PyObject *arg)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    geo_id_t id;
    if (geo_id_from_object(arg, &id) < 0)
        return NULL;

    line_object_t *const line = line_from_indices(this->lines[id.index].begin, this->lines[id.index].end);
    if (!line)
        return NULL;

    if (id.reverse)
    {
        line->value.begin.reverse = !line->value.begin.reverse;
        line->value.end.reverse = !line->value.end.reverse;
    }

    return (PyObject *)line;
}

static PyObject *manifold2d_get_surface(PyObject *self, PyObject *arg)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    geo_id_t id;

    if (geo_id_from_object(arg, &id) < 0)
    {
        return NULL;
    }
    if (id.index >= (long)this->n_surfaces)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld is our of bounds for a mesh with %u surfaces.", id.index,
                     this->n_surfaces);
        return NULL;
    }

    const size_t offset_0 = this->surf_counts[id.index];
    const size_t offset_1 = this->surf_counts[id.index + 1];
    surface_object_t *surf = surface_object_from_value(offset_1 - offset_0, this->surf_lines + offset_0);
    if (!surf)
    {
        return NULL;
    }

    if (id.reverse)
    {
        for (unsigned i = 0; i < surf->n_lines; ++i)
        {
            surf->lines[i].reverse = !surf->lines[i].reverse;
        }
    }
    return (PyObject *)surf;
}

// TODO: oh God have mercy
// static PyObject *pyvl_mesh_compute_dual(PyObject *self, PyObject *Py_UNUSED(arg))
// {
//     PyVL_MeshObject *that = (PyVL_MeshObject *)pyvl_mesh_type.tp_alloc(&pyvl_mesh_type, 0);
//     if (!that)
//     {
//         return NULL;
//     }
//     const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
//     const int stat = mesh_dual_from_primal(&that->mesh, &this->mesh, &CVL_OBJ_ALLOCATOR);
//     if (stat != 0)
//     {
//         PyErr_Format(PyExc_RuntimeError, "Could not compute dual to the mesh.");
//         Py_DECREF(that);
//         return NULL;
//     }
//     return (PyObject *)that;
// }

// static void cleanup_memory(PyObject *cap)
// {
//     void *const ptr = PyCapsule_GetPointer(cap, NULL);
//     PyMem_Free(ptr);
// }

// TODO: Lord have mercy
// static PyObject *pyvl_mesh_to_element_connectivity(PyObject *self, PyObject *Py_UNUSED(arg))
// {
//     const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
//     unsigned *point_counts, *flat_points;
//     const unsigned n_elements = mesh_to_elements(&this->mesh, &point_counts, &flat_points, &CVL_MEM_ALLOCATOR);
//     if (n_elements != this->mesh.n_surfaces)
//     {
//         if (!PyErr_Occurred())
//         {
//             PyErr_Format(PyExc_RuntimeError, "Could not convert mesh to elements.");
//         }
//         return NULL;
//     }
//     PyObject *const cap = PyCapsule_New(point_counts, NULL, cleanup_memory);
//     if (!cap)
//     {
//         PyMem_Free(point_counts);
//         PyMem_Free(flat_points);
//         return NULL;
//     }
//     const npy_intp n_counts = n_elements;
//     PyObject *const counts_array = PyArray_SimpleNewFromData(1, &n_counts, NPY_UINT, point_counts);
//     if (!counts_array)
//     {
//         Py_DECREF(cap);
//         PyMem_Free(flat_points);
//         return NULL;
//     }
//     if (PyArray_SetBaseObject((PyArrayObject *)counts_array, cap) < 0)
//     {
//         Py_DECREF(counts_array);
//         Py_DECREF(cap);
//         PyMem_Free(flat_points);
//         return NULL;
//     }
//
//     PyObject *const cap_2 = PyCapsule_New(flat_points, NULL, cleanup_memory);
//     if (!cap_2)
//     {
//         Py_DECREF(counts_array);
//         PyMem_Free(flat_points);
//         return NULL;
//     }
//     npy_intp n_flat = 0;
//     for (unsigned i = 0; i < n_elements; ++i)
//     {
//         n_flat += point_counts[i];
//     }
//     PyObject *const points_array = PyArray_SimpleNewFromData(1, &n_flat, NPY_UINT, flat_points);
//     if (!points_array)
//     {
//         Py_DECREF(counts_array);
//         Py_DECREF(cap_2);
//         return NULL;
//     }
//     if (PyArray_SetBaseObject((PyArrayObject *)points_array, cap_2) < 0)
//     {
//         Py_DECREF(counts_array);
//         Py_DECREF(points_array);
//         return NULL;
//     }
//
//     PyObject *out = PyTuple_Pack(2, counts_array, points_array);
//     if (!out)
//     {
//         Py_DECREF(counts_array);
//         Py_DECREF(points_array);
//     }
//     return out;
// }

// TODO: E^{1, 0}
// static PyObject *pyvl_line_velocities_from_point_velocities(PyObject *self, PyObject *const *args,
//                                                             const Py_ssize_t nargs)
// {
//     // args:
//     //  1.  Point velocities
//     //  2.  Output array of line velocities
//     if (nargs != 2)
//     {
//         PyErr_Format(PyExc_TypeError, "Static method requires 2 arguments, but was called with %u instead.",
//                      (unsigned)nargs);
//         return NULL;
//     }
//
//     const PyVL_MeshObject *primal = (PyVL_MeshObject *)self;
//
//     PyArrayObject *const point_velocities =
//         pyvl_ensure_array(args[0], 2, (const npy_intp[2]){primal->mesh.n_points, 3},
//                           NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Point velocities");
//     if (!point_velocities)
//         return NULL;
//     PyArrayObject *const line_buffer = pyvl_ensure_array(
//         args[1], 2, (const npy_intp[2]){primal->mesh.n_lines, 3},
//         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64, "Output array");
//     if (!line_buffer)
//         return NULL;
//
//     _Static_assert(3 * sizeof(npy_float64) == sizeof(real3_t), "Types must have the same size.");
//     real3_t const *restrict velocities_in = PyArray_DATA(point_velocities);
//     real3_t *restrict velocities_out = PyArray_DATA(line_buffer);
//
//     unsigned i;
// #pragma omp parallel for default(none) shared(primal, velocities_in, velocities_out)
//     for (i = 0; i < primal->mesh.n_lines; ++i)
//     {
//         const line_t *ln = primal->mesh.lines + i;
//         velocities_out[i] = real3_mul1(real3_add(velocities_in[ln->p1.value], velocities_in[ln->p2.value]), 0.5);
//     }
//
//     Py_RETURN_NONE;
// }

// TODO: maybe later
// static PyObject *pyvl_mesh_merge(PyObject *type, PyObject *const *args, Py_ssize_t nargs)
// {
//     unsigned n_surfaces = 0, n_lines = 0, n_points = 0, n_surface_entries = 0;
//
//     for (unsigned i = 0; i < (unsigned)nargs; ++i)
//     {
//         PyObject *const o = args[i];
//         if (!PyObject_TypeCheck(o, &pyvl_mesh_type))
//         {
//             PyErr_Format(PyExc_TypeError, "Element %u in the input sequence was not a Mesh, but was instead %R", i,
//                          Py_TYPE(o));
//             return NULL;
//         }
//         const PyVL_MeshObject *const this = (PyVL_MeshObject *)o;
//         n_surfaces += this->mesh.n_surfaces;
//         n_lines += this->mesh.n_lines;
//         n_points += this->mesh.n_points;
//         n_surface_entries += this->mesh.surface_offsets[this->mesh.n_surfaces];
//     }
//
//     PyVL_MeshObject *const this = (PyVL_MeshObject *)((PyTypeObject *)type)->tp_alloc((PyTypeObject *)type, 0);
//     if (!this)
//     {
//         return NULL;
//     }
//
//     line_t *const lines = PyObject_Malloc(sizeof *lines * n_lines);
//     unsigned *const surface_offsets = PyObject_Malloc(sizeof *surface_offsets * (n_surfaces + 1));
//     geo_id_t *const surface_lines = PyObject_Malloc(sizeof *surface_lines * n_surface_entries);
//
//     if (!lines || !surface_offsets || !surface_lines)
//     {
//         PyObject_Free(surface_lines);
//         PyObject_Free(surface_offsets);
//         PyObject_Free(lines);
//         return NULL;
//     }
//
//     unsigned cnt_pts = 0, cnt_lns = 0, cnt_surf = 0, cnt_entr = 0;
//     line_t *l = lines;
//     for (unsigned i = 0; i < (unsigned)nargs; ++i)
//     {
//         const PyVL_MeshObject *const m = (PyVL_MeshObject *)args[i];
//         // Lines are copied, but incremented
//         for (unsigned il = 0; il < m->mesh.n_lines; ++il)
//         {
//             const line_t *p_line = m->mesh.lines + il;
//             *l = (line_t){
//                 .p1 = (geo_id_t){.orientation = p_line->p1.orientation, .value = p_line->p1.value + cnt_pts},
//                 .p2 = (geo_id_t){.orientation = p_line->p2.orientation, .value = p_line->p2.value + cnt_pts},
//             };
//             l += 1;
//         }
//         // Surfaces are also copied with increments
//         for (unsigned is = 0; is < m->mesh.n_surfaces; ++is)
//         {
//             surface_offsets[cnt_surf + is] = m->mesh.surface_offsets[is] + cnt_entr;
//         }
//         for (unsigned is = 0; is < m->mesh.surface_offsets[m->mesh.n_surfaces]; ++is)
//         {
//             const geo_id_t original_line = m->mesh.surface_lines[is];
//             surface_lines[cnt_entr + is] =
//                 (geo_id_t){.orientation = original_line.orientation, .value = original_line.value + cnt_lns};
//         }
//         cnt_pts += m->mesh.n_points;
//         cnt_lns += m->mesh.n_lines;
//         cnt_surf += m->mesh.n_surfaces;
//         cnt_entr += m->mesh.surface_offsets[m->mesh.n_surfaces];
//     }
//     surface_offsets[n_surfaces] = cnt_entr;
//     this->mesh = (mesh_t){
//         .n_points = cnt_pts,
//         .n_lines = cnt_lns,
//         .lines = lines,
//         .n_surfaces = cnt_surf,
//         .surface_offsets = surface_offsets,
//         .surface_lines = surface_lines,
//     };
//
//     return (PyObject *)this;
// }
//
// static PyObject *pyvl_mesh_copy(PyObject *self, PyObject *Py_UNUSED(args))
// {
//     const PyVL_MeshObject *const origin = (PyVL_MeshObject *)self;
//
//     PyVL_MeshObject *const this = (PyVL_MeshObject *)pyvl_mesh_type.tp_alloc(&pyvl_mesh_type, 0);
//     if (!this)
//     {
//         return NULL;
//     }
//
//     this->mesh.n_points = origin->mesh.n_points;
//     this->mesh.n_lines = origin->mesh.n_lines;
//     this->mesh.n_surfaces = origin->mesh.n_surfaces;
//
//     this->mesh.lines = PyObject_Malloc(sizeof(*origin->mesh.lines) * origin->mesh.n_lines);
//     this->mesh.surface_offsets = PyObject_Malloc(sizeof(*origin->mesh.surface_offsets) * (origin->mesh.n_surfaces +
//     1)); this->mesh.surface_lines =
//         PyObject_Malloc(sizeof(*origin->mesh.surface_lines) *
//         (origin->mesh.surface_offsets[origin->mesh.n_surfaces]));
//     if (!this->mesh.surface_offsets || !this->mesh.lines || !this->mesh.surface_lines)
//     {
//         PyObject_Free(this->mesh.surface_lines);
//         PyObject_Free(this->mesh.lines);
//         PyObject_Free(this->mesh.surface_lines);
//         return NULL;
//     }
//
//     memcpy(this->mesh.lines, origin->mesh.lines, sizeof(*origin->mesh.lines) * origin->mesh.n_lines);
//     memcpy(this->mesh.surface_offsets, origin->mesh.surface_offsets,
//            sizeof(*origin->mesh.surface_offsets) * (origin->mesh.n_surfaces + 1));
//     memcpy(this->mesh.surface_lines, origin->mesh.surface_lines,
//            sizeof(*origin->mesh.surface_lines) * (origin->mesh.surface_offsets[origin->mesh.n_surfaces]));
//
//     return (PyObject *)this;
// }

// TODO: Applying E^{1, 0}
// static PyObject *pyvl_mesh_line_gradient(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
// {
//     // Arguments:
//     // self - mesh
//     // 0 - point value array
//     // 1 - output line array (optional)
//     if (nargs < 1 || nargs > 2)
//     {
//         PyErr_Format(PyExc_TypeError, "Function takes 1 to 2 arguments, but %u were given.", (unsigned)nargs);
//         return NULL;
//     }
//     const PyVL_MeshObject *const this = (PyVL_MeshObject *)self;
//     PyArrayObject *const point_values =
//         pyvl_ensure_array(args[1], 1, (const npy_intp[1]){this->mesh.n_points},
//                           NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Point value array");
//     if (!point_values)
//         return NULL;
//
//     PyArrayObject *line_values = NULL;
//     if (nargs == 2 || Py_IsNone(args[1]))
//     {
//         line_values = pyvl_ensure_array(args[2], 1, (const npy_intp[1]){this->mesh.n_lines},
//                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
//                                         NPY_FLOAT64, "Line circulation array");
//         Py_XINCREF(line_values);
//     }
//     else
//     {
//         const npy_intp nl = this->mesh.n_lines;
//         line_values = (PyArrayObject *)PyArray_SimpleNew(1, &nl, NPY_FLOAT64);
//     }
//     if (!line_values)
//         return NULL;
//
//     const unsigned n_lns = this->mesh.n_lines;
//     const line_t *const restrict lines = this->mesh.lines;
//     const real_t *const restrict v_in = PyArray_DATA(point_values);
//     real_t *const restrict v_out = PyArray_DATA(point_values);
//
//     unsigned i;
// #pragma omp parallel for default(none) shared(lines, v_in, v_out, n_lns)
//     for (i = 0; i < n_lns; ++i)
//     {
//         real_t x = 0;
//         const line_t ln = lines[i];
//         if (ln.p1.value != INVALID_ID)
//         {
//             x -= v_in[ln.p1.value];
//         }
//         if (ln.p2.value != INVALID_ID)
//         {
//             x += v_in[ln.p2.value];
//         }
//         v_out[i] = x;
//     }
//
//     return (PyObject *)line_values;
// }

// TODO: meh
static PyObject *maifold2d_from_irregular(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg_lines;
    PyObject *arg_surf;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IOO",
                                     (char *[4]){"n_points", "line_connectivity", "surface_connectivity", NULL}, &npts,
                                     &arg_lines, &arg_surf))
    {
        return NULL;
    }

    PyArrayObject *const line_array = (PyArrayObject *)PyArray_FromAny(
        arg_lines, PyArray_DescrFromType(NPY_INT), 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!line_array)
        return NULL;
    const unsigned n_lines = PyArray_DIM(line_array, 0);
    if (PyArray_DIM(line_array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(line_array, 0), (unsigned)PyArray_DIM(line_array, 1));
        Py_DECREF(line_array);
        return NULL;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    manifold2d_object_t *const this = (manifold2d_object_t *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(line_array);
        return NULL;
    }

    this->n_points = npts;
    this->n_lines = n_lines;
    this->n_surfaces = 0;

    this->lines = NULL;
    this->surf_lines = NULL;
    this->surf_counts = NULL;

    this->lines = PyObject_Malloc(sizeof *this->lines * n_lines);
    if (!this->lines)
    {
        Py_DECREF(this);
        Py_DECREF(line_array);
        return NULL;
    }

    const int *restrict p_in = PyArray_DATA(line_array);
    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        const geo_id_t begin = geo_id_pack(p_in[0]);
        const geo_id_t end = geo_id_pack(p_in[1]);

        if (begin.index >= npts || end.index >= npts)
        {
            PyErr_Format(PyExc_ValueError, "Line %u has points (%u, %u), but there were only %u points specified.",
                         i_ln, begin.index, end.index, npts);
            Py_DECREF(this);
            Py_DECREF(line_array);
            return NULL;
        }
        this->lines[i_ln] = (line_t){
            .begin = begin,
            .end = end,
        };
        p_in += 2;
    }
    Py_DECREF(line_array);

    PyObject *seq = PySequence_Fast(arg_surf, "Surface connectivity must be a sequence");
    if (!seq)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->n_surfaces = PySequence_Fast_GET_SIZE(seq);

    this->surf_counts = PyObject_Malloc(sizeof *this->surf_counts * (this->n_surfaces + 1));
    if (!this->surf_counts)
    {
        Py_DECREF(this);
        return NULL;
    }

    // Count up the offsets
    this->surf_counts[0] = 0;
    size_t n_surf_lines = 0;

    PyObject *const *const seq_items = PySequence_Fast_ITEMS(seq);
    for (unsigned i = 0; i < this->n_surfaces; ++i)
    {
        PyObject *const o = seq_items[i];
        const Py_ssize_t sz = PySequence_Size(o);
        if (sz < 0)
        {
            Py_DECREF(seq);
            Py_DECREF(this);
            return NULL;
        }
        n_surf_lines += (size_t)sz;
        this->surf_counts[i + 1] = n_surf_lines;
    }

    this->surf_lines = PyObject_Malloc(sizeof(*this->surf_lines) * n_surf_lines);
    if (!this->surf_lines)
    {
        Py_DECREF(seq);
        Py_DECREF(this);
        return NULL;
    }
    int same_size = 1;
    const size_t first_size = this->n_surfaces ? this->surf_counts[1] - this->surf_counts[0] : 0;
    for (unsigned i = 0; i < this->n_surfaces; ++i)
    {
        PyObject *const o = seq_items[i];
        PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(o, PyArray_DescrFromType(NPY_INT), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (!array)
        {
            Py_DECREF(seq);
            Py_DECREF(this);
            return NULL;
        }
        const int *ptr = PyArray_DATA(array);
        const size_t offset = this->surf_counts[i];
        const size_t len = this->surf_counts[i + 1] - offset;
        geo_id_t *const surf_lines = this->surf_lines + offset;
        for (unsigned j = 0; j < len; ++j)
        {
            const geo_id_t id = geo_id_pack(ptr[j]);
            if (id.index != GEO_ID_INVALID && id.index >= this->n_lines)
            {
                PyErr_Format(PyExc_ValueError,
                             "Surface %u specified a line with index %u, which is larger than"
                             "the number of lines in the mesh (%u).",
                             i, id.index, this->n_lines);
                Py_DECREF(array);
                Py_DECREF(seq);
                Py_DECREF(this);
                return NULL;
            }
            surf_lines[j] = id;
        }
        Py_DECREF(array);

        geo_id_t end;
        {
            const geo_id_t id1 = surf_lines[len - 1];
            if (id1.reverse)
            {
                end = this->lines[id1.index].begin;
            }
            else
            {
                end = this->lines[id1.index].end;
            }
        }
        for (unsigned j = 0; j < len; ++j)
        {

            geo_id_t begin, new_end;
            const geo_id_t id2 = surf_lines[j];
            if (id2.reverse)
            {
                begin = this->lines[id2.index].end;
                new_end = this->lines[id2.index].begin;
            }
            else
            {
                begin = this->lines[id2.index].begin;
                new_end = this->lines[id2.index].end;
            }

            if (begin.index != end.index)
            {
                PyErr_Format(PyExc_ValueError,
                             "Line %u does not end (point %u) where line %u begins (point %u) for surface %u.",
                             j == 0 ? len - 1 : j - 1, end.index, j, begin.index, i);
                Py_DECREF(seq);
                Py_DECREF(this);
                return NULL;
            }
            end = new_end;
        }
        same_size = same_size && (len == first_size);
    }
    Py_DECREF(seq);
    if (same_size)
    {
        PyErr_WarnFormat(PyExc_UserWarning, 0,
                         "Consider calling the Manifold2D.from_regular,"
                         " since all surfaces have the same length of %u.",
                         (unsigned)first_size);
    }

    return (PyObject *)this;
}

static PyObject *maifold2d_from_regular(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg_lines;
    PyObject *arg_surf;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IOO",
                                     (char *[4]){"n_points", "line_connectivity", "surface_connectivity", NULL}, &npts,
                                     &arg_lines, &arg_surf))
    {
        return NULL;
    }

    PyArrayObject *const line_array = (PyArrayObject *)PyArray_FromAny(
        arg_lines, PyArray_DescrFromType(NPY_INT), 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!line_array)
        return NULL;
    const unsigned n_lines = PyArray_DIM(line_array, 0);
    if (PyArray_DIM(line_array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(line_array, 0), (unsigned)PyArray_DIM(line_array, 1));
        Py_DECREF(line_array);
        return NULL;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    manifold2d_object_t *const this = (manifold2d_object_t *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(line_array);
        return NULL;
    }

    this->n_points = npts;
    this->n_lines = n_lines;
    this->n_surfaces = 0;

    this->lines = NULL;
    this->surf_lines = NULL;
    this->surf_counts = NULL;

    this->lines = PyObject_Malloc(sizeof *this->lines * n_lines);
    if (!this->lines)
    {
        Py_DECREF(this);
        Py_DECREF(line_array);
        return NULL;
    }

    const int *restrict p_in = PyArray_DATA(line_array);
    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        const geo_id_t begin = geo_id_pack(p_in[0]);
        const geo_id_t end = geo_id_pack(p_in[1]);

        if (begin.index >= npts || end.index >= npts)
        {
            PyErr_Format(PyExc_ValueError, "Line %u has points (%u, %u), but there were only %u points specified.",
                         i_ln, begin.index, end.index, npts);
            Py_DECREF(this);
            Py_DECREF(line_array);
            return NULL;
        }
        this->lines[i_ln] = (line_t){
            .begin = begin,
            .end = end,
        };
        p_in += 2;
    }
    Py_DECREF(line_array);

    PyArrayObject *const surfaces = (PyArrayObject *)PyArray_FromAny(arg_surf, PyArray_DescrFromType(NPY_INT), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!surfaces)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->n_surfaces = PyArray_DIM(surfaces, 0);
    const unsigned n_per_surf = PyArray_DIM(surfaces, 1);

    this->surf_counts = PyObject_Malloc(sizeof *this->surf_counts * (this->n_surfaces + 1));
    if (!this->surf_counts)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->surf_lines = PyObject_Malloc(sizeof(*this->surf_lines) * this->n_surfaces * n_per_surf);
    if (!this->surf_lines)
    {
        Py_DECREF(surfaces);
        Py_DECREF(this);
        return NULL;
    }

    // Count up the offsets
    this->surf_counts[0] = 0;
    const int *const lines = PyArray_DATA(surfaces);
    for (unsigned i = 0; i < this->n_surfaces; ++i)
    {
        const size_t offset = this->surf_counts[i];
        this->surf_counts[i + 1] = offset + n_per_surf;
        geo_id_t *const surf_lines = this->surf_lines + offset;
        for (unsigned j = 0; j < n_per_surf; ++j)
        {
            const geo_id_t id = geo_id_pack(lines[offset + j]);
            if (id.index != GEO_ID_INVALID && id.index >= this->n_lines)
            {
                PyErr_Format(PyExc_ValueError,
                             "Surface %u specified a line with index %u, which is larger than"
                             "the number of lines in the mesh (%u).",
                             i, id.index, this->n_lines);
                Py_DECREF(surfaces);
                Py_DECREF(this);
                return NULL;
            }
            surf_lines[j] = id;
        }

        geo_id_t end;
        {
            const geo_id_t id1 = surf_lines[n_per_surf - 1];
            if (id1.reverse)
            {
                end = this->lines[id1.index].begin;
            }
            else
            {
                end = this->lines[id1.index].end;
            }
        }
        for (unsigned j = 0; j < n_per_surf; ++j)
        {

            geo_id_t begin, new_end;
            const geo_id_t id2 = surf_lines[j];
            if (id2.reverse)
            {
                begin = this->lines[id2.index].end;
                new_end = this->lines[id2.index].begin;
            }
            else
            {
                begin = this->lines[id2.index].begin;
                new_end = this->lines[id2.index].end;
            }

            if (begin.index != end.index)
            {
                PyErr_Format(PyExc_ValueError,
                             "Line %u does not end (point %u) where line %u begins (point %u) for surface %u.",
                             j == 0 ? n_per_surf - 1 : j - 1, end.index, j, begin.index, i);
                Py_DECREF(surfaces);
                Py_DECREF(this);
                return NULL;
            }
            end = new_end;
        }
    }
    Py_DECREF(surfaces);

    return (PyObject *)this;
}

static PyMethodDef manifold2d_object_methods[] = {
    {.ml_name = "get_line",
     .ml_meth = manifold2d_get_line,
     .ml_flags = METH_O,
     .ml_doc = "get_line(index: int | GeoID, /) -> Line\n"
               "Get the line from the mesh.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "index : int or GeoID\n"
               "   Id of the line to get in 1-based indexing or GeoID. If negative,\n"
               "   the orientation will be reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Line\n"
               "   Line object corresponding to the ID.\n"},
    {.ml_name = "get_surface",
     .ml_meth = manifold2d_get_surface,
     .ml_flags = METH_O,
     .ml_doc = "get_surface(index: int | GeoID, /) -> Surface\n"
               "Get the surface from the mesh.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "index : int or GeoID\n"
               "   Id of the surface to get in 1-based indexing or GeoID. If negative,\n"
               "   the orientation will be reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Surface\n"
               "   Surface object corresponding to the ID.\n"},
    // {.ml_name = "compute_dual",
    //  .ml_meth = pyvl_mesh_compute_dual,
    //  .ml_flags = METH_NOARGS,
    //  .ml_doc = "Create dual to the mesh."},
    // {.ml_name = "to_element_connectivity",
    //  .ml_meth = pyvl_mesh_to_element_connectivity,
    //  .ml_flags = METH_NOARGS,
    //  .ml_doc = "Convert mesh connectivity to arrays list of element lengths and indices."},
    // {.ml_name = "merge_meshes",
    //  .ml_meth = (void *)pyvl_mesh_merge,
    //  .ml_flags = METH_CLASS | METH_FASTCALL,
    //  .ml_doc = "Merge sequence of meshes together into a single mesh."},
    // {.ml_name = "copy", .ml_meth = pyvl_mesh_copy, .ml_flags = METH_NOARGS, .ml_doc = "Create a copy of the mesh."},
    // {.ml_name = "line_gradient",
    //  .ml_meth = (void *)pyvl_mesh_line_gradient,
    //  .ml_flags = METH_FASTCALL,
    //  .ml_doc = "Compute line gradient from point values."},
    {.ml_name = "from_irregular",
     .ml_meth = (void *)maifold2d_from_irregular,
     .ml_flags = METH_VARARGS | METH_CLASS | METH_KEYWORDS,
     .ml_doc = "from_irregular(\n"
               "    n_points: int,\n"
               "    line_connectivity: array_like,\n"
               "    surface_connectivity: Sequence[array_like],\n"
               ") -> Self\n"
               "Create Manifold2D from surfaces with non-constant number of lines.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n_points : int\n"
               "    Number of points in the mesh.\n"
               "line_connectivity : (N, 2) array_like\n"
               "    Connectivity of points which form lines in 0-based indexing.\n"
               "surface_connectivity : Sequence of array_like\n"
               "    Sequence of arrays specifying connectivity of mesh surfaces in 1-based\n"
               "    indexing, where a negative value means that the line's orientation is\n"
               "    reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Two dimensional manifold.\n"},
    {.ml_name = "from_regular",
     .ml_meth = (void *)maifold2d_from_regular,
     .ml_flags = METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "from_regular(\n"
               "    n_points: int,\n"
               "    line_connectivity: array_like,\n"
               "    surface_connectivity: array_like,\n"
               ") -> Self\n"
               "Create Manifold2D from surfaces with constant number of lines.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n_points : int\n"
               "    Number of points in the mesh.\n"
               "line_connectivity : (N, 2) array_like\n"
               "    Connectivity of points which form lines in 0-based indexing.\n"
               "surface_connectivity : array_like\n"
               "    Two dimensional array-like object specifying connectivity of mesh\n"
               "    surfaces in 1-based indexing, where a negative value means that\n"
               "    the line's orientation is reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Two dimensional manifold.\n"},
    {0},
};

// static PyObject *pyvl_mesh_rich_compare(PyObject *self, PyObject *other, const int op)
// {
//     if (!PyObject_TypeCheck(other, &manifold2d_type_object) || (op != Py_EQ && op != Py_NE))
//     {
//         Py_RETURN_NOTIMPLEMENTED;
//     }
//     int res = 1;
//     const manifold2d_object_t *const this = (manifold2d_object_t *)self;
//     const manifold2d_object_t *const that = (manifold2d_object_t *)other;
//     if (this->n_points != that->n_points || this->n_lines != that->n_lines || this->n_surfaces != that->n_surfaces ||
//         memcmp(this->lines, that->lines, sizeof(*this->lines) * this->n_lines) != 0 ||
//         memcmp(this->surf_counts, that->surf_counts, sizeof(*this->surf_counts) * (this->n_surfaces + 1)) != 0 ||
//         memcmp(this->surf_lines, that->surf_lines, sizeof(*this->surf_lines) * this->surf_counts[this->n_surfaces])
//         !=
//             0)
//     {
//         res = 0;
//     }

//     res = (op == Py_EQ) ? res : !res;
//     if (res)
//     {
//         Py_RETURN_TRUE;
//     }
//     Py_RETURN_FALSE;
// }

INTERPLIB_INTERNAL
PyTypeObject manifold2d_type_object = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "interplib._mimetic.Manifold2D",
    .tp_basicsize = sizeof(manifold2d_object_t),
    .tp_itemsize = 0,
    .tp_str = manifold2d_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = manifold2d_type_docstring,
    .tp_methods = manifold2d_object_methods,
    .tp_getset = manifold2d_getset,
    // .tp_new = manifold2d_new,
    .tp_dealloc = manifold2d_dealloc,
    // .tp_richcompare = pyvl_mesh_rich_compare,
    .tp_base = &manifold_type_object,
};
