//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _mfv2d
#include "common/common_defines.h"
//  Common definitions

//  Python
#include <Python.h>
//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

//  Geometry
#include "geometry/geoidobject.h"
#include "geometry/lineobject.h"
#include "geometry/manifold2d.h"
#include "geometry/mesh.h"
#include "geometry/surfaceobject.h"

// Evaluation
#include "evaluation/bytecode.h"
#include "evaluation/element_system.h"
#include "evaluation/incidence.h"
#include "evaluation/integrating_fields.h"

// Fem space
#include "fem_space/basis.h"
#include "fem_space/element_fem_space.h"
#include "fem_space/fem_space.h"
#include "fem_space/integration_rule.h"

// Polynomials
#include "polynomials/gauss_lobatto.h"
#include "polynomials/lagrange.h"
#include "polynomials/legendre.h"

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyMethodDef module_methods[] = {
    {
        "check_bytecode",
        check_bytecode,
        METH_O,
        check_bytecode_docstr,
    },
    {
        "check_incidence",
        (void *)check_incidence,
        METH_VARARGS | METH_KEYWORDS,
        check_incidence_docstring,
    },
    {
        .ml_name = "lagrange1d",
        .ml_meth = interp_lagrange,
        .ml_flags = METH_VARARGS,
        .ml_doc = interp_lagrange_doc,
    },
    {
        .ml_name = "dlagrange1d",
        .ml_meth = interp_dlagrange,
        .ml_flags = METH_VARARGS,
        .ml_doc = interp_dlagrange_doc,
    },
    {
        .ml_name = "compute_gll",
        .ml_meth = (void *)compute_gauss_lobatto_nodes,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_gll_docstring,
    },
    {
        .ml_name = "compute_element_matrix_test",
        .ml_meth = (void *)compute_element_mass_matrices,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_element_mass_matrices_docstr,
    },
    {
        .ml_name = "compute_element_matrix",
        .ml_meth = (void *)compute_element_matrix,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_element_matrix_docstr,
    },
    {
        .ml_name = "compute_element_projector",
        .ml_meth = (void *)compute_element_projector,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_element_projector_docstr,
    },
    {
        .ml_name = "compute_element_mass_matrix",
        .ml_meth = (void *)compute_element_mass_matrix,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_element_mass_matrix_docstr,
    },
    {
        .ml_name = "compute_element_vector",
        .ml_meth = (void *)compute_element_vector,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_element_vector_docstr,
    },
    {
        .ml_name = "compute_legendre",
        .ml_meth = (void *)compute_legendre_polynomials,
        .ml_doc = compute_legendre_polynomials_docstring,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
    },
    {
        .ml_name = "compute_integrating_fields",
        .ml_meth = (void *)compute_integrating_fields,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_integrating_fields_docstring,
    },
    {NULL, NULL, 0, NULL}, // sentinel
};

static PyModuleDef module = {.m_base = PyModuleDef_HEAD_INIT,
                             .m_name = "mfv2d._mfv2d",
                             .m_doc = "Internal C-extension implementing required functionality.",
                             .m_size = -1,
                             .m_methods = module_methods,
                             .m_slots = NULL,
                             .m_traverse = NULL,
                             .m_clear = NULL,
                             .m_free = NULL};

PyMODINIT_FUNC PyInit__mfv2d(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }

    PyObject *mod = NULL;
    if (!((mod = PyModule_Create(&module))) || PyModule_AddType(mod, &geo_id_type_object) < 0 ||
        PyModule_AddType(mod, &line_type_object) < 0 || PyModule_AddType(mod, &surface_type_object) < 0 ||
        PyModule_AddType(mod, &manifold2d_type_object) < 0 || PyModule_AddType(mod, &integration_rule_1d_type) < 0 ||
        PyModule_AddType(mod, &basis_1d_type) < 0 || PyModule_AddType(mod, &basis_2d_type) < 0 ||
        PyModule_AddType(mod, &element_fem_space_2d_type) < 0 || PyModule_AddType(mod, &mesh_type_object) < 0 ||
        PyModule_AddIntMacro(mod, ELEMENT_SIDE_BOTTOM) < 0 || PyModule_AddIntMacro(mod, ELEMENT_SIDE_RIGHT) < 0 ||
        PyModule_AddIntMacro(mod, ELEMENT_SIDE_TOP) < 0 || PyModule_AddIntMacro(mod, ELEMENT_SIDE_LEFT) < 0 ||
        PyModule_AddIntMacro(mod, MATOP_INVALID) < 0 || PyModule_AddIntMacro(mod, MATOP_IDENTITY) < 0 ||
        PyModule_AddIntMacro(mod, MATOP_MASS) < 0 || PyModule_AddIntMacro(mod, MATOP_INCIDENCE) < 0 ||
        PyModule_AddIntMacro(mod, MATOP_PUSH) < 0 || PyModule_AddIntMacro(mod, MATOP_MATMUL) < 0 ||
        PyModule_AddIntMacro(mod, MATOP_SCALE) < 0 || PyModule_AddIntMacro(mod, MATOP_SUM) < 0 ||
        PyModule_AddIntMacro(mod, MATOP_INTERPROD) < 0)
    {
        Py_XDECREF(mod);
        return NULL;
    }

    return mod;
}
