//
// Created by jan on 27.6.2025.
//

#ifndef ELEMENT_EQUATION_H
#define ELEMENT_EQUATION_H

#include "../common.h"

typedef struct
{
    PyObject_HEAD;
    unsigned element;   // Index/ID of the element these belong to.
    Py_ssize_t n_pairs; // Number of DoF index-coefficient pairs
    unsigned *dofs;     // Degree of freedom indices
    double *coeffs;     // Degree of freedom coefficients
} element_dof_equation_t;

MFV2D_INTERNAL
extern PyTypeObject element_dof_equation_type;

#endif // ELEMENT_EQUATION_H
