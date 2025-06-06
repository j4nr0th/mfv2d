#ifndef BASIS_H
#define BASIS_H
#include "../basis/gausslobatto.h"
#include "../common.h"
#include "fem_space.h"

typedef struct
{
    PyObject_HEAD;
    integration_rule_1d_t *integration_rule;
    unsigned order;
    double *nodal_basis;
    double *edge_basis;
} basis_1d_t;

MFV2D_INTERNAL
extern PyTypeObject basis_1d_type;

static inline fem_space_1d_t basis_1d_as_fem_space(const basis_1d_t *self)
{
    return (fem_space_1d_t){
        .order = self->order,
        .wgts = self->integration_rule->weights,
        .node = self->nodal_basis,
        .edge = self->edge_basis,
        .n_pts = self->integration_rule->order + 1,
        .pnts = self->integration_rule->nodes,
    };
}

typedef struct
{
    PyObject_HEAD;
    basis_1d_t *basis_xi;
    basis_1d_t *basis_eta;
} basis_2d_t;

MFV2D_INTERNAL
extern PyTypeObject basis_2d_type;

#endif // BASIS_H
