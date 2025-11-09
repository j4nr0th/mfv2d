//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common/common.h"

typedef struct
{
    PyObject_HEAD;
    unsigned order;
    double *nodes;
    double *weights;
} integration_rule_1d_t;

MFV2D_INTERNAL
// extern PyTypeObject integration_rule_1d_type;
extern PyType_Spec integration_rule_1d_type_spec;

#endif // GAUSSLOBATTO_H
