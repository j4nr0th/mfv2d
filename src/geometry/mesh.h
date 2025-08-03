#ifndef MESH_H
#define MESH_H

#include "../common/common.h"
#include "../common/error.h"
#include "manifold2d.h"

typedef enum
{
    ELEMENT_TYPE_LEAF = 0,
    ELEMENT_TYPE_NODE = 1,
} element_type_t;

typedef struct
{
    quad_info_t corners;
    index_2d_t orders;
} leaf_data;

typedef struct
{
    element_type_t type;
    unsigned parent;
} element_base_t;

typedef struct
{
    element_base_t base;
    leaf_data data;
    unsigned leaf_index;
} element_leaf_t;

typedef struct
{
    element_base_t base;
    union {
        unsigned children[4];
        struct
        {
            unsigned child_bottom_left;
            unsigned child_bottom_right;
            unsigned child_top_right;
            unsigned child_top_left;
        };
    };
} element_node_t;

typedef union {
    element_base_t base;
    element_node_t node;
    element_leaf_t leaf;
} element_t;

typedef struct
{
    const allocator_callbacks *allocator;
    unsigned count;
    unsigned capacity;
    unsigned leaf_count;
    unsigned leaves_at_last_indexing;
    element_t *elements;
} element_mesh_t;

typedef struct
{
    PyObject_HEAD;
    element_mesh_t element_mesh;
    manifold2d_object_t *primal;
    manifold2d_object_t *dual;
    unsigned boundary_count;
    unsigned *boundary_indices;
} mesh_t;

MFV2D_INTERNAL
extern PyTypeObject mesh_type_object;

typedef enum
{
    ELEMENT_SIDE_BOTTOM = 1,
    ELEMENT_SIDE_RIGHT = 2,
    ELEMENT_SIDE_TOP = 3,
    ELEMENT_SIDE_LEFT = 4,
} element_side_t;

#endif // MESH_H
