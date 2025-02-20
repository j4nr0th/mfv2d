//
// Created by jan on 19.2.2025.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include "../common.h"

typedef struct
{
    allocator_callbacks base;
    size_t capacity;
    size_t position;
    uint8_t *memory;
} allocator_stack_t;

allocator_stack_t *allocator_stack_create(size_t size, const allocator_callbacks *base);

size_t allocator_stack_usage(const allocator_stack_t *this);

void allocator_stack_reset(allocator_stack_t *this);

#endif // ALLOCATOR_H
