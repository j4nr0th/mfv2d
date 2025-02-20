//
// Created by jan on 19.2.2025.
//

#include "allocator.h"

static size_t round_to_dword(const size_t sz)
{
    const size_t rem = sz & 7;
    if (rem)
    {
        return sz + (8 - rem);
    }
    return sz;
}

static void *stack_allocator_allocate(void *param, size_t sz)
{
    allocator_stack_t *const this = param;
    sz = round_to_dword(sz);
    const size_t new_pos = this->position + sz;
    if (this->capacity < new_pos)
    {
        return NULL;
    }
    uint8_t *const ptr = this->memory + this->position;
    // memset(ptr, 0, sz);
    this->position = new_pos;
    return ptr;
}

static void *stack_allocator_reallocate(void *param, void *ptr, size_t sz)
{
    (void)param;
    (void)ptr;
    (void)sz;
    return NULL;
}

static void stack_allocator_free(void *param, void *ptr)
{
    (void)param;
    (void)ptr;
}

static const allocator_callbacks STACK_ALLOCATOR_BASE = {
    .alloc = stack_allocator_allocate,
    .realloc = stack_allocator_reallocate,
    .free = stack_allocator_free,
    .state = NULL,
};

allocator_stack_t *allocator_stack_create(size_t size, const allocator_callbacks *base)
{
    allocator_stack_t *const this = allocate(base, sizeof *this);
    if (!this)
        return NULL;

    void *const memory = allocate(base, size);
    if (!memory)
    {
        deallocate(base, this);
        return NULL;
    }
    *this = (allocator_stack_t){.base = STACK_ALLOCATOR_BASE, .capacity = size, .position = 0, .memory = memory};
    this->base.state = this;
    return this;
}

size_t allocator_stack_usage(const allocator_stack_t *this)
{
    return this->position;
}

void allocator_stack_reset(allocator_stack_t *this)
{
    this->position = 0;
}
