//
// Created by jan on 29.9.2024.
//

#ifndef ERROR_H
#define ERROR_H
#include "common_defines.h"

typedef enum
{
    INTERP_SUCCESS = 0,
    INTERP_ERROR_NOT_IN_DOMAIN,
    INTERP_ERROR_NOT_INCREASING,
    INTERP_ERROR_FAILED_ALLOCATION,
    INTERP_ERROR_BAD_SYSTEM,

    INTERP_ERROR_COUNT,
} interp_error_t;

INTERPLIB_INTERNAL
const char *interp_error_str(interp_error_t error);

INTERPLIB_INTERNAL
const char *interp_error_msg(interp_error_t error);

#endif // ERROR_H
