//
// Created by jan on 29.9.2024.
//

#include "error.h"

#define ERROR_ENUM_ENTRY(entry, msg) [(entry)] = {#entry, (msg)}

static const struct {const char* str, *msg;} error_messages[INTERP_ERROR_COUNT] = {
    ERROR_ENUM_ENTRY(INTERP_SUCCESS, "Success"),
    ERROR_ENUM_ENTRY(INTERP_ERROR_NOT_IN_DOMAIN, "Argument was not inside the domain."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_NOT_INCREASING, "Input was not monotonically increasing."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_FAILED_ALLOCATION, "Could not allocate desired amount of memory."),
};


const char* interp_error_str(interp_error_t error)
{
    if (ASSERT(error > 0 && error < INTERP_ERROR_COUNT, "Error enum is out of range")) return "UNKNOWN";
    return error_messages[error].str;
}

const char* interp_error_msg(interp_error_t error)
{
    if (ASSERT(error > 0 && error < INTERP_ERROR_COUNT, "Error enum is out of range")) return "UNKNOWN";
    return error_messages[error].msg;
}
