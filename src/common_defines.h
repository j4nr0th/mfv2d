//
// Created by jan on 29.9.2024.
//

#ifndef COMMON_DEFINES_H
#define COMMON_DEFINES_H

#define PY_SSIZE_T_CLEAN
#ifndef PY_LIMITED_API
#define PY_LIMITED_API 0x030A0000
#endif

#ifdef __GNUC__
#define MFV2D_INTERNAL __attribute__((visibility("hidden")))
#define MFV2D_EXTERNAL __attribute__((visibility("default")))

#define MFV2D_ARRAY_ARG(arr, sz) arr[sz]

#define MFV2D_EXPECT_CONDITION(x) (__builtin_expect(x, 1))

#endif

#ifndef ASSERT
#ifdef MFV2D_ASSERTS
/**
 * Macro used to check the condition holds at runtime for Debug builds. If the specified condition is false,
 * the program will terminate. In Release configuration, the macro is replaced with a compiler-specific "assume"
 * directive, or a zero if that is not known for the specific compiler used.
 */
#define ASSERT(condition, message, ...)                                                                                \
    ((condition) ? 0                                                                                                   \
                 : (fprintf(stderr, "%s:%d: %s: Assertion '%s' failed - " message "\n", __FILE__, __LINE__, __func__,  \
                            #condition __VA_OPT__(, ) __VA_ARGS__),                                                    \
                    exit(EXIT_FAILURE), 1))
#else
#ifdef __GNUC__
#define ASSUME(condition, message) __assume(condition)
#endif
#ifndef ASSERT
#define ASSERT(condition, message) 0
#endif
#endif
#endif

#ifndef MFV2D_EXPECT_CONDITION
#define MFV2D_EXPECT_CONDITION(x) (x)
#endif

#ifndef MFV2D_INTERNAL
#define MFV2D_INTERNAL
#endif

#ifndef MFV2D_EXTERNAL
#define MFV2D_EXTERNAL
#endif

#ifdef MFV2D_EXPORT_ALL
#undef MFV2D_INTERNAL
#define MFV2D_INTERNAL MFV2D_EXTERNAL
#endif

#ifndef MFV2D_ARRAY_ARG
#define MFV2D_ARRAY_ARG(arr, sz) *arr
#endif

//  Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

//  Prevent numpy from being re-imported
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _mfv2d
#endif

#include <Python.h>
#include <stdio.h>

#endif // COMMON_DEFINES_H
