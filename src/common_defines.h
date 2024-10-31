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
#   define INTERPLIB_INTERNAL __attribute__((visibility("hidden")))
#   define INTERPLIB_EXTERNAL __attribute__((visibility("default")))

#   define INTERPLIB_ARRAY_ARG(arr, sz) arr[sz]

#   define INTERPLIB_EXPECT_CONDITION(x) (__builtin_expect(x, 1))

#endif


#ifndef ASSERT
#   ifdef INTERPLIB_ASSERTS
/**
 * @brief is a macro, which tests a condition and only evaluates it once. If it is false, then it is reported to
 * stderr. The macro returns !(condition), so if condition holds it returns 0 and if it does not it returns 0.
 * The intended usage is as follows:
 *
 * if (ASSERT(cnd))
 * {
 *     return ERROR_CODE;
 * }
 *
 * @note ASSERT does all this only when building in Debug mode. For Release configuration, the macro is replaced
 * with compiler specific assume directive, or a zero if that is not known for the specific compiler used.
 */
#      define ASSERT(condition, message) ((condition) ? 0 : (fprintf(stderr, "%s:%d: %s: Assertion '%s' failed - %s\n", __FILE__, __LINE__, __func__, #condition, (message)), 1))
#   else
#      ifdef __GNUC__
#          define ASSUME(condition, message) __assume(condition)
#      endif
#      ifndef ASSERT
#          define ASSERT(condition, message) 0
#      endif
#   endif
#endif





#ifndef INTERPLIB_EXPECT_CONDITION
#   define INTERPLIB_EXPECT_CONDITION(x) (x)
#endif


#ifndef INTERPLIB_INTERNAL
#   define INTERPLIB_INTERNAL
#endif

#ifndef INTERPLIB_EXTERNAL
#   define INTERPLIB_EXTERNAL
#endif

#ifndef INTERPLIB_ARRAY_ARG
#   define INTERPLIB_ARRAY_ARG(arr, sz) *arr
#endif


//  Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#   define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#   define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

//  Prevent numpy from being re-imported
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#   define NO_IMPORT_ARRAY
#   define PY_ARRAY_UNIQUE_SYMBOL _interp
#endif

#include <Python.h>
#include <stdio.h>

#endif //COMMON_DEFINES_H
