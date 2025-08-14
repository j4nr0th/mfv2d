#ifndef MFV2D_CRS_MATRIX_H
#define MFV2D_CRS_MATRIX_H
#include "../common/common.h"
#include <jmtx/double/matrices/sparse_row_compressed.h>

typedef struct
{
    PyObject_HEAD;
    jmtxd_matrix_crs *matrix;
    unsigned built_rows;
} crs_matrix_t;

MFV2D_INTERNAL
extern PyTypeObject crs_matrix_type_object;

static inline int check_jmtx_call(const jmtx_result res, const char *file, const int line, const char *func,
                                  const char *call)
{
    if (res == JMTX_RESULT_SUCCESS)
    {
        return 1;
    }

    raise_exception_from_current(PyExc_RuntimeError, "%s:%u - %s: Failed a jmtx call \"%s\" with error code %s(%u)\n",
                                 file, line, func, call, jmtx_result_to_str(res), res);
    return 0;
}
#ifndef JMTX_SUCCEEDED
#define JMTX_SUCCEEDED(call) check_jmtx_call((call), __FILE__, __LINE__, __func__, #call)
#endif // JMTX_SUCCEEDED

#endif // MFV2D_CRS_MATRIX_H
