cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "InterpLib")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(OpenMP)

execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(COMMON_SOURCE_FILES
        src/common.c
        src/error.c
)
set(COMMON_HEADER_FILES
        src/common.h
        src/common_defines.h
        src/error.h
)

set(INTERP_SOURCE_FILES
        src/interp/_module.c
        src/interp/lagrange.c
        src/interp/cubic_splines.c
        src/interp/basis1d.c
        src/interp/polynomial1d.c
        src/interp/spline1d.c
        src/interp/bernstein.c
        src/interp/gausslobatto.c
)
set(INTERP_HEADER_FILES
        src/interp/lagrange.h
        src/interp/cubic_splines.h
        src/interp/basis1d.h
        src/interp/polynomial1d.h
        src/interp/spline1d.h
        src/interp/bernstein.h
        src/interp/gausslobatto.h
)

set(MIMETIC_SOURCE_FILES
        src/mimetic/_module.c
        src/mimetic/geoidobject.c
        src/mimetic/lineobject.c
        src/mimetic/surfaceobject.c
        src/mimetic/manifold.c
        src/mimetic/manifold1d.c
        src/mimetic/manifold2d.c
)
set(MIMETIC_HEADER_FILES
        src/mimetic/module.h
        src/mimetic/geoidobject.h
        src/mimetic/lineobject.h
        src/mimetic/surfaceobject.h
        src/mimetic/manifold.h
        src/mimetic/manifold1d.h
        src/mimetic/manifold2d.h
)

set(EVAL_SOURCE_FILES
        src/mimetic/eval/allocator.c
        src/mimetic/eval/error.c
        src/mimetic/eval/evaluation.c
        src/mimetic/eval/incidence.c
        src/mimetic/eval/precomp.c
        src/mimetic/eval/bytecode.c
        src/mimetic/eval/matrix.c
        src/mimetic/eval/connectivity.c
)
set(EVAL_HEADER_FILES
        src/mimetic/eval/error.h
        src/mimetic/eval/allocator.h
        src/mimetic/eval/evaluation.h
        src/mimetic/eval/incidence.h
        src/mimetic/eval/precomp.h
        src/mimetic/eval/bytecode.h
        src/mimetic/eval/matrix.h
        src/mimetic/eval/connectivity.h
)

set(MAT_HEADER_FILES
        src/mimetic/solve/qr_solve.h
        src/mimetic/solve/svector.h
        src/mimetic/solve/givens.h
        src/mimetic/solve/lil_matrix.h
)

set(MAT_SOURCE_FILES
        src/mimetic/solve/qr_solve.c
        src/mimetic/solve/svector.c
        src/mimetic/solve/givens.c
        src/mimetic/solve/lil_matrix.c
)

Python_add_library(_interp MODULE ${INTERP_SOURCE_FILES} ${COMMON_SOURCE_FILES} WITH_SOABI)# USE_SABI 3.10) STABLE API HATE!
Python_add_library(_mimetic
        MODULE
        ${MIMETIC_SOURCE_FILES}
        ${EVAL_SOURCE_FILES}
        ${MAT_SOURCE_FILES}
        ${COMMON_SOURCE_FILES}
        WITH_SOABI
)# USE_SABI 3.10) STABLE API HATE!
# Python_add_library(_eval MODULE ${EVAL_SOURCE_FILES} ${COMMON_SOURCE_FILES} WITH_SOABI)# USE_SABI 3.10) STABLE API HATE!
install(TARGETS _interp _mimetic DESTINATION ${SKBUILD_PROJECT_NAME})

#if (CMAKE_BUILD_TYPE EQUAL Debug)
target_compile_definitions(_interp PRIVATE INTERPLIB_ASSERTS)
target_compile_definitions(_mimetic PRIVATE INTERPLIB_ASSERTS)
# target_compile_definitions(_eval PRIVATE INTERPLIB_ASSERTS)
#endif()
if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_interp PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
target_include_directories(_mimetic PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
# target_include_directories(_eval PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID EQUAL "GNU")
    target_compile_options(_interp PRIVATE -Wall -Wextra)
    target_compile_options(_mimetic PRIVATE -Wall -Wextra -fwhole-program -flto=auto)# -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g)
    # target_compile_options(_eval PRIVATE -Wall -Wextra -fwhole-program -flto=auto)# -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g)
endif ()

if (OPENMP_FOUND)
    target_compile_options(_mimetic PRIVATE ${OpenMP_C_FLAGS})
    target_include_directories(_mimetic PRIVATE ${OpenMP_C_INCLUDE_DIRS})
    target_link_libraries(_mimetic PRIVATE ${OpenMP_C_LIBRARIES})
endif ()

enable_testing()
add_test(NAME pytest COMMAND ${Python_EXECUTABLE} -m pytest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
