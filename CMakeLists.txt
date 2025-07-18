cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "mfv2d")
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

set(MFV2D_SOURCE_FILES
        # Common
        src/_module.c
        src/common.c
        src/error.c
        # Basis
        src/basis/lagrange.c
        src/basis/gausslobatto.c
        # Topology
        src/topology/geoidobject.c
        src/topology/lineobject.c
        src/topology/surfaceobject.c
        src/topology/manifold.c
        src/topology/manifold2d.c
        src/topology/mesh.c
        # Evaluation
        src/eval/allocator.c
        src/eval/incidence.c
        src/eval/bytecode.c
        src/eval/matrix.c
        src/eval/element_system.c
        src/eval/system_template.c
        src/eval/fem_space.c
        src/eval/element_eval.c
        src/eval/basis.c
        src/eval/element_cache.c
        # Solver
        src/solve/qr_solve.c
        src/solve/svector.c
        src/solve/givens.c
        src/solve/lil_matrix.c
)
set(MFV2D_HEADER_FILES
        # Common
        src/module.h
        src/common.h
        src/common_defines.h
        src/error.h
        # Basis
        src/basis/lagrange.h
        src/basis/gausslobatto.h
        # Topology
        src/topology/geoidobject.h
        src/topology/lineobject.h
        src/topology/surfaceobject.h
        src/topology/manifold.h
        src/topology/manifold2d.h
        src/topology/mesh.h
        # Evaluation
        src/eval/allocator.h
        src/eval/incidence.h
        src/eval/bytecode.h
        src/eval/matrix.h
        src/eval/element_system.h
        src/eval/system_template.h
        src/eval/fem_space.h
        src/eval/element_eval.h
        src/eval/basis.h
        src/eval/element_cache.h
        # Solver
        src/solve/qr_solve.h
        src/solve/svector.h
        src/solve/givens.h
        src/solve/lil_matrix.h
)

Python_add_library(_mfv2d
        MODULE
        ${MFV2D_SOURCE_FILES}
        WITH_SOABI
)
install(TARGETS _mfv2d DESTINATION ${SKBUILD_PROJECT_NAME})
set_property(TARGET _mfv2d PROPERTY C_STANDARD 17)

target_compile_definitions(_mfv2d PRIVATE MFV2D_ASSERTS)
if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_mfv2d PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID EQUAL "GNU")
    target_compile_options(_mfv2d PRIVATE -Wall -Wextra -fwhole-program -flto=auto -fanalyzer)# -g -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g)
endif ()

if (OPENMP_FOUND)
    target_compile_options(_mfv2d PRIVATE ${OpenMP_C_FLAGS})
    target_include_directories(_mfv2d PRIVATE ${OpenMP_C_INCLUDE_DIRS})
    target_link_libraries(_mfv2d PRIVATE ${OpenMP_C_LIBRARIES})
endif ()

enable_testing()
add_test(NAME pytest COMMAND ${Python_EXECUTABLE} -m pytest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
