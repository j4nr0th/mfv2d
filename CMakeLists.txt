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
        src/module.c
        # common
        src/common/allocator.c
        src/common/common.c
        src/common/error.c
        # evaluation
        src/evaluation/bytecode.c
        src/evaluation/element_eval.c
        src/evaluation/element_system.c
        src/evaluation/forms.c
        src/evaluation/incidence.c
        src/evaluation/integrating_fields.c
        src/evaluation/matrix.c
        src/evaluation/system_template.c
        # fem_space
        src/fem_space/basis.c
        src/fem_space/element_fem_space.c
        src/fem_space/fem_space.c
        src/fem_space/integration_rule.c
        # geometry
        src/geometry/geoidobject.c
        src/geometry/lineobject.c
        src/geometry/manifold2d.c
        src/geometry/mesh.c
        src/geometry/surfaceobject.c
        # polynomials
        src/polynomials/gauss_lobatto.c
        src/polynomials/lagrange.c
        src/polynomials/legendre.c
)
set(MFV2D_HEADER_FILES
        src/module.h
        # common
        src/common/allocator.h
        src/common/common.h
        src/common/error.h
        # evaluation
        src/evaluation/bytecode.h
        src/evaluation/element_eval.h
        src/evaluation/element_system.h
        src/evaluation/forms.h
        src/evaluation/incidence.h
        src/evaluation/integrating_fields.h
        src/evaluation/matrix.h
        src/evaluation/system_template.h
        # fem_space
        src/fem_space/basis.h
        src/fem_space/element_fem_space.h
        src/fem_space/fem_space.h
        src/fem_space/integration_rule.h
        # geometry
        src/geometry/geoidobject.h
        src/geometry/lineobject.h
        src/geometry/manifold2d.h
        src/geometry/mesh.h
        src/geometry/surfaceobject.h
        # polynomials
        src/polynomials/gauss_lobatto.h
        src/polynomials/lagrange.h
        src/polynomials/legendre.h
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
