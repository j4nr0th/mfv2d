cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "InterpLib")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)


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

Python_add_library(_interp MODULE ${INTERP_SOURCE_FILES} ${COMMON_SOURCE_FILES} WITH_SOABI)# USE_SABI 3.10) STABLE API HATE!
Python_add_library(_mimetic MODULE ${MIMETIC_SOURCE_FILES} ${COMMON_SOURCE_FILES} WITH_SOABI)# USE_SABI 3.10) STABLE API HATE!
install(TARGETS _interp _mimetic DESTINATION ${SKBUILD_PROJECT_NAME})

#if (CMAKE_BUILD_TYPE EQUAL Debug)
target_compile_definitions(_interp PRIVATE INTERPLIB_ASSERTS)
target_compile_definitions(_mimetic PRIVATE INTERPLIB_ASSERTS)
#endif()
if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_interp PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
target_include_directories(_mimetic PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID EQUAL "GNU")
    target_compile_options(_interp PRIVATE -Wall -Wextra)
    target_compile_options(_mimetic PRIVATE -Wall -Wextra)
endif ()

enable_testing()
add_test(NAME pytest COMMAND ${Python_EXECUTABLE} -m pytest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
