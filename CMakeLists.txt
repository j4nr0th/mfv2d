cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "InterpLib")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)


execute_process(
        COMMAND "${Python3_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)


set(INTERP_SOURCE_FILES
        src/_module.c
        src/lagrange.c
        src/error.c
        src/rbf.c
        src/cubic_splines.c
        src/common.c
        src/basis1d.c
        src/poly_basis.c
)
set(INTERP_HEADER_FILES
        src/common_defines.h
        src/lagrange.h
        src/error.h
        src/rbf.h
        src/cubic_splines.h
        src/common.h
        src/basis1d.h
        src/poly_basis.h
)

Python_add_library(_interp MODULE ${INTERP_SOURCE_FILES} WITH_SOABI USE_SABI 3.10)
install(TARGETS _interp DESTINATION ${SKBUILD_PROJECT_NAME})


# if (CMAKE_BUILD_TYPE EQUAL Debug)
target_compile_definitions(_interp PRIVATE INTERPLIB_ASSERTS)
# endif ()
message("Got numpy at ${NumPy_INCLUDE_DIRS}")
target_include_directories(_interp PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID EQUAL "GNU")
    target_compile_options(_interp PRIVATE -Wall -Wextra)
endif ()

enable_testing()
add_test(NAME pytest COMMAND ${Python3_EXECUTABLE} -m pytest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
