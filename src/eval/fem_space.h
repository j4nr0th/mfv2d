#ifndef FEM_SPACE_H
#define FEM_SPACE_H

#include "matrix.h"

/**
 * @brief Type describing FEM space defined by functions in 1D.
 */
typedef struct
{
    /// @brief Order of FEM Space.
    unsigned order;
    /// @brief Number of points where it is defined.
    unsigned n_pts;
    /// @brief Coordinates where values are defined on reference space (shape [n_pts]).
    const double *pnts;
    /// @brief Integration weights at integration points (shape [n_pts]).
    const double *wgts;
    /// @brief Values of nodal basis polynomials at integration points (shape [order + 1, n_pts]).
    const double *node;
    /// @brief Values of edge basis polynomials at the integration points (shape [order, n_pts]).
    const double *edge;
} fem_space_1d_t;

/**
 * @brief Two-dimensional FEM space which is the computed from outer-product of two 1D spaces.
 */
typedef struct
{
    /// @brief Space used in the first ("horizontal") dimension.
    fem_space_1d_t space_1;
    /// @brief Space used in the second ("vertical") dimension.
    fem_space_1d_t space_2;

    /**
     * @brief Array of 2x2 Jacobian matrix entries and its determinants.
     *
     * This array contains Jacobian matrices representing transformations
     * in a 2D space, along with their determinants. Each element contains
     * the components of the Jacobian matrix (j00, j01, j10, j11) and the determinant (det).
     *
     * This is used in for mapping between reference and physical spaces.
     */
    jacobian_t jacobian[];
} fem_space_2d_t;

typedef struct
{
    double x0, y0;
    double x1, y1;
    double x2, y2;
    double x3, y3;
} quad_info_t;

/**
 * @brief Computes the Jacobian matrix for a given quadrilateral and finite element method (FEM) space.
 *
 * This method computes the Jacobian matrix based on the geometry of the specified quadrilateral
 * and the properties of the provided 2D finite element space. The computed matrix is stored
 * in the output parameter.
 *
 * @param space_h A pointer to the 1D finite element space used for the first dimension.
 * @param space_v A pointer to the 1D finite element space used for the second dimension.
 * @param quad A pointer to a quadrilateral structure containing its geometric information.
 * @param p_out Pointer which receives the computed Jacobian entries.
 * @param allocator Allocator to use for memory allocation.
 *
 * @return An evaluation result indicating success or the type of failure encountered.
 */
MFV2D_INTERNAL
mfv2d_result_t fem_space_2d_create(const fem_space_1d_t *space_h, const fem_space_1d_t *space_v,
                                   const quad_info_t *quad, fem_space_2d_t **p_out,
                                   const allocator_callbacks *allocator);

/**
 * @brief Computes and assembles the nodal mass matrix for a given finite element space in 2D.
 *
 * This function calculates the nodal mass matrix for a specified 2D finite element space
 * using nodal basis functions defined by the space. The function also takes an allocator
 * for memory allocation tasks.
 *
 * @param space A pointer to the 2D finite element space structure (fem_space_2d_t)
 *        containing all required basis function and weight information.
 * @param p_out A pointer to the full matrix (matrix_full_t) structure where the
 *        final computed mass matrix will be stored.
 * @param allocator A pointer to the allocator_callbacks structure used for
 *        memory allocations during the matrix assembly process.
 * @return A result code indicating the status of the computation. Possible values include:
 *         - MFV2D_SUCCESS: Successful computation.
 *         - EVAL_FAILED_ALLOC: Memory allocation failed.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_node(const fem_space_2d_t *space, matrix_full_t *p_out,
                                        const allocator_callbacks *allocator);

/**
 * @brief Computes and assembles the edge mass matrix for a given finite element space in 2D.
 *
 * This function calculates the edge mass matrix for a specified 2D finite element space
 * using edge basis functions defined by the space. The function also takes an allocator
 * for memory allocation tasks.
 *
 * @param space A pointer to the 2D finite element space structure (fem_space_2d_t)
 *        containing all required basis function and weight information.
 * @param p_out A pointer to the full matrix (matrix_full_t) structure where the
 *        final computed mass matrix will be stored.
 * @param allocator A pointer to the allocator_callbacks structure used for
 *        memory allocations during the matrix assembly process.
 * @return A result code indicating the status of the computation. Possible values include:
 *         - MFV2D_SUCCESS: Successful computation.
 *         - EVAL_FAILED_ALLOC: Memory allocation failed.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge(const fem_space_2d_t *space, matrix_full_t *p_out,
                                        const allocator_callbacks *allocator);

/**
 * @brief Computes and assembles the surface mass matrix for a given finite element space in 2D.
 *
 * This function calculates the surface mass matrix for a specified 2D finite element space
 * using surface basis functions defined by the space. The function also takes an allocator
 * for memory allocation tasks.
 *
 * @param space A pointer to the 2D finite element space structure (fem_space_2d_t)
 *        containing all required basis function and weight information.
 * @param p_out A pointer to the full matrix (matrix_full_t) structure where the
 *        final computed mass matrix will be stored.
 * @param allocator A pointer to the allocator_callbacks structure used for
 *        memory allocations during the matrix assembly process.
 * @return A result code indicating the status of the computation. Possible values include:
 *         - MFV2D_SUCCESS: Successful computation.
 *         - EVAL_FAILED_ALLOC: Memory allocation failed.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_surf(const fem_space_2d_t *space, matrix_full_t *p_out,
                                        const allocator_callbacks *allocator);

MFV2D_INTERNAL
extern const char compute_element_mass_matrices_docstr[];

MFV2D_INTERNAL
PyObject *compute_element_mass_matrices(PyObject *self, PyObject *args, PyObject *kwargs);

/**
 * Creates FEM space from Python objects.
 *
 * All objects must be numpy arrays of dtype float64 and remain valid as long as the FEM space is used.
 *
 * @param order Order of the FEM space.
 * @param pts Positions of the integration points on the reference space.
 * @param wts Integration weights.
 * @param node_val Values of nodal basis functions at integration points.
 * @param edge_val Values of edge basis functions at integration points.
 * @param p_out Pointer to the output FEM space structure.
 * @return MFV2D_SUCCESS if successful, otherwise a non-zero error code.
 */
MFV2D_INTERNAL
mfv2d_result_t fem_space_1d_from_python(unsigned order, PyObject *pts, PyObject *wts, PyObject *node_val,
                                        PyObject *edge_val, fem_space_1d_t *p_out);

MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_node_edge(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field, int transpose);
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge_edge(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field, int dual);

MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge_surf(const fem_space_2d_t *fem_space, matrix_full_t *p_out,
                                             const allocator_callbacks *allocator, const double *field, int transpose);
/**
 * @brief Computes the nodal mass matrix for given input and output finite element spaces.
 *
 * This function calculates the nodal mass matrix by evaluating the overlap
 * of nodal basis functions between the input and output finite element spaces.
 * The computed matrix is stored in the provided output matrix structure.
 *
 * @param space_in Pointer to the input finite element space.
 * @param space_out Pointer to the output finite element space.
 * @param p_out Pointer to the full matrix where the computed mass matrix will be stored.
 * @param allocator Pointer to the allocator callbacks for memory allocation.
 * @return mfv2d_result_t The result of the computation. Returns `MFV2D_SUCCESS` if successful,
 *         `MFV2D_FAILED_ALLOC` if memory allocation fails, or `MFV2D_DIMS_MISMATCH` if the
 *         integration node dimensions of `space_in` and `space_out` do not match.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_node_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator);
/**
 * @brief Computes the edge mass matrix for given input and output finite element spaces.
 *
 * This function calculates the edge mass matrix by evaluating the overlap
 * of edge basis functions between the input and output finite element spaces.
 * The computed matrix is stored in the provided output matrix structure.
 *
 * @param space_in Pointer to the input finite element space.
 * @param space_out Pointer to the output finite element space.
 * @param p_out Pointer to the full matrix where the computed mass matrix will be stored.
 * @param allocator Pointer to the allocator callbacks for memory allocation.
 * @return mfv2d_result_t The result of the computation. Returns `MFV2D_SUCCESS` if successful,
 *         `MFV2D_FAILED_ALLOC` if memory allocation fails, or `MFV2D_DIMS_MISMATCH` if the
 *         integration node dimensions of `space_in` and `space_out` do not match.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_edge_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator);
/**
 * @brief Computes the surface mass matrix for given input and output finite element spaces.
 *
 * This function calculates the surface mass matrix by evaluating the overlap
 * of surface basis functions between the input and output finite element spaces.
 * The computed matrix is stored in the provided output matrix structure.
 *
 * @param space_in Pointer to the input finite element space.
 * @param space_out Pointer to the output finite element space.
 * @param p_out Pointer to the full matrix where the computed mass matrix will be stored.
 * @param allocator Pointer to the allocator callbacks for memory allocation.
 * @return mfv2d_result_t The result of the computation. Returns `MFV2D_SUCCESS` if successful,
 *         `MFV2D_FAILED_ALLOC` if memory allocation fails, or `MFV2D_DIMS_MISMATCH` if the
 *         integration node dimensions of `space_in` and `space_out` do not match.
 */
MFV2D_INTERNAL
mfv2d_result_t compute_mass_matrix_surf_double(const fem_space_2d_t *space_in, const fem_space_2d_t *space_out,
                                               matrix_full_t *p_out, const allocator_callbacks *allocator);

#endif // FEM_SPACE_H
