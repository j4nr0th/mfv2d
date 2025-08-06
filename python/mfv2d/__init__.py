"""Package dedicated to solving PDEs on 2D quad meshes.

This file includes re-exports types and functions that are expected to be used
by users, either for directly creating them, or to just use them for type-hinting.
"""

# _mfv2d
from mfv2d._mfv2d import Mesh as Mesh

# Boundary
from mfv2d.boundary import BoundaryCondition2DSteady as BoundaryCondition2DSteady

# K-forms
from mfv2d.kform import KEquation as KEquation
from mfv2d.kform import KFormUnknown as KFormUnknown
from mfv2d.kform import KWeight as KWeight
from mfv2d.kform import UnknownFormOrder as UnknownFormOrder

# Mimetic2D
from mfv2d.mimetic2d import mesh_create as mesh_create

# Refinement settings
from mfv2d.refinement import ErrorEstimateCustom as ErrorEstimateCustom
from mfv2d.refinement import ErrorEstimateLocalInverse as ErrorEstimateLocalInverse
from mfv2d.refinement import RefinementLimitElementCount as RefinementLimitElementCount
from mfv2d.refinement import RefinementLimitErrorValue as RefinementLimitErrorValue
from mfv2d.refinement import RefinementLimitUnknownCount as RefinementLimitUnknownCount
from mfv2d.refinement import RefinementSettings as RefinementSettings
from mfv2d.refinement import (
    compute_legendre_coefficients as compute_legendre_coefficients,
)
from mfv2d.refinement import (
    compute_legendre_error_estimates as compute_legendre_error_estimates,
)

# Solve system
from mfv2d.solve_system import SolutionStatistics as SolutionStatistics
from mfv2d.solve_system import SolverSettings as SolverSettings
from mfv2d.solve_system import SystemSettings as SystemSettings
from mfv2d.solve_system import TimeSettings as TimeSettings

# Actual solving
from mfv2d.solve_system_2d import solve_system_2d as solve_system_2d

# System
from mfv2d.system import KFormSystem as KFormSystem
