"""Package dedicated to solving PDEs on 2D quad meshes.

This file includes re-exports types and functions that are expected to be used
by users, either for directly creating them, or to just use them for type-hinting.
"""

# Boundary
from mfv2d.boundary import BoundaryCondition2DSteady as BoundaryCondition2DSteady

# K-forms
from mfv2d.kform import KEquation as KEquation
from mfv2d.kform import KFormSystem as KFormSystem
from mfv2d.kform import KFormUnknown as KFormUnknown
from mfv2d.kform import KWeight as KWeight

# Mimetic2D
from mfv2d.mimetic2d import ElementLeaf2D as ElementLeaf2D
from mfv2d.mimetic2d import Mesh2D as Mesh2D

# Solve system
from mfv2d.solve_system import RefinementSettings as RefinementSettings
from mfv2d.solve_system import SolverSettings as SolverSettings
from mfv2d.solve_system import SystemSettings as SystemSettings
from mfv2d.solve_system import TimeSettings as TimeSettings

# Actual solving
from mfv2d.solve_system_2d import solve_system_2d as solve_system_2d
