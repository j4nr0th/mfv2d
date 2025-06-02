"""Package dedicated to interpolation using data defined on different topologies."""

# Differential forms
from interplib import kforms as kforms

# Mimetic stuff
from interplib import mimetic as mimetic

# C base types
from interplib._interp import Basis1D as Basis1D
from interplib._interp import Polynomial1D as Polynomial1D
from interplib._interp import Spline1D as Spline1D
from interplib._interp import Spline1Di as Spline1Di
from interplib._interp import bernstein1d as bernstein1d
from interplib._interp import bernstein_coefficients as bernstein_coefficients
from interplib._interp import compute_gll as compute_gll
from interplib._interp import dlagrange1d as dlagrange1d
from interplib._interp import lagrange1d as lagrange1d
from interplib._interp import test as test

# Bernstein polynomials
from interplib.bernstein import Bernstein1D as Bernstein1D

# C wrapper functions
from interplib.lagrange import (
    interp1d_2derivative_samples as interp1d_2derivative_samples,
)
from interplib.lagrange import (
    interp1d_derivative_samples as interp1d_derivative_samples,
)
from interplib.lagrange import interp1d_function_samples as interp1d_function_samples
from interplib.lagrange import (
    lagrange_2derivative_samples as lagrange_2derivative_samples,
)
from interplib.lagrange import (
    lagrange_derivative_samples as lagrange_derivative_samples,
)
from interplib.lagrange import lagrange_function_samples as lagrange_function_samples

# Product Basis
from interplib.product_basis import BasisProduct2D as BasisProduct2D
from interplib.solve_system import RefinementSettings as RefinementSettings

# Solve system
from interplib.solve_system import SolverSettings as SolverSettings
from interplib.solve_system import SystemSettings as SystemSettings
from interplib.solve_system import TimeSettings as TimeSettings

# Splines
from interplib.splines import SplineBC as SplineBC

# TODO: maybe fix this some day.
# from interplib.splines import (
#     element_interpolating_spline as element_interpolating_spline,
# )
from interplib.splines import (
    element_interpolating_splinei as element_interpolating_splinei,
)

# from interplib.splines import nodal_interpolating_spline as nodal_interpolating_spline
from interplib.splines import nodal_interpolating_splinei as nodal_interpolating_splinei

# System
from interplib.system import solve_system_1d as solve_system_1d
