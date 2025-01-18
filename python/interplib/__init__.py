"""Package dedicated to interpolation using data defined on different topologies."""

# C base types
from interplib._interp import Basis1D as Basis1D
from interplib._interp import Polynomial1D as Polynomial1D
from interplib._interp import Spline1D as Spline1D
from interplib._interp import Spline1Di as Spline1Di
from interplib._interp import test as test

# Bernstein polynomials
from interplib.bernstein import Bernstein1D as Bernstein1D

# Differential forms
from interplib.kform import KFormDerivative as KFormDerivative
from interplib.kform import KFormDual as KFormDual
from interplib.kform import KFormEquaton as KFormEquaton
from interplib.kform import KFormInnerProduct as KFormInnerProduct
from interplib.kform import KFormPrimal as KFormPrimal
from interplib.kform import KFormProjection as KFormProjection
from interplib.kform import KFormSum as KFormSum
from interplib.kform import KFromSystem as KFromSystem
from interplib.kform import element_system as element_system

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

# Mimetic stuff
from interplib.mimetic import Element1D as Element1D

# Product Basis
from interplib.product_basis import BasisProduct2D as BasisProduct2D

# Splines
from interplib.splines import SplineBC as SplineBC
from interplib.splines import element_interpolating_spline as element_interpolating_spline
from interplib.splines import (
    element_interpolating_splinei as element_interpolating_splinei,
)
from interplib.splines import nodal_interpolating_spline as nodal_interpolating_spline
from interplib.splines import nodal_interpolating_splinei as nodal_interpolating_splinei
