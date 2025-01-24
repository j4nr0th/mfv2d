"""Submodule to deal with different differentional forms."""

# Boundary conditions
from interplib.kforms.boundary import (
    BoundaryCondition1D as BoundaryCondition1D,
)
from interplib.kforms.boundary import (
    BoundaryCondition1DStrong as BoundaryCondition1DStrong,
)
from interplib.kforms.boundary import (
    BoundaryCondition1DWeak as BoundaryCondition1DWeak,
)
from interplib.kforms.kform import KEquaton as KEquaton

# Basic K-forms
from interplib.kforms.kform import KForm as KForm

# from interplib.kforms.kform import KFormDerivative as KFormDerivative
# from interplib.kforms.kform import KFormProjection as KFormProjection
from interplib.kforms.kform import KFormSystem as KFormSystem

# from interplib.kforms.kform import KInnerProduct as KInnerProduct
# from interplib.kforms.kform import KSum as KSum
# from interplib.kforms.kform import KWeight as KWeight
# from interplib.kforms.kform import KWeightDerivative as KWeightDerivative
from interplib.kforms.kform import element_system as element_system
