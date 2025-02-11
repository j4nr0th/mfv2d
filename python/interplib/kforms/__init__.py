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
from interplib.kforms.boundary import (
    BoundaryCondition2DStrong as BoundaryCondition2DStrong,
)

# Basic K-forms
from interplib.kforms.kform import KEquaton as KEquaton
from interplib.kforms.kform import KFormSystem as KFormSystem
from interplib.kforms.kform import KFormUnknown as KFormUnknown
