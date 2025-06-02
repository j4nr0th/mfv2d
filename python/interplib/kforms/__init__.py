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
    BoundaryCondition2D as BoundaryCondition2D,
)
from interplib.kforms.boundary import (
    BoundaryCondition2DSteady as BoundaryCondition2DSteady,
)
from interplib.kforms.boundary import (
    BoundaryCondition2DUnsteady as BoundaryCondition2DUnsteady,
)

# Basic K-forms
from interplib.kforms.kform import KEquation as KEquation
from interplib.kforms.kform import KForm as KForm
from interplib.kforms.kform import KFormSystem as KFormSystem
from interplib.kforms.kform import KFormUnknown as KFormUnknown
