"""Submodule dealing with mimetics."""

# C types
from interplib._mimetic import GeoID as GeoID
from interplib._mimetic import Line as Line
from interplib._mimetic import Manifold as Manifold
from interplib._mimetic import Manifold1D as Manifold1D
from interplib._mimetic import Manifold2D as Manifold2D
from interplib._mimetic import Surface as Surface

# 1D types
from interplib.mimetic.mimetic1d import Element1D as Element1D
from interplib.mimetic.mimetic1d import Mesh1D as Mesh1D

# 2D types
from interplib.mimetic.mimetic2d import Element2D as Element2D
from interplib.mimetic.mimetic2d import ElementLeaf2D as ElementLeaf2D
from interplib.mimetic.mimetic2d import Mesh2D as Mesh2D
