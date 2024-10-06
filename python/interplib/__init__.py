"""Package dedicated to interpolation using data defined on different topological objects."""

from interplib._interp import test as test

from interplib.interp1d import lagrange1d as lagrange1d
from interplib.interp1d import dlagrange1d as dlagrange1d
from interplib.interp1d import d2lagrange1d as d2lagrange1d
