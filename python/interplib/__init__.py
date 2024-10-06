"""Package dedicated to interpolation using data defined on different topological objects."""

from interplib._interp import test as test

from interplib.interp1d import interp1d_function_samples as interp1d_function_samples
from interplib.interp1d import interp1d_derivative_samples as interp1d_derivative_samples
from interplib.interp1d import interp1d_2derivative_samples as interp1d_2derivative_samples
from interplib.interp1d import (
    lagrange_function_samples as lagrange_function_samples
)
from interplib.interp1d import (
    lagrange_derivative_samples as lagrange_derivative_samples
)
from interplib.interp1d import (
    lagrange_2derivative_samples as lagrange_2derivative_samples
)