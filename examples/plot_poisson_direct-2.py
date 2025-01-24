"""
Example 1: Direct Formulation of Poisson Equation in 1D
=======================================================

.. currentmodule:: interplib

This example shows how the Python module can be used to solve the direct formulation of the
Poisson equation in 1D. This is probably the simplest differential equation in 1D that can
be solved while meaningfully illustrating the way mimetic methods work.
"""  # noqa

# %%
#
# Mimetic Derivation
# ------------------
#
# The first thing which ought to be done is to define what problem we're solving exactly.
# The goal is to solve the Poisson equation in 1D, which is written as:
#
# .. math::
#
#   \begin{align}
#   \nabla^2 \phi = - f(x) && x \in [x_0, x_1]
#   \end{align}
#
# With boundary conditions given as:
#
# .. math::
#
#   \begin{align}
#   \phi(x_0) = \phi_0 && \phi(x_1) = \phi_1
#   \end{align}
#
# The way this is done with  the mimetic method is to first try and define the problem
# in terms of operations which can be performed on any manifold. In this case, the first
# step is to write the problem in two steps:
#
# .. math::
#
#   \begin{align}
#   d \phi - u &= 0\\
#          d u &= - f\\
#   \end{align}
#
# With this formulation we can infer some things about how to create the direct
# formulation:
#
# - If :math:`u` is a :math:`k`-form, then :math:`f` must be a :math:`(k+1)`-form.
# - If :math:`u` is a :math:`k`-form, then :math:`phi` must be a :math:`(k - 1)`-form.
#
# Both of these can not hold at the same time in 1D, so we must introduce another step.
#
#
# .. math::
#
#   \begin{align}
#   d \phi - \tilde{u} &= 0\\
#   u - \star \tilde{u} &= 0\\
#          d u &= - f
#   \end{align}
#
# Here we introduced another variable :math:`\tilde{u}`, which is used to solve
# the problem which arose previously. The :math:`\star` represents the Hodge operator,
# which for 1D means projection of a 0-form onto a 1-form or reverse. As such, we can
# now write the equation with:
#
# .. math::
#
#   \begin{align}
#   d \phi^{(0)} - \tilde{u}^{(1)} &= 0\\
#   u^{(0)} - \star \tilde{u}^{(1)} &= 0\\
#   d u^{(0)}  &= - f^{(1)}
#   \end{align}
#
# Which is now a system which is consistent. If we now represent each of the operations
# with their respective matrix operators, we obtain the following discrete system:
#
# .. math::
#
#   \begin{align}
#   \tilde{\mathbb{E}}^{0, 1} \vec{\phi}^{(0)} &= \vec{\tilde{u}}^{(1)}
#   \mathbb{H}^{0, 1} \vec{\tilde{u}}^{(1)} &= \vec{u}^{(0)}
#   \mathbb{E}^{0, 1} \vec{u}^{(0)}  &= - \vec{f}^{(1)}
#   \end{align}
#
# This can more concisely be simplified into a single equation with only one variable:
#
# .. math::
#
#   \mathbb{E}^{0, 1} \mathbb{H}^{0, 1} \tilde{\mathbb{E}}^{0, 1} \vec{\phi}^{(0)}
#   = \mathbb{E}^{0, 1} \mathbb{H}^{0, 1} \left(\mathbb{E}^{0, 1}\right)^T
#   = - \vec{f}^{(1)}
#
# While this is a very mimetic way to derive the equation, there is also a more
# "traditional" way of getting to it.
#
#
# Classical Derivation
# --------------------
#
# A more classic way of deriving it is to simply take the equation and multiply with an
# arbitrary weight function :math:`w: \mathbb{R} \leftarrow \mathbb{R}`:
#
# .. math::
#
#   w (\nabla^2 \phi + f) = 0
#
# If the equation is satisfied exactly, then it follows that the integral of this must
# also be zero thruought the entire domain for all choices :math:`w` that can be made:
#
# .. math::
#   :label: weak-1
#
#   \int\limits_{x_0}^{x_1} w (\nabla^2 \phi + f) {dx} = 0
#
# Since it is not possible to do this in practice for the infinite set of functions from
# which the weight :math:`w` could be chosen, we must limit ourselves to a limited set of
# functions. A very common and convenient choice for such cases is to use the fact that
# we must represent :math:`\phi` with a finite number of degrees of freedom, which we
# associate with different basis functions. As such, by using the basis functions for
# which the degrees of freedom are not known, the number of equations which can be
# generated from this equation always matches the number of the unknowns.
#
# We must also deal with the fact that we currently can not express the equation
# :eq:`weak-1` in terms of differential forms, due to the double exteriro derivative.
#
# As such, the first step is to use integration by parts and break the equation apart.
#
# .. math::
#   :label: weak-2
#
#   \int\limits_{x_0}^{x_1} d w d \phi {dx} = \int\limits_{x_0}^{x_1} w f {dx}
#   + \left[ \hat{n} w \phi \right]_{x_0}^{x_1}
#
# Now :math:`\phi`,:math:`w`, :math:`f` can be expressed as 0-forms:
#
#
# .. math::
#   :label: weak-3
#
#   \int\limits_{x_0}^{x_1} d w^{(0)} d \phi^{(0)} {dx} = \int\limits_{x_0}^{x_1} w^{(0)}
#   f^{(0)} {dx} + \left[ \hat{n} w^{(0)} \phi^{(0)} \right]_{x_0}^{x_1}
#
#
#
# Manufactured Solution
# ---------------------
#
# To validate the solver works, consider a case with a manufactured solution given by:
#
# .. math::
#   :label: man-phi
#
#   \phi(x) = - \sin\left( \alpha \pi x + \beta \right), \quad \alpha, \beta \in
#   \mathbb{R}
#
# This gives the following forcing function:
#
# .. math::
#   :label: man-f
#
#   f(x) = - \left(\alpha \pi x \right)^2 \sin\left( \alpha \pi x + \beta \right)
#
import numpy as np
import numpy.typing as npt
from interplib import kforms, mimetic, solve_system_on_mesh

ALPHA = 1.2
BETA = 0.2
A = 0.1
B = -0.02


def f_exact(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute f(x)."""
    t = ALPHA * np.pi * np.asarray(x) + BETA
    return -np.astype(
        A * (ALPHA * np.pi) * np.cos(t) - B * ((ALPHA * np.pi) ** 2) * np.sin(t),
        np.float64,
    )


def phi_exact(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute phi(x)."""
    return np.astype(np.sin(ALPHA * np.pi * np.asarray(x) + BETA), np.float64)


# %%
#
# Mesh
# ----
#
# Now we end up with the expression, which can be  solved using the Python module.
# The first step is to define the mesh (which for 1D is simply a line). For this case,
# Chebyshev nodes are used for the mesh. Each element also needs its own order, which
# determines the order of local 0-form basis functions and consequently number of basis
# and degrees of freedom for all other forms.
#
# These orders are given in the ``element_orders`` array.
#
# #

element_orders = [3, 3]
mesh = mimetic.Mesh1D(
    positions=(1 - np.cos(np.linspace(0, np.pi, len(element_orders) + 1))) / 2.0,
    element_order=element_orders,
)

# %%
#
# Equation
# --------
#
# Next, the equation must be defined. This equation will be solved for each element on
# the mesh, then the solution continuity and boundary conditions will be used to make
# the problem well posed.

# %%
#
# First the forms must be used. For weight functions we have to use
# the :class:`kforms.KFormDual` object, whereas the unknown must be a
# :class:`kforms.KFormPrimal` object.
#
## Unknown forms
phi = kforms.KForm(mesh.manifold, "phi", 0)
v = phi.weight

# %%
#
# With these defined, we can now formulate the equation that will be solved
equation = (
    -A * (v * ~(phi.derivative)) + B * (v.derivative * phi.derivative) == v * f_exact
)

# %%
#
# Given that this is the only equation for the entire system, we can now create
# the :class:`kforms.KFormSystem`.
#
system = kforms.KFormSystem(equation)
print(system)

# %%
#
# Solving
# -------
#
# To solve the 1D problem, we can now call the :func:`solve_system_on_mesh` function.
# This is also where we specify which forms must be continuous (in this case :math:`\phi`)
# and the boundary conditions.
#
# The resulting piecewise polynomial reconstruction of the solution is returned for each
# of the forms, though in this case, there's only :math:`\phi`.

resulting_splines = solve_system_on_mesh(
    system,
    mesh,
    continuous=[phi],
    bcs_left=kforms.BoundaryCondition1DStrong(
        {phi: 1}, float(phi_exact(mesh.positions[0]))
    ),
    bcs_right=kforms.BoundaryCondition1DStrong(
        {phi: 1}, float(phi_exact(mesh.positions[-1]))
    ),
)

# %%
#
# Visualizing the Results
# -----------------------
#
# Results can now be plotted. This case shows the plot of the
# plot of the solution and the error from the exact solution.
#
#
#
#
#
#
#
# #

from matplotlib import pyplot as plt  # noqa: E402

nplt = 100 * mesh.element_orders.size
xplt = np.linspace(0, 1, nplt)

plt.figure()
plt.title("Solution")
plt.scatter((0,), (phi_exact(0.0),), label="BC for $\\phi(0)$")
plt.scatter((1,), (phi_exact(1.0),), label="BC for $\\phi(1)$")
plt.plot(xplt, resulting_splines[phi](xplt), label="$\\hat{\\phi}$", color="orange")
plt.plot(xplt, phi_exact(xplt), label="$\\phi$", linestyle="dashed")
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Error")
plt.plot(xplt, phi_exact(xplt) - resulting_splines[phi](xplt))
plt.xlabel("$x$")
plt.ylabel("$\\varepsilon$")
plt.grid()
plt.show()
