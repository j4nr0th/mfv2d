"""
Example 2: Mixed Formulation of Poisson Equation in 1D
======================================================

.. currentmodule:: interplib

This example shows how the Python module can be used to solve the mixed formulation of the
Poisson equation in 1D. This formulation is typically regarded as more difficult way to solve
the Poisson equation, but allows for higher order accuracy in the gradient of the solution, which
may be more desirable in some cases.
"""  # noqa

# %%
#
# Mimetic Derivation
# ------------------
#
# The first thing which ought to be done is to define the problem the same way it was
# defined in :ref:`example 1<sphx_glr_auto_examples_plot_poisson_direct.py>`.
# The goal is once gain to solve the Poisson equation in 1D:
#
# .. math::
#
#   \begin{align}
#   \nabla^2 \phi = - f(x) && x \in [x_0, x_1]
#   \end{align}
#
# With boundary conditions given for this case as Dirichlet on one side and Neumann
# on the other:
#
# .. math::
#
#   \begin{align}
#   \phi(x_0) = \phi_0 && \frac{d \phi}{d x}(x_1) = u_1
#   \end{align}
#
# The initial part of the mimetic derivation is the same as was in
# :ref:`example 1<sphx_glr_auto_examples_plot_poisson_direct.py>`:
#
# .. math::
#   :label: first
#
#   d \phi - u = 0
#
# .. math::
#   :label: second
#
#   d u = - f
#
# However, the difference now is that the goal is to keep :math:`u`. As such we know for
# sure that :math:`u` must be a 0-form, as otherwise we can not take its exterior
# derivative. This then makes :math:`f` a 1-form. The issue is then in how to reconcile
# the equation :eq:`first`. Here, we solve this by multiplying the equation with a weight
# function :math:`w`, which then allows us to move the exterior derivative on it,
# instead of :math:`\phi`. This also means, that the weight function should be a 0-form:
#
# .. math::
#
#   \int\limits_{x_0}^{x_1} w \left( u - d \phi \right) {dx} =
#   \int\limits_{x_0}^{x_1} w u + \int\limits_{x_0}^{x_1} d w \phi {dx}
#   - \left[ w \phi \right]_{x_0}^{x_1} = 0
#
# This now determines the following:
#
# - :math:`u` is a 0-form,
# - :math:`w` is a 0-form,
# - :math:`\phi` is a 1-form.
#
# To solve equation :math:`second`, we must then use a 1-form weight function :math:`q`,
# which thus makes :math:`f` a 1-form. With that and some abuse of notation, the entire
# system of equations becomes:
#
# .. math::
#
#   \begin{align}
#   \int_\Omega w u + \int_\Omega dw \phi &= \int_{\partial\Omega} w \phi \\
#   \int_\Omega q {du}  &= - \int_\Omega q f \\
#   \end{align}
#
# Manufactured Solution
# ---------------------
#
# To validate the solver works, consider a case with a manufactured solution given by:
#
# .. math::
#
#   \phi(x) = - \sin\left( \alpha \pi x + \beta \right), \quad \alpha, \beta \in
#   \mathbb{R}
#
# This gives the following forcing function:
#
# .. math::
#
#   f(x) = - \left(\alpha \pi x \right)^2 \sin\left( \alpha \pi x + \beta \right)
#
# And the gradient as:
#
# .. math::
#
#   u(x) = - (\alpha \pi) \cos\left( \alpha \pi x + \beta \right)
#
#

import numpy as np
import numpy.typing as npt
from interplib import kforms, mimetic, solve_system_on_mesh

ALPHA = 1.2
BETA = 0.2


def f_exact(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute f(x)."""
    return np.astype(
        -((ALPHA * np.pi) ** 2) * np.sin(ALPHA * np.pi * np.asarray(x) + BETA), np.float64
    )


def u_exact(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute u(x)."""
    return np.astype(
        -(ALPHA * np.pi) * np.cos(ALPHA * np.pi * np.asarray(x) + BETA), np.float64
    )


def phi_exact(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute phi(x)."""
    return np.astype(-np.sin(ALPHA * np.pi * np.asarray(x) + BETA), np.float64)


# %%
#
# Mesh
# ----
#
# The mesh will once again be a simple line, since this is 1D. For this case, elements
# of the exact same order :math:`p` will be used.
# #

p = 3
n_elem = 5
mesh = mimetic.Mesh1D(
    positions=(1 - np.cos(np.linspace(0, np.pi, n_elem + 1))) / 2.0,
    element_order=p,
)

# %%
#
# Equation
# --------
#
# The equations are defined in the same way
# :ref:`example 1<sphx_glr_auto_examples_plot_poisson_direct.py>`.
# This time, the difference is that there are two equations which are to be solved
# per element.

# %%
#
# The first equation is the one which enforces :math:`u` being equal to the exterior
# derivative of :math:`\phi`. Note that the weak boundary conditions are to be specified
# later, once the solution is being computed.
#

## Weight form

## Unknown forms
phi = kforms.KForm(mesh.manifold, "phi", 1)
u = kforms.KForm(mesh.manifold, "u", 0)
w = u.weight

# Brackets are for readability
eq1 = (w * u) + (w.derivative * phi) == w * 0

# %%
#
# Now the second equation can be defined, which forces the exterior derivative of
# :math:`u` to match the prescribed forcing :math:`f`:
#

## New weight form
q = phi.weight

# Brackets are for readability
eq2 = (q * u.derivative) == (q * (lambda x: -f_exact(x)))

# %%
#
# With these defined, the system of equations can now be formed. This time, the
# the forms are sorted based on their order. This is mainly just to get a consistent
# ordering.

system = kforms.KFormSystem(
    eq1,
    eq2,
    sorting=lambda form: -form.order,
)
print(system)

# %%
#
# Solving
# -------
#
# The problem can now be solved. This time, the continuity is enforced only on :math:`u`,
# since that's a 0-form, while :math:`phi` is a 1-form. This time, a weak boundary
# condition on :math:`\phi` is given on the left boundary.
with np.printoptions(2):
    resulting_splines = solve_system_on_mesh(
        system,
        mesh,
        continuous=[u],
        bcs_left=kforms.BoundaryCondition1DWeak(phi, float(phi_exact(mesh.positions[0]))),
        bcs_right=kforms.BoundaryCondition1DStrong(
            {u: 1}, float(u_exact(mesh.positions[-1]))
        ),
    )

# %%
#
# Visualizing the Results
# -----------------------
#
# Both results for :math:`\phi` and :math:`u` are plotted.
#
#
# #

from matplotlib import pyplot as plt  # noqa: E402

nplt = 100
xplt = np.linspace(0, 1, nplt)

plt.figure()
plt.title("Solution for $\\phi$")
plt.scatter((0,), (phi_exact(0.0),), label="BC for $\\phi(0)$")
plt.plot(xplt, resulting_splines[phi](xplt), label="$\\hat{\\phi}$", color="orange")
plt.plot(xplt, phi_exact(xplt), label="$\\phi$", linestyle="dashed")
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Error in $\\phi$")
plt.plot(xplt, phi_exact(xplt) - resulting_splines[phi](xplt))
plt.xlabel("$x$")
plt.ylabel("$\\varepsilon$")
plt.grid()
plt.show()

plt.figure()
plt.title("Solution for $u$")
plt.scatter((1,), (u_exact(1),), label="BC for $u(1)$")
plt.plot(xplt, resulting_splines[u](xplt), label="$\\hat{u}$", color="orange")
plt.plot(xplt, u_exact(xplt), label="$u$", linestyle="dashed")
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Error in $u$")
plt.plot(xplt, u_exact(xplt) - resulting_splines[u](xplt))
plt.xlabel("$x$")
plt.ylabel("$\\varepsilon$")
plt.grid()
plt.show()
