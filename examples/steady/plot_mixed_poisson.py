r"""
Poisson Equation in the Mixed Formulation
==========================================

This example shows how the Poisson equation can be solved using the mixed
formulation. This means that the equation :eq:`steady-mixed-poisson-equation`
is formulated in two steps, given in equations :eq:`steady-mixed-poisson-first`
and :eq:`steady-mixed-poisson-second`.

.. math::
    :label: steady-mixed-poisson-equation

    \nabla^2 u = - f

.. math::
    :label: steady-mixed-poisson-first

    \vec{q} - \nabla u = 0

.. math::
    :label: steady-mixed-poisson-second

    \nabla \cdot \vec{q} = -f

The mixed formulation requires that the unknown :math:`u` be a differential
2-form, defined in :math:`L^2`, and that :math:`\vec{q}` be a differential
1-form. If the equation :eq:`steady-mixed-poisson-equation` were to be rewritten
with that in mind, it would lead to equation :eq:`steady-mixed-poisson-diff`.

.. math::
    :label: steady-mixed-poisson-diff

    \mathrm{d} \star \mathrm{d} \star  u^{(2)} = -f^{(2)}

This can be rewritten in the variational form as per equation
:eq:`steady-mixed-poisson-var`

.. math::
    :label: steady-mixed-poisson-var

    \int_\Omega \mathrm{d} p^{(1)} \wedge \star u^{(2)} +
    \int_\Omega p^{(1)} \wedge \star q^{(1)} =
    \int_\Omega p^{(1)} \wedge \star u^{(2)} \quad
    \forall p^{(1)} \in \Lambda^{(1)}\left( \mathcal{M} \right)

    \int_\Omega v^{(2)} \wedge \star \mathrm{d} q^{(1)} =
    \int_{\partial \Omega} v^{(2)} \wedge \star f^{(2)} \quad
    \forall v^{(2)} \in \Lambda^{(2)}\left( \mathcal{M} \right)

Error for this case will be measured in two ways. First is in the typical
:math:`L^2` norm defined by equation :eq:`steady-mixed-poisson-l2-norm`.

.. math::
    :label: steady-mixed-poisson-l2-norm

    \varepsilon_{L^2}(u) = \sqrt{\int_\Omega
    \left( u_\mathrm{exact} - u\right) {\mathrm{d}\Omega}}

The second is the semi-norm defined by equation :eq:`steady-mixed-poisson-h1-norm`.
The reason to also use this norm, is that is is the norm which is induced by the
Laplace operator, which the Poisson equation is defined with.

.. math::
    :label: steady-mixed-poisson-h1-norm

    \varepsilon_{H^1}(u) = \int_\Omega \left|\left|
    \nabla  u_\mathrm{exact} - \nabla u \right|\right| {\mathrm{d}\Omega} =
    \int_\Omega \left|\left| \nabla  u_\mathrm{exact} - q \right|\right|
    {\mathrm{d}\Omega}


"""  # noqa

import numpy as np
import numpy.typing as npt
import pyvista as pv
import rmsh
from matplotlib import pyplot as plt
from mfv2d import (
    KFormSystem,
    KFormUnknown,
    SolverSettings,
    SystemSettings,
    UnknownFormOrder,
    mesh_create,
    solve_system_2d,
)

# %%
#
# Setup
# -----
#
# The first thing is to setup the necessary prerequisites. This first of all means
# defining the manufactured solution used for the verification. The manufactured
# solution for :math:`u^{(2)}` is given by equation
# :eq:`steady-mixed-poisson-manufactured-u`, with its gradient :math:`q^{(1)}`
# given by equation :eq:`steady-mixed-poisson-manufactured-q`.
#
# .. math::
#     :label: steady-mixed-poisson-manufactured-u
#
#     u^{(2)}(x, y) = 2 \cos\left(\frac{\pi x}{2}\right) \cos\left(\frac{\pi y}{2}\right)
#
# .. math::
#     :label: steady-mixed-poisson-manufactured-q
#
#     q^{(1)}(x, y) = -\pi \sin\left(\frac{\pi x}{2}\right)
#     \cos\left(\frac{\pi y}{2}\right) dx - \pi \cos\left(\frac{\pi x}{2}\right)
#     \sin\left(\frac{\pi y}{2}\right) dy
#
# The source term on the right side of the equation is thus given by equation
# :eq:`steady-mixed-poisson-manufactured-f`.
#
# .. math::
#     :label: steady-mixed-poisson-manufactured-f
#
#     f^{(2)}(x, y) = - \pi^2 \cos\left(\frac{\pi x}{2}\right)
#     \cos\left(\frac{\pi y}{2}\right)
#


def u_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact solution."""
    return 2 * np.cos(np.pi / 2 * x) * np.cos(np.pi / 2 * y) + 5


def q_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact gradient of solution."""
    return np.stack(
        (
            -np.pi * np.sin(np.pi / 2 * x) * np.cos(np.pi / 2 * y),
            -np.pi * np.cos(np.pi / 2 * x) * np.sin(np.pi / 2 * y),
        ),
        axis=-1,
    )


def source_exact(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Exact heat flux divergence."""
    return -(np.pi**2) * np.cos(np.pi / 2 * x) * np.cos(np.pi / 2 * y)


# %%
#
# System Setup
# ------------
#
# Here the system is set up. The equations in the system bellow together represnt
# system :eq:`steady-mixed-poisson-var`. Note that the weak boundary conditions are
# introduced throught the boundary integral ``p ^ u_exact``.

u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
v = u.weight
q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)
p = q.weight

system = KFormSystem(
    p.derivative * u - p * q == p ^ u_exact,
    v * q.derivative == -(v * source_exact),
    sorting=lambda f: f.order,
)
print(system)

# %%
#
# Making The Mesh
# ---------------
#
# The mesh this is being solved on is a a single block of 6 by 6 quatrilaterals.
# The boundaries of the mesh are defined by B-splines with 4 knots, meaning they
# are cubic splines. The mesh is presented in the plot bellow.
#

N = 6
n1 = N
n2 = N

m, rx, ry = rmsh.create_elliptical_mesh(
    rmsh.MeshBlock(
        None,
        rmsh.BoundaryCurve.from_knots(
            n1, (-1, -1), (-0.5, -1.1), (+0.5, -0.6), (+1, -1)
        ),  # bottom
        rmsh.BoundaryCurve.from_knots(
            n2, (+1, -1), (+1.5, -0.7), (+1, 0.0), (+1, +1)
        ),  # right
        rmsh.BoundaryCurve.from_knots(
            n1, (+1, +1), (0.5, 0.5), (-0.5, 0.5), (-1, +1)
        ),  # top
        rmsh.BoundaryCurve.from_knots(
            n2, (-1, +1), (-0.5, 0.33), (-1, -0.5), (-1, -1)
        ),  # left
    )
)
assert rx < 1e-6 and ry < 1e-6

# Show the mesh for the first time.
fig, ax = plt.subplots(1, 1)
xlim, ylim = m.plot(ax)
ax.set_xlim(1.1 * xlim[0], 1.1 * xlim[1])
ax.set_ylim(1.1 * ylim[0], 1.1 * ylim[1])
ax.set_aspect("equal")
plt.show()
# %%
#
# Check the Result
# ----------------
#
# Before checking the convergence, let us first just check on how the solution
# looks.
pval = 3  # Test polynomial order
msh = mesh_create(pval, np.stack((m.pos_x, m.pos_y), axis=-1), m.lines + 1, m.surfaces)

solution, stats, mesh = solve_system_2d(
    msh,
    system_settings=SystemSettings(system),
    solver_settings=SolverSettings(absolute_tolerance=1e-10, relative_tolerance=0),
    print_residual=False,
    recon_order=25,
)


sol: pv.UnstructuredGrid = solution[-1]
pv.set_plot_theme("document")
plotter = pv.Plotter(shape=(1, 3), window_size=(1600, 800), off_screen=True)

plotter.subplot(0, 0)
plotter.add_mesh(sol.copy(), scalars=u.label, show_scalar_bar=True)
plotter.add_text("Computed")
plotter.view_xy()

sol.point_data["u_exact"] = u_exact(sol.points[:, 0], sol.points[:, 1])
plotter.subplot(0, 1)
plotter.add_mesh(sol.copy(), scalars="u_exact", show_scalar_bar=True)
plotter.add_text("Exact")
plotter.view_xy()

sol.point_data["abs_error"] = np.abs(sol.point_data["u_exact"] - sol.point_data[u.label])
plotter.subplot(0, 2)
plotter.add_mesh(sol.copy(), scalars="abs_error", show_scalar_bar=True, log_scale=True)
plotter.add_text("Absolute Error")
plotter.view_xy()


# %%
#
# Solve for Different Orders
# --------------------------
#
# So we solve for different orders.

p_vals = np.arange(1, 7)
h1_err = np.zeros(p_vals.size)
l2_err = np.zeros(p_vals.size)

for ip, pval in enumerate(p_vals):
    msh = mesh_create(
        pval, np.stack((m.pos_x, m.pos_y), axis=-1), m.lines + 1, m.surfaces
    )

    solution, stats, mesh = solve_system_2d(
        msh,
        system_settings=SystemSettings(system),
        solver_settings=SolverSettings(absolute_tolerance=1e-10, relative_tolerance=0),
        print_residual=False,
        recon_order=25,
    )

    sol = solution[-1]
    sol.point_data["q_err2"] = np.linalg.norm(
        sol.point_data["q"] - q_exact(sol.points[:, 0], sol.points[:, 1]), axis=-1
    )
    sol.point_data["u_err2"] = (
        sol.point_data["u"] - u_exact(sol.points[:, 0], sol.points[:, 1])
    ) ** 2

    total_error = sol.integrate_data()
    h1_err[ip] = total_error.point_data["q_err2"][0]
    l2_err[ip] = np.sqrt(total_error.point_data["u_err2"][0])
    print(f"Finished {pval=:d}")

# %%
#
# Plot Results
# ------------
#
# Here we plot the results.
#
# :math:`H^1` Norm
# ~~~~~~~~~~~~~~~~
#

k1, k0 = np.polyfit((p_vals), np.log(h1_err), 1)
k1, k0 = np.exp(k1), np.exp(k0)

print(f"Solution converges with p as: {k0:.3g} * ({k1:.3g}) ** p in H1 norm.")
plt.figure()

plt.scatter(p_vals, h1_err)
plt.semilogy(
    p_vals,
    k0 * k1**p_vals,
    label=f"${k0:.3g} \\cdot \\left( {{{k1:+.3g}}}^p \\right)$",
    linestyle="dashed",
)
plt.gca().set(
    xlabel="$p$",
    ylabel="$\\left|\\left| \\nabla u - \\nabla \\bar{u} \\right|\\right|$",
    yscale="log",
)
plt.legend()
plt.grid()
plt.show()

# %%
#
# :math:`L^2` Norm
# ~~~~~~~~~~~~~~~~
#

k1, k0 = np.polyfit((p_vals), np.log(l2_err), 1)
k1, k0 = np.exp(k1), np.exp(k0)

print(f"Solution converges with p as: {k0:.3g} * ({k1:.3g}) ** p in L2 norm.")
plt.figure()

plt.scatter(p_vals, l2_err)
plt.semilogy(
    p_vals,
    k0 * k1**p_vals,
    label=f"${k0:.3g} \\cdot \\left( {{{k1:+.3g}}}^p \\right)$",
    linestyle="dashed",
)
plt.gca().set(
    xlabel="$p$",
    ylabel="$\\varepsilon_{L^2}$",
    yscale="log",
)
plt.legend()
plt.grid()
plt.show()
