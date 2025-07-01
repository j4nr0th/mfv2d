r"""
Unsteady Heat Equations in Direct Formulation
=============================================

The (unsteady) heat equation is essentially just the Poisson equation with a time
derivative term added to the left side, as per equation
:eq:`unsteady-heat-direct-equation`.

.. math::
    :label: unsteady-heat-direct-equation

    \frac{\partial u}{\partial t} + \alpha \nabla^2 u = f


Just as was the case for the Poisson equation, this can be written in either direct
or mixed formulation. For this example the direct formulation is used. As such,
the variational form of :eq:`unsteady-heat-direct-equation` used is given by
equation :eq:`unsteady-heat-direct-variational`.

.. math::
    :label: unsteady-heat-direct-variational

    \left(v^{(0)}, \frac{\partial u}{\partial t}\right)_\Omega +
    \alpha \left(\mathrm{d} v^{(0)}, \mathrm{d} u^{(0)}\right)_\Omega =
    \int_{\partial\Omega} v^{(0)} \wedge \star \mathrm{d} u^{(0)}
    - \left(v^{(0)}, f\right)_\Omega
"""  # noqa: D205, D400

import numpy as np
import numpy.typing as npt
import rmsh
from matplotlib import pyplot as plt
from mfv2d import (
    BoundaryCondition2DSteady,
    KFormSystem,
    KFormUnknown,
    Mesh2D,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    UnknownFormOrder,
    solve_system_2d,
)
from scipy.integrate import trapezoid

# %%
#
# Setup
# -----
#
# For the verification here the manufactured solution is used. It should have the
# form given by :eq:`unsteady-heat-direct-manufactured`, with :math:`u_s` being a
# steady state solution. Due to the boundary conditions not being possible to
# change with time, the solution and mesh are taken such that the solution would
# remain zero on the boundary for the entire time.
#
# .. math::
#     :label: unsteady-heat-direct-manufactured
#
#     u_s(x, y) = \cos(\frac{\pi x}{2})\cos(\frac{\pi y}{2})
#
# As such, the mesh chosen is the :math:`[-1, +1] \times [-1, +1]` square and the
# steady solution is given by equation :eq:`unsteady-heat-direct-steady`.
# As for the values of conduction coefficient and decay coefficient, values of
# :math:`\alpha = 0.02` and :math:`\beta = 1` were taken.
#
# .. math::
#     :label: unsteady-heat-direct-steady
#
#     u_s(x, y) = \cos(\frac{\pi x}{2})\cos(\frac{\pi y}{2})
#
# Forcing needed to have the solution above forcing :math:`f` also had to be
# computed. To obtain the solution given by equation
# :eq:`unsteady-heat-direct-manufactured`
# the forcing had to be given by equation :eq:`unsteady-heat-direct-source`.
# The terms in :math:`u` could also be embedded in the system by moving them on the right
# side of the equation, which would also make the method a direct solve, instead of
# fixed-point iteration.
#
# .. math::
#     :label: unsteady-heat-direct-source
#
#     f = \beta (u_s - u) + \frac{\alpha \pi^2}{2} u


ALPHA = 0.02
BETA = 1


def steady_u(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Steady state solution."""
    return np.cos(np.pi * x / 2) * np.cos(np.pi * y / 2)


# %%
#
# System Setup
# ------------
#
# System setup is what was discussed above. What should be noted is the
# fact that since now there are involving the solution itself on the
# right side of the equation, this is now an iterative solve.


u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0)
v = u.weight

system = KFormSystem(
    ALPHA * (v.derivative * u.derivative)
    == BETA * (v * steady_u) - (BETA - ALPHA * np.pi**2 / 2) * (v * u),
    sorting=lambda f: f.order,
)
print(system)
# %%
#
# Making the Mesh
# ---------------
#
# The mesh is made on the :math:`[-1, +1] \times [-1, +1]` square, as mentioned
# before. As for the time steps, values of 2, 4, 8, 16, 32, 64, and 128 are used.
#
#

N = 11
P = 3
T_END = 2

n1 = N
n2 = N
rect_mesh, rx, ry = rmsh.create_elliptical_mesh(
    rmsh.MeshBlock(
        label=None,
        bottom=rmsh.BoundaryCurve.from_line(n1, (-1, -1), (+1, -1)),
        right=rmsh.BoundaryCurve.from_line(n2, (+1, -1), (+1, +1)),
        top=rmsh.BoundaryCurve.from_line(n2, (+1, +1), (-1, +1)),
        left=rmsh.BoundaryCurve.from_line(n2, (-1, +1), (-1, -1)),
    )
)
assert rx < 1e-6 and ry < 1e-6
mesh = Mesh2D(
    P,
    np.stack((rect_mesh.pos_x, rect_mesh.pos_y), axis=-1),
    rect_mesh.lines + 1,
    rect_mesh.surfaces,
)

nt_vals = np.logspace(start=1, stop=6, num=7, base=2, dtype=np.uint32)
er_vals = np.zeros(nt_vals.size)
dt_vals = np.zeros(nt_vals.size)

# %%
#
# Running the Calculations
# ------------------------
#
# Now we run the calculations and get the error.

for i_nt, nt in enumerate(nt_vals):
    dt = float(T_END / nt)
    solutions, stats = solve_system_2d(
        mesh,
        system_settings=SystemSettings(
            system,
            boundary_conditions=[
                BoundaryCondition2DSteady(u, mesh.boundary_indices, steady_u)
            ],
        ),
        solver_settings=SolverSettings(
            maximum_iterations=20, relative_tolerance=0, absolute_tolerance=1e-10
        ),
        time_settings=TimeSettings(dt=dt, nt=nt, time_march_relations={v: u}),
        recon_order=25,
    )

    n_sol = len(solutions)
    err_vals = np.zeros(n_sol)
    time_vals = np.zeros(n_sol)
    for isol, sol in enumerate(solutions):
        time = float(sol.field_data["time"][0])

        u_exact = steady_u(sol.points[:, 0], sol.points[:, 1]) * (
            1 - np.exp(-BETA * time)
        )
        u_err = sol.point_data["u"] - u_exact
        sol.point_data["u_err"] = np.abs(u_err)
        sol.point_data["u_exact"] = u_exact

        integrated = sol.integrate_data()
        err = float(integrated.point_data["u_err"][0])
        time_vals[isol] = time
        err_vals[isol] = err

    total_time_error = trapezoid(err_vals, time_vals)
    er_vals[i_nt] = total_time_error
    dt_vals[i_nt] = dt
    print(f"For {dt=} total error was {total_time_error:.3e}.")

# %%
#
# Plotting the Error
# ------------------
#
# Now we plot the error. As you can see, we magically got
# another order of accuracy out of fucking thin air. If I had
# to guess it is related to the fact that the time integration
# is symplectic.

k1, k0 = np.polyfit(np.log(dt_vals), np.log(er_vals), 1)
k0 = np.exp(k0)

fig, ax = plt.subplots(1, 1)
ax.scatter(dt_vals, er_vals)
ax.plot(
    dt_vals,
    k0 * dt_vals**k1,
    linestyle="dashed",
    label=f"${k0:.3g} \\cdot {{\\Delta t}}^{{{k1:+.3g}}}$",
)
ax.grid()
ax.legend()
ax.set(
    xlabel="$\\Delta t$",
    ylabel="$\\int \\left|u - \\bar{u}\\right| {dt}$",
    xscale="log",
    yscale="log",
)
ax.xaxis_inverted()
fig.tight_layout()
plt.show()
