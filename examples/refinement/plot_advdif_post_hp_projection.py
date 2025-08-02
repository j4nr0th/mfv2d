"""
Approximating Error from Projections with Advection-Diffusion
=============================================================

Using exact solution as a refinement criterion is a luxury that's not
often availabel for real problems. As such, process in the example
:ref:`sphx_glr_auto_examples_refinement_plot_direct_poison_post_hp.py`
may not be possible to replicate.

As such, this example shows how error can be estimated using projection
to a finer mesh. This means that a problem is solved twice, with same
topology, but different element orders. The finer one is then taken
as being much closer to the real solution and thus used to compute
an error estimate.
"""  # noqa

from functools import partial

import numpy as np
import numpy.typing as npt
import pyvista as pv
import rmsh
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mfv2d import (
    ErrorEstimateCustom,
    KFormSystem,
    KFormUnknown,
    Mesh,
    RefinementLimitElementCount,
    RefinementSettings,
    SystemSettings,
    UnknownFormOrder,
    compute_legendre_coefficients,
    mesh_create,
    solve_system_2d,
)

# %%
#
# Problem Description
# -------------------
#
# The model problem that is to be solved here is the linear advection
# diffusion equation on a deformed square domain, with the solution
# given by equation :eq:`refine-advdif-post-hp-2`,
# where the function :math:`s` is specified by equation :eq:`refine-advdif-post-hp-1`.
# The advection vector field is given by
#
# .. math::
#     :label: refine-advdif-post-hp-1
#
#     s(r, t_0; t) = e^{-r (t - t_0)^2}
#
# .. math::
#     :label: refine-advdif-post-hp-2
#
#     u(x, y) = s(40, 0.75; x) \cdot s(40, 0.75; y)
#
# .. math::
#     :label: refine-advdif-post-hp-3
#
#     \vec{a} = \begin{bmatrix} 3x + y \\ x^2 - y^3 \end{bmatrix}
#
# The cross section of the solution can be seen in the plot bellow.

R = 40.0


def s(t: npt.NDArray[np.float64], r: float, t0: float) -> npt.NDArray[np.floating]:
    """Compute source term."""
    return np.exp(-r * (t - t0) ** 2)


def dsdt(t: npt.NDArray[np.float64], r: float, t0: float) -> npt.NDArray[np.floating]:
    """Compute derivative source term."""
    return -2 * r * (t - t0) * np.exp(-r * (t - t0) ** 2)


def d2sdt2(t: npt.NDArray[np.float64], r: float, t0: float) -> npt.NDArray[np.floating]:
    """Compute second derivative source term."""
    return 2 * r * (2 * r * (t - t0) ** 2 - 1) * np.exp(-r * (t - t0) ** 2)


def u_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact solution."""
    return s(x, R, 0.75) * s(y, R, 0.75)


def q_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact gradient of solution."""
    return np.stack(
        (
            dsdt(x, R, 0.75) * s(y, R, 0.75),
            s(x, R, 0.75) * dsdt(y, R, 0.75),
        ),
        axis=-1,
    )


def adv_field(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Advection field."""
    return np.stack(
        (3 * x + y, x**2 - y**3),
        axis=-1,
    )


def source_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact heat flux divergence."""
    return (
        s(x, R, 0.75) * d2sdt2(y, R, 0.75) + d2sdt2(x, R, 0.75) * s(y, R, 0.75)
    ) + np.sum(adv_field(x, y) * q_exact(x, y), axis=-1)


fig, ax = plt.subplots()

xplt = np.linspace(-1, +1, 501, dtype=np.float64)
ax.plot(xplt, s(xplt, R, 0.75))
ax.grid()
ax.set(xlabel="$x$", ylabel="$y$")
fig.tight_layout()
plt.show()

# %%
#
# System Setup
# ------------
#
# System setup for this is exactly the same as it was for other examples,
# with the system and boundary conditions specified. Since the problem is
# written in the mixed formulation, Dirichlet boundary conditions are weakly
# imposed.

u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
v = u.weight

q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)
p = q.weight

system = KFormSystem(
    p * q + p.derivative * u == p ^ u_exact,
    v * q.derivative - (v * (adv_field * ~q)) == v * source_exact,
    sorting=lambda f: f.order,
)
print(system)

# %%
#
# Making the Mesh
# ---------------
#
# The mesh is a deformed square, as mentioned before. Its corners
# match the unit square, but the sides are curved. There are
# five elements in every direction.
#
# Initial mesh can be seen bellow.

N = 6
n1 = N
n2 = N


m, rx, ry = rmsh.create_elliptical_mesh(
    rmsh.MeshBlock(
        None,
        rmsh.BoundaryCurve.from_knots(
            n1, (-1, -1), (-0.75, -1.3), (+0.5, -0.9), (+1, -1)
        ),  # bottom
        rmsh.BoundaryCurve.from_knots(
            n2, (+1, -1), (+1.5, -0.7), (+1, 0.0), (+1, +1)
        ),  # right
        rmsh.BoundaryCurve.from_knots(
            n1, (+1, +1), (0.5, 1.5), (-0.5, 1.25), (-1, +1)
        ),  # top
        rmsh.BoundaryCurve.from_knots(
            n2, (-1, +1), (-0.5, 0.33), (-1, -0.5), (-1, -1)
        ),  # left
    )
)
assert rx < 1e-6 and ry < 1e-6

fig, ax = plt.subplots()
m.plot(ax)
ax.autoscale()
ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
fig.tight_layout()
plt.show()


# %%
#
# Initialize the Mesh
# ~~~~~~~~~~~~~~~~~~~
#
# With geometry created, it is converted into suitable form to use with :mod:`mfv2d`.
# Also a convinience function ``plot_mesh_comparisons`` is defined here, to allow for
# simple comparison of meshes.


PSTART = 1  # Test polynomial order
mesh_initial = mesh_create(
    PSTART, np.stack((m.pos_x, m.pos_y), axis=-1), m.lines + 1, m.surfaces
)


def plot_mesh_comparisons(*meshes: tuple[str, Mesh]) -> None:
    """Plot one or more meshes with given titles."""
    fig, axes = plt.subplots(len(meshes), 1, figsize=(5, 7 * len(meshes)))

    for ax, (title, mesh) in zip(axes, meshes, strict=True):
        vertices = [mesh.get_leaf_corners(idx) for idx in mesh.get_leaf_indices()]
        ax.add_collection(PolyCollection(vertices, facecolors="none", antialiased=True))
        for idx, quad in zip(mesh.get_leaf_indices(), vertices):
            ax.text(
                *np.mean(quad, axis=0),
                f"{np.linalg.norm(mesh.get_leaf_orders(idx)) / np.sqrt(2):.3g}",
                ha="center",
                va="center",
                color="red",
                fontsize=6,
            )
        ax.autoscale()
        ax.set(aspect="equal", title=title)

    fig.tight_layout()

    plt.show()


# %%
#
# Refinement Settings
# -------------------
#
# For error refinement, the main difference occurs in how the error calculation
# function is written. Instead of using the exact solution, the ``fine_solution``
# keyword argument is taken, interpolated on the integration points' positions
# ``x`` and ``y``, which is then used to estimate error.


def error_calc_function(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    order_1: int,
    order_2: int,
    xi: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    **kwargs,
) -> tuple[float, float]:
    """Compute L2 error "estimate" and H1 refinement cost."""
    u = kwargs["u"]
    mesh: pv.UnstructuredGrid = kwargs["fine_solution"]
    assert x.shape == y.shape
    points = pv.PolyData(np.stack((x.flatten(), y.flatten(), np.zeros(x.size)), axis=-1))
    pts = points.sample(mesh)
    real_u = np.reshape(pts.point_data["u"], x.shape)
    err = real_u - u
    coeffs_err = compute_legendre_coefficients(order_1, order_2, xi, eta, err * w)
    coeffs_u = compute_legendre_coefficients(order_1, order_2, xi, eta, u * w)
    norm = 4 / (
        (2 * np.arange(order_1 + 1) + 1)[None, :]
        * (2 * np.arange(order_2 + 1) + 1)[:, None]
    )
    measure = coeffs_u * (coeffs_u + 2 * coeffs_err) / norm
    estimate = (
        np.sum(measure[order_1 // 2 :, order_2 // 2 :])
        + np.sum(measure[order_1 // 2 :, : order_2 // 2])
        + np.sum(measure[: order_1 // 2, order_2 // 2 :])
    )
    return np.sum(err**2 * w), np.abs(estimate)


N_ROUNDS = 12
system_settings = SystemSettings(system=system)


# %%
#
# Evaluating a Refinement Strategy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Evaluation of refinement strategies is similar to how it was in
# :ref:`sphx_glr_auto_examples_refinement_plot_direct_poison_post_hp.py`,
# with the difference being that the problem is first solved on a mesh
# where all elements are have all elements with orders increased by ``dp``,
# which is then passed to the error estimation function as ``fine_solution``.

RUN_COUNT = 0


def run_refinement_strategy(dp: int, h_ratio: float, max_elements: int, mesh: Mesh):
    """Compute errors and resulting mesh for a refinement strategy."""
    global RUN_COUNT
    errors_local: list[tuple[int, float]] = list()
    new_mesh = mesh.copy()
    # new_mesh.uniform_p_change(+dp, +dp)
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900), shape=(1, 2))
    plotter.open_gif(f"direct-poisson-post-hp-{RUN_COUNT:d}.gif", fps=1)
    RUN_COUNT += 1

    for i_round in range(N_ROUNDS):
        mesh = new_mesh
        # Obtain the finer solution
        mesh.uniform_p_change(+dp, +dp)
        fine_solutions, statistics, _ = solve_system_2d(
            mesh,
            system_settings=system_settings,
            refinement_settings=None,
            recon_order=15,
        )
        # Change mesh order back
        mesh.uniform_p_change(-dp, -dp)
        refinement_settings = RefinementSettings(
            # Specifying how the error is estimated
            error_estimate=ErrorEstimateCustom(
                # Required by the error function
                required_forms=[u],
                # The error (estimation) function
                error_calculation_function=partial(
                    error_calc_function, fine_solution=fine_solutions[-1]
                ),
            ),
            # H-refinement when ratio of h-cost and error less than this
            h_refinement_ratio=h_ratio,
            # When to stop refining
            refinement_limit=RefinementLimitElementCount(1.0, max_elements),
            # Print error distribution to terminal
            report_error_distribution=True,
            # Print element order distribution to terminal
            report_order_distribution=True,
        )

        solutions, statistics, new_mesh = solve_system_2d(
            mesh,
            system_settings=system_settings,
            refinement_settings=refinement_settings,
            recon_order=15,
        )

        solution = solutions[-1]
        # solution = fine_solutions[-1]
        u_computed = solution.point_data[u.label]
        u_real = u_exact(solution.points[:, 0], solution.points[:, 1])
        l2_err2 = (u_real - u_computed) ** 2
        solution.point_data["l2_error2"] = l2_err2

        errors_local.append(
            (
                statistics.n_total_dofs,
                np.sqrt(solution.integrate_data().point_data["l2_error2"][0]),
            )
        )

        # Plotting code here
        plotter.subplot(0, 0)
        plotter.add_mesh(
            solution,
            scalars="l2_error2",
            log_scale=True,
            label="$L^2$ error",
            clim=(1e-15, 1e-1),
            name="solution",
        )
        plotter.add_mesh(
            solution.extract_all_edges(), scalars=None, color="black", name="boundaries"
        )
        plotter.view_xy()

        plotter.subplot(0, 1)
        sol = solution.copy()
        sol.cell_data["geometrical order"] = np.linalg.norm(
            [mesh.get_leaf_orders(ie) for ie in mesh.get_leaf_indices()], axis=-1
        ) / np.sqrt(2)
        plotter.add_mesh(sol, scalars="geometrical order", name="orders", clim=(1, 12))
        plotter.add_text(f"Round {i_round + 1:d}", name="title")
        plotter.view_xy()

        plotter.write_frame()

    plotter.close()
    return errors_local, mesh


# %%
#
# Baseline Data
# -------------
#
# Again, uniform p-refinement is used as baseline to show benefits of local hp-refinement.

errors_uniform: list[tuple[int, float]] = list()

uniform_mesh = mesh_initial.copy()
for pval in range(PSTART, 9):
    solutions, statistics, _ = solve_system_2d(
        uniform_mesh,
        system_settings=system_settings,
        refinement_settings=None,
        recon_order=15,
    )
    solution = solutions[-1]
    u_computed = solution.point_data[u.label]
    u_real = u_exact(solution.points[:, 0], solution.points[:, 1])
    l2_err2 = (u_real - u_computed) ** 2
    solution.point_data["l2_error2"] = l2_err2
    errors_uniform.append(
        (
            statistics.n_total_dofs,
            np.sqrt(solution.integrate_data().point_data["l2_error2"][0]),
        )
    )
    uniform_mesh.uniform_p_change(1, 1)  # Up the mesh orders

# %%
#
# Running hp-Refinement
# ---------------------
#
# For all hp-runs, the values of 0.1 and 10 were taken as values of
# h-refinement ratio and element limit count. The things that were
# changed was the polynomial order difference ``dp``.
#
# :math:`dp = 1`
# ~~~~~~~~~~~~~~~
#


BASE_H_RATIO = 0.05
BASE_ELEMENT_LIMIT = 10
errors_local_1, local_mesh_1 = run_refinement_strategy(
    1, BASE_H_RATIO, BASE_ELEMENT_LIMIT, mesh_initial
)


def err_name(dp: int) -> str:
    """Format name for error plots."""
    return f"$\\Delta p ={dp:d}$"


errors_local_name_1 = err_name(1)


# %%
#
# :math:`dp = 2`
# ~~~~~~~~~~~~~~~
#

errors_local_2, local_mesh_2 = run_refinement_strategy(
    2, BASE_H_RATIO, BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_2 = err_name(2)

# %%
#
# :math:`dp = 3`
# ~~~~~~~~~~~~~~~
#
errors_local_3, local_mesh_3 = run_refinement_strategy(
    3, BASE_H_RATIO, BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_3 = err_name(3)


# %%
# Plotting Results
# ----------------
#


def plot_error_rates(*args: tuple[str, npt.ArrayLike]) -> None:
    """Plot a comparison of convergence rates with their names."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, data in args:
        el = np.array(data)
        err = el[:, 1]
        ns = el[:, 0] / 100
        kl1, kl0 = np.polyfit(ns, np.log(err), 1)
        kl1, kl0 = np.exp(kl1), np.exp(kl0)

        ax.scatter(el[:, 0], err, marker="x")
        ax.plot(
            el[:, 0],
            kl0 * kl1 ** (ns),
            label=label
            + f": ${kl0:.3g} \\cdot {{{kl1:.3g}}}^{{\\frac{{N_\\mathrm{{dofs}}}}"
            f"{{100}}}}$",
            linestyle="dashed",
        )
    ax.grid()
    ax.legend()
    ax.set(
        xlabel="$N_\\mathrm{dofs}$",
        ylabel="$\\left|\\left| u - \\bar{u} \\right|\\right|_{L^2}$",
        yscale="log",
    )
    fig.tight_layout()

    plt.show()


plot_error_rates(
    ("uniform", errors_uniform),
    (errors_local_name_1, errors_local_1),
    (errors_local_name_2, errors_local_2),
    (errors_local_name_3, errors_local_3),
)

# %%
# Compare the Meshes
# ~~~~~~~~~~~~~~~~~~
#
#

plot_mesh_comparisons(
    ("Baseline", mesh_initial),
    (errors_local_name_1, local_mesh_1),
    (errors_local_name_2, local_mesh_2),
    (errors_local_name_3, local_mesh_3),
)
