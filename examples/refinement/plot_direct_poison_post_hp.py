"""
Post-Solver Combined hp-refinement with Direct Poisson
======================================================

This solver showcases refinement process using post-solver refinement.
The idea here is to use both h-refinement and p-refinement, which allows
for much more efficient convergence for locally sharp solutions.
"""  # noqa

import numpy as np
import numpy.typing as npt
import pyvista as pv
import rmsh
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mfv2d import (
    BoundaryCondition2DSteady,
    ErrorEstimateExplicit,
    KFormSystem,
    KFormUnknown,
    Mesh,
    RefinementLimitElementCount,  # Need a refinement limit
    RefinementSettings,  # Need refinement settings
    SystemSettings,
    UnknownFormOrder,
    mesh_create,
    solve_system_2d,
    system_as_string,
)

# %%
#
# Problem Description
# -------------------
#
# The model problem that is to be solved here is the direct Poisson equation
# on a deformed square domain, with the solution given by equation
# :eq:`refine-direct-poisson-post-hp-2`, where the function :math:`s` is
# specified by equation :eq:`refine-direct-poisson-post-hp-1`.
#
# .. math::
#     :label: refine-direct-poisson-post-hp-1
#
#     s(r, t_0; t) = e^{-r (t - t_0)^2}
#
# .. math::
#     :label: refine-direct-poisson-post-hp-2
#
#     u(x, y) = s(40, 0.75; x) \cdot s(40, 0.75; y)
#
#

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
    """Exact curl of solution."""
    return np.stack(
        (
            s(x, R, 0.75) * dsdt(y, R, 0.75),
            -dsdt(x, R, 0.75) * s(y, R, 0.75),
        ),
        axis=-1,
    )


def source_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact heat flux divergence."""
    return s(x, R, 0.75) * d2sdt2(y, R, 0.75) + d2sdt2(x, R, 0.75) * s(y, R, 0.75)


# %%
#
# System Setup
# ------------
#
# System setup for this is exactly the same as it was for other examples,
# with the system and weak boundary conditions specified.

u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_0)
v = u.weight

system = KFormSystem(
    v.derivative @ u.derivative == -(v @ source_exact) + (v ^ q_exact),
)
print(system_as_string(system))

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

fig, ax = plt.subplots()
m.plot(ax)
ax.autoscale()
ax.set(xlabel="$x$", ylabel="$y$")
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
    fig, axes = plt.subplots(1, len(meshes), figsize=(6 * len(meshes), 5))

    for ax, (title, mesh) in zip(axes, meshes, strict=True):
        vertices = [mesh.get_leaf_corners(idx) for idx in mesh.get_leaf_indices()]
        ax.add_collection(PolyCollection(vertices, facecolors="none", antialiased=True))
        for idx, quad in zip(mesh.get_leaf_indices(), vertices):
            ax.text(
                *np.mean(quad, axis=0),
                str(mesh.get_leaf_orders(idx)),
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
# Post-solver refinement is controlled with the :class:`RefinementSettings` type.
# The main setting is it is the ``error_calculation_function``, which should be
# set to a function which computes two values:
#
# - Error in some norm or semi-norm,
# - Cost of h-refinement in that norm,
#
# If the ratio of the cost of h-refinement and error is bellow the value specified
# as the ``h_refinement_ratio``, then the element will be h-refined. Otherwise it
# will be a candidate for p-refinement.
#
# For this example, the error measure is exact :math:`L^2` norm (since the
# manufactured solution is known), and measure of h-refinement cost is given as just
# an arbitrary constant, since ``h_refinement_ratio = 0``, meaning it will never happen.
#
# To specify when the refinement should stop, ``refinement_limit`` should be given.
# In this case, :class:`RefinementLimitElementCount` is used to specify that at
# either 10 elements or 100 % of the elements, whichever is lower, can be refined
# each iteration.


N_ROUNDS = 10
system_settings = SystemSettings(
    system=system,
    boundary_conditions=[
        BoundaryCondition2DSteady(u, mesh_initial.boundary_indices, u_exact)
    ],
)


# %%
#
# Evaluating a Refinement Strategy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To evaluate a refinement strategy, the problem will be solved ``N_ROUNDS``
# times, with post-solve refinement being run after each of these. At each
# square of the error in the :math:`L^2` norm is computed, to be able to
# compare with all other strategies.
#
# The difference between these strategies is how they decide between
# h-refinement and p-refinemenet, as well as how many elements they
# modify each round.

RUN_COUNT = 0


def run_refinement_strategy(h_ratio: float, max_elements: int, mesh: Mesh):
    """Compute errors and resulting mesh for a refinement strategy."""
    global RUN_COUNT
    errors_local: list[tuple[int, float]] = list()
    new_mesh = mesh
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900), shape=(1, 2))
    plotter.open_gif(f"direct-poisson-post-hp-{RUN_COUNT:d}.gif", fps=1)
    RUN_COUNT += 1

    for i_round in range(N_ROUNDS):
        mesh = new_mesh
        refinement_settings = RefinementSettings(
            # Specifying how the error is estimated
            error_estimate=ErrorEstimateExplicit(
                # Required by the error function
                target_form=u,
                # The error (estimation) function
                solution_estimate=u_exact,
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
            clim=(1e-20, 1e-4),
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
# To show benefits of local hp-refinement, first consider uniform p-refinement.
# This is the baseline, which is very effective for very smooth solutions.

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
# Now different hp-adaptive strategies can be run.
#
# Baseline
# ~~~~~~~~
#
# As baseline for the hp-runs, the values of 0.1 and 10 were taken as
# h-refinement ratio and element limit count.


BASE_H_RATIO = 0.1
BASE_ELEMENT_LIMIT = 10
errors_local_1, local_mesh_1 = run_refinement_strategy(
    BASE_H_RATIO, BASE_ELEMENT_LIMIT, mesh_initial
)


def err_name(h: float, n: int) -> str:
    """Format name for error plots."""
    return f"(h={h:g}, n={n:d})"


errors_local_name_1 = err_name(BASE_H_RATIO, BASE_ELEMENT_LIMIT)


# %%
#
# More Permissive h-refinement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# By increasing the value of h-refinement error ratio, the refinement stragegy
# can be made more prone to h-refinement over p-refinement.

errors_local_2, local_mesh_2 = run_refinement_strategy(
    BASE_H_RATIO * 10, BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_2 = err_name(BASE_H_RATIO * 10, BASE_ELEMENT_LIMIT)

# %%
#
# More Strict h-refinement
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Inversly, the refinement can be made more prone towards p-refinement only by
# lowering the allowed h-refinement error fraction.

errors_local_3, local_mesh_3 = run_refinement_strategy(
    BASE_H_RATIO / 10, BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_3 = err_name(BASE_H_RATIO / 10, BASE_ELEMENT_LIMIT)


# %%
#
# More Confident Refinement
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# By increasing the number of elements that can be changed during each
# refienemnt round, the effectiveness may drop, since error will be
# affected by each refinement step.

errors_local_4, local_mesh_4 = run_refinement_strategy(
    BASE_H_RATIO, 2 * BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_4 = err_name(BASE_H_RATIO, 2 * BASE_ELEMENT_LIMIT)


# %%
#
# Pure Local p-refinement
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# By setting h-refinement error ratio to zero and with an error estimation
# function, which returns strictly positive values for estimation of h-refinement
# error increase, it can be guaranteed that h-refinement will never occur. This
# can be a good comparison to how does hp-refinement work compared to purely p-refinement.

errors_local_5, local_mesh_5 = run_refinement_strategy(
    0.0, BASE_ELEMENT_LIMIT, mesh_initial
)
errors_local_name_5 = err_name(0.0, BASE_ELEMENT_LIMIT)


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
    (errors_local_name_4, errors_local_4),
    (errors_local_name_5, errors_local_5),
)


plot_mesh_comparisons(
    ("Baseline", mesh_initial),
    (errors_local_name_1, local_mesh_1),
    (errors_local_name_2, local_mesh_2),
    (errors_local_name_3, local_mesh_3),
    (errors_local_name_4, local_mesh_4),
    (errors_local_name_5, local_mesh_5),
)
