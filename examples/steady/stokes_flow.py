"""The code that shows how steady heat equation can be solved."""

import numpy as np
import numpy.typing as npt
import pyvista as pv
import rmsh
from matplotlib import pyplot as plt
from mfv2d import (
    BoundaryCondition2DSteady,
    KFormSystem,
    KFormUnknown,
    Mesh2D,
    SolverSettings,
    SystemSettings,
    solve_system_2d,
)


def vel_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    """Exact velocity."""
    return np.stack(
        (np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)),
        axis=-1,
    )


def prs_exact(x, y):
    """Exact pressure."""
    return 0 * x * y


def vor_exact(x, y):
    """Exact vorticity."""
    return -2 * np.sin(x) * np.sin(y) + 0 * x * y


def momentum_source(x, y):
    """Exact momentum equation source term."""
    return -2 * np.stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)), axis=-1)


if __name__ == "__main__":
    prs = KFormUnknown(2, "prs", 2)
    w_prs = prs.weight
    vel = KFormUnknown(2, "vel", 1)
    w_vel = vel.weight
    vor = KFormUnknown(2, "vor", 0)
    w_vor = vor.weight

    system = KFormSystem(
        w_vor.derivative * vel + w_vor * vor == w_vor ^ vel_exact,
        w_vel * vor.derivative + w_vel.derivative * prs
        == (w_vel ^ prs_exact) + w_vel * momentum_source,
        w_prs * vel.derivative == 0,
        sorting=lambda f: f.order,
    )
    print(system)

    N = 6
    n1 = N
    n2 = N

    p_vals = np.arange(1, 7)
    e_vals = np.zeros(p_vals.size)

    for ip, pval in enumerate(p_vals):
        block = rmsh.MeshBlock(
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

        m, rx, ry = rmsh.create_elliptical_mesh(block)
        assert rx < 1e-6 and ry < 1e-6

        if pval == p_vals[0]:
            # Show the mesh for the first time.
            fig, ax = plt.subplots(1, 1)
            xlim, ylim = m.plot(ax)
            ax.set_xlim(1.1 * xlim[0], 1.1 * xlim[1])
            ax.set_ylim(1.1 * ylim[0], 1.1 * ylim[1])
            ax.set_aspect("equal")
            plt.show()

        msh = Mesh2D(pval, np.stack((m.pos_x, m.pos_y), axis=-1), m.lines + 1, m.surfaces)
        del m

        solution, stats = solve_system_2d(
            msh,
            system_settings=SystemSettings(
                system,
                constrained_forms=[(0.0, prs)],
                boundary_conditions=[
                    BoundaryCondition2DSteady(vel, msh.boundary_indices, vel_exact)
                ],
            ),
            solver_settings=SolverSettings(
                absolute_tolerance=1e-10, relative_tolerance=0, maximum_iterations=1
            ),
            recon_order=25,
        )

        sol: pv.UnstructuredGrid = solution[-1]
        sol.point_data["vel_err2"] = np.linalg.norm(
            sol.point_data["vel"] - vel_exact(sol.points[:, 0], sol.points[:, 1]), axis=-1
        )
        sol.point_data["vor_err2"] = np.abs(
            sol.point_data["vor"] - vor_exact(sol.points[:, 0], sol.points[:, 1])
        )
        sol.point_data["prs_err2"] = np.abs(
            sol.point_data["prs"] - prs_exact(sol.points[:, 0], sol.points[:, 1])
        )

        total_error = sol.integrate_data()
        total_q = total_error.point_data["vel_err2"]
        e_vals[ip] = total_q[0]
        print(f"Finished {pval=:d}")

    k1, k0 = np.polyfit((p_vals), np.log(e_vals), 1)
    k1, k0 = np.exp(k1), np.exp(k0)

    print(f"Solution converges with p as: {k0:.3g} * ({k1:.3g}) ** p")
    plt.figure()

    plt.scatter(p_vals, e_vals)
    plt.semilogy(
        p_vals,
        k0 * k1**p_vals,
        label=f"${k0:.3g} \\cdot \\left( {{{k1:+.3g}}}^p \\right)$",
        linestyle="dashed",
    )
    plt.gca().set(
        xlabel="$p$",
        ylabel="$\\left|\\left| \\vec{v} - \\bar{\\vec{v}} \\right|\\right|$",
        yscale="log",
    )
    plt.legend()
    plt.grid()
    plt.show()
