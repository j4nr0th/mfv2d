"""Unsteady reaction equation."""

import numpy as np
import numpy.typing as npt
import rmsh
from matplotlib import pyplot as plt
from mfv2d import (
    KFormSystem,
    KFormUnknown,
    Mesh2D,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    solve_system_2d,
)
from scipy.integrate import trapezoid

ALPHA = 0.25


def initial_u(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Screw initial solution."""
    return 2 * np.cos(np.pi * x / 2) * np.cos(np.pi * y / 2)


def initial_q(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Screw initial curl."""
    return np.stack(
        (
            -np.pi * np.cos(np.pi * x / 2) * np.sin(np.pi * y / 2),
            np.pi * np.sin(np.pi * x / 2) * np.cos(np.pi * y / 2),
        ),
        axis=-1,
    )


def final_u(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Steady state forcing."""
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def final_q(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Steady state curl."""
    return np.stack(
        (
            -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y),
            -np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
        ),
        axis=-1,
    )


if __name__ == "__main__":
    u = KFormUnknown(2, "u", 0)
    v = u.weight
    q = KFormUnknown(2, "q", 1)
    p = q.weight

    system = KFormSystem(
        ALPHA * (v * u) == ALPHA * (v * final_u),
        p * q - p * u.derivative == 0,
        sorting=lambda f: f.order,
    )

    N = 6
    P = 3

    n1 = N
    n2 = N
    T_END = 2

    nt_vals = np.array((10, 20, 50, 100, 200))
    er_vals = np.zeros(nt_vals.size)
    dt_vals = np.zeros(nt_vals.size)

    for i_nt, nt in enumerate(nt_vals):
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

        dt = float(T_END / nt)
        solutions, stats = solve_system_2d(
            mesh,
            system_settings=SystemSettings(system, initial_conditions={u: initial_u}),
            solver_settings=SolverSettings(
                maximum_iterations=10, relative_tolerance=0, absolute_tolerance=1e-10
            ),
            time_settings=TimeSettings(dt=dt, nt=nt, time_march_relations={v: u}),
        )

        n_sol = len(solutions)
        err_vals = np.zeros(n_sol)
        time_vals = np.zeros(n_sol)

        for isol, sol in enumerate(solutions):
            time = float(sol.field_data["time"][0])

            u_exact = initial_u(sol.points[:, 0], sol.points[:, 1]) * np.exp(
                -ALPHA * time
            ) + final_u(sol.points[:, 0], sol.points[:, 1]) * (1 - np.exp(-ALPHA * time))
            q_exact = initial_q(sol.points[:, 0], sol.points[:, 1]) * np.exp(
                -ALPHA * time
            ) + final_q(sol.points[:, 0], sol.points[:, 1]) * (1 - np.exp(-ALPHA * time))

            u_err = sol.point_data["u"] - u_exact

            q_err = sol.point_data["q"] - q_exact
            sol.point_data["u_err"] = np.abs(u_err)
            sol.point_data["u_real"] = u_exact
            sol.point_data["q_err"] = np.linalg.norm(q_err, axis=-1)
            sol.point_data["q_real"] = q_exact

            integrated = sol.integrate_data()
            err = float(integrated.point_data["q_err"][0])
            time_vals[isol] = time
            err_vals[isol] = err

        total_time_error = trapezoid(err_vals, time_vals)
        er_vals[i_nt] = total_time_error
        dt_vals[i_nt] = dt
        print(f"For {dt=} total error was {total_time_error:.3e}.")

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
