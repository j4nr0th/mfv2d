"""The code that shows how steady heat equation can be solved."""

import numpy as np
import numpy.typing as npt
import pyvista as pv
import rmsh
from matplotlib import pyplot as plt
from mfv2d import (
    KFormSystem,
    KFormUnknown,
    Mesh2D,
    SolverSettings,
    SystemSettings,
    solve_system_2d,
)

NU = -0.05


def a_field(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Advection vector field."""
    return np.stack(((3 * y - x), (2 - y + 0 * x)), axis=-1)


def u_exact(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Exact solution."""
    return 2 * np.cos(np.pi / 2 * x) * np.cos(np.pi / 2 * y)


def q_exact(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Exact gradient of solution."""
    return np.stack(
        (
            np.pi * np.sin(np.pi / 2 * x) * np.cos(np.pi / 2 * y),
            np.pi * np.cos(np.pi / 2 * x) * np.sin(np.pi / 2 * y),
        ),
        axis=-1,
    )


def source_exact(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
    """Exact source term."""
    return (
        np.sum(a_field(x, y) * q_exact(x, y), axis=-1) - NU * np.pi**2 * u_exact(x, y) / 2
    )


if __name__ == "__main__":
    u = KFormUnknown(2, "u", 2)
    v = u.weight
    q = KFormUnknown(2, "q", 1)
    p = q.weight

    system = KFormSystem(
        p.derivative * u + p * q == p ^ u_exact,
        NU * (v * q.derivative) + (v * (a_field * ~q)) == (v * source_exact),
        sorting=lambda f: f.order,
    )
    print(system)

    N = 6
    n1 = N
    n2 = N

    p_vals = np.arange(1, 7)
    e_vals = np.zeros(p_vals.size)

    for ip, pval in enumerate(p_vals):
        # pval = 1
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
            system_settings=SystemSettings(system),
            solver_settings=SolverSettings(
                absolute_tolerance=1e-10, relative_tolerance=0
            ),
            print_residual=False,
            recon_order=25,
        )

        sol: pv.UnstructuredGrid = solution[-1]

        sol.point_data["q_err2"] = np.linalg.norm(
            sol.point_data["q"] - q_exact(sol.points[:, 0], sol.points[:, 1]), axis=-1
        )

        total_error = sol.integrate_data()
        total_q = total_error.point_data["q_err2"]
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
        ylabel="$\\int\\left|\\left| \\nabla \\times u - \\nabla \\times"
        " \\bar{u} \\right|\\right|$",
        yscale="log",
    )
    plt.legend()
    plt.grid()
    plt.show()
