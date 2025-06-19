"""Steady-state Navier-Stokes flow."""

import numpy as np
import rmsh
from mfv2d import (
    BoundaryCondition2DSteady,
    KFormSystem,
    KFormUnknown,
    Mesh2D,
    RefinementSettings,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    solve_system_2d,
)

RE = 1e1


def boundary_velocty(x, y):
    """Exact velocity solution."""
    vx = (y == 1) + 0 * x
    vy = 0 * (x + y)
    return np.stack((vx, vy), axis=-1)


if __name__ == "__main__":
    pre = KFormUnknown(2, "pre", 2)
    w_pre = pre.weight
    vel = KFormUnknown(2, "vel", 1)
    w_vel = vel.weight
    vor = KFormUnknown(2, "vor", 0)
    w_vor = vor.weight

    system = KFormSystem(
        w_vor.derivative * vel - w_vor * vor == w_vor ^ boundary_velocty,
        # No weak BC for pressure, since normal velocity is given
        (1 / RE) * (w_vel * vor.derivative) + w_vel.derivative * pre
        == -(w_vel * (vel ^ (~vor))),
        w_pre * vel.derivative == 0,
        sorting=lambda f: f.order,
    )
    print(system)

    N = 6
    P = 3

    n1 = N
    n2 = N

    rect_mesh, rx, ry = rmsh.create_elliptical_mesh(
        rmsh.MeshBlock(
            label=None,
            bottom=rmsh.BoundaryCurve.from_knots(n1, (-1, -1), (+1, -1)),
            right=rmsh.BoundaryCurve.from_knots(n2, (+1, -1), (+1, +1)),
            top=rmsh.BoundaryCurve.from_knots(n1, (+1, +1), (-1, +1)),
            left=rmsh.BoundaryCurve.from_knots(n2, (-1, +1), (-1, -1)),
        )
    )
    assert rx < 1e-6, ry < 1e-6

    mesh = Mesh2D(
        P,
        np.stack((rect_mesh.pos_x, rect_mesh.pos_y), axis=-1),
        rect_mesh.lines + 1,
        rect_mesh.surfaces,
    )

    def refine_test(e, _: int) -> bool:
        """Check if element should be refined."""
        corners = np.array([e.bottom_left, e.bottom_right, e.top_right, e.top_left])
        return bool(np.any(np.linalg.norm(corners, axis=-1) > 0.5))

    solutions, stats = solve_system_2d(
        mesh,
        SystemSettings(
            system,
            [BoundaryCondition2DSteady(vel, mesh.boundary_indices, boundary_velocty)],
            [(0.0, pre)],
        ),
        solver_settings=SolverSettings(
            maximum_iterations=20,
            absolute_tolerance=1e-10,
            relative_tolerance=0,
        ),
        time_settings=TimeSettings(dt=1e-1, nt=10, time_march_relations={w_vel: vel}),
        refinement_settings=RefinementSettings(2, refine_test),
        print_residual=True,
        recon_order=25,
    )
    print(stats)

    # for isol, sol in enumerate(solutions):
    #     sol.save(f"sandbox/cavity/res-{isol:04d}.vtu")
