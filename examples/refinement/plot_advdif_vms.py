"""The example is intended as a way to show how accurate VMS is."""

from functools import partial

import numpy as np
import pyvista as pv
from mfv2d import (
    ErrorEstimateVMS,
    KFormSystem,
    KFormUnknown,
    RefinementLimitElementCount,
    RefinementSettings,
    SystemSettings,
    UnknownFormOrder,
    integrate_over_elements,
    solve_system_2d,
)
from mfv2d.examples import unit_square_mesh
from numpy import typing as npt

# %%
#
# Create the Mesh
# ---------------

mesh = unit_square_mesh(
    4,
    4,
    3,
    deformation=lambda xi, eta: (
        (xi + 0.0 * np.sin(np.pi * eta) * np.sin(np.pi * xi) + 1) / 2,
        (eta - 0.0 * np.sin(np.pi * eta) * np.sin(np.pi * xi) + 1) / 2,
    ),
)

# %%
#
# Define the exact solution
# -------------------------

ALPHA = 1e2


def fun(t: npt.NDArray[np.float64], alpha: float) -> npt.NDArray:
    """Compute the 1D solution."""
    return t - (np.exp(alpha * (t - 1)) - np.exp(-alpha)) / (1 - np.exp(-alpha))


def dfundx(t: npt.NDArray[np.float64], alpha: float) -> npt.NDArray:
    """Compute the 1D solution gradient."""
    return 1 - alpha * np.exp(alpha * (t - 1)) / (1 - np.exp(-alpha))


def d2fundx2(t: npt.NDArray[np.float64], alpha: float) -> npt.NDArray:
    """Compute the 1D solution second derivative."""
    return -(alpha**2) * np.exp(alpha * (t - 1)) / (1 - np.exp(-alpha))


def u_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray:
    """Compute the exact solution."""
    return fun(x, ALPHA) * fun(y, ALPHA)


def grad_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray:
    """Compute the exact gradient."""
    return np.stack(
        (dfundx(x, ALPHA) * fun(y, ALPHA), fun(x, ALPHA) * dfundx(y, ALPHA)),
        axis=-1,
    )


def laplacian_exact(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray:
    """Compute the exact laplacian."""
    return (d2fundx2(x, ALPHA) * fun(y, ALPHA) + fun(x, ALPHA) * d2fundx2(y, ALPHA)) / 100


def advection_field(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray:
    """Compute the exact advection vector field."""
    return np.stack(
        (0 * x * y + 1, 1 + 0 * x**2 + 0 * y),
        axis=-1,
    )


def source_exact(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray:
    """Compute the exact equation source."""
    return laplacian_exact(x, y) - np.sum(
        advection_field(x, y) * grad_exact(x, y), axis=-1
    )


# %%
#
# Define the System
# -----------------

u = KFormUnknown("u", UnknownFormOrder.FORM_ORDER_2)
q = KFormUnknown("q", UnknownFormOrder.FORM_ORDER_1)

v = u.weight
p = q.weight

full_system = KFormSystem(
    100 * (p * q) + p.derivative * u == p ^ u_exact,
    v * q.derivative + 100 * (v * (advection_field * ~q)) == v * source_exact,
    sorting=lambda f: f.order,
)

projection_system = KFormSystem(
    100 * (p * q) + p.derivative * u == p ^ u_exact,
    v * q.derivative == v * laplacian_exact,
    sorting=lambda f: f.order,
)

# %%
#
# Find the Galerkin Soltuion
# --------------------------

solutions, stats, _ = solve_system_2d(
    mesh,
    system_settings=SystemSettings(full_system, over_integration_order=4),
    recon_order=10,
    refinement_settings=RefinementSettings(
        error_estimate=ErrorEstimateVMS(
            target_form=u,
            symmetric_system=KFormSystem(
                100 * (p * q) + p.derivative * u == 0,
                v * q.derivative == 0,
                sorting=lambda f: f.order,
            ),
            antisymmetric_system=KFormSystem(
                0 * (p * q) == 0,
                100 * (v * (advection_field * ~q)) == 0,
                sorting=lambda f: f.order,
            ),
            order_increase=3,
            max_iters=100,
            atol=0,
            rtol=1e-15,
        ),
        refinement_limit=RefinementLimitElementCount(0, 0),
    ),
)
galerkin_solution = solutions[-1]
u_ex = u_exact(galerkin_solution.points[:, 0], galerkin_solution.points[:, 1])
err_g = galerkin_solution.point_data[u.label] - u_ex
galerkin_solution.point_data["error"] = err_g**2
print(
    "Error Galerkin:"
    f" {np.sqrt(galerkin_solution.integrate_data().point_data['error'][0]):.5e}"
)

plotter = pv.Plotter(off_screen=False, window_size=(900, 900))
galerkin_solution.point_data["unresolved"] = np.abs(err_g)
# plotter.add_mesh(galerkin_solution, scalars="unresolved", log_scale=True)
galerkin_solution.points[:, 2] = galerkin_solution.point_data[u.label]
plotter.add_mesh(galerkin_solution, scalars=u.label)
# plotter.view_xy()
plotter.show_grid()
plotter.show()
del plotter

# %%
#
# Find the Projected Solution
# ---------------------------

solutions, stats, _ = solve_system_2d(
    mesh,
    system_settings=SystemSettings(projection_system, over_integration_order=3),
    recon_order=10,
)
projected_solution = solutions[-1]
u_ex = u_exact(projected_solution.points[:, 0], projected_solution.points[:, 1])
err_p = u_ex - projected_solution.point_data[u.label]
projected_solution.point_data["error"] = err_p**2
print(
    "Error Projected:"
    f" {np.sqrt(projected_solution.integrate_data().point_data['error'][0]):.5e}"
)


plotter = pv.Plotter(off_screen=False, window_size=(900, 900))
projected_solution.point_data["unresolved"] = np.abs(err_p)
# plotter.add_mesh(projected_solution, scalars="unresolved", log_scale=True)
projected_solution.points[:, 2] = projected_solution.point_data[u.label]
plotter.add_mesh(projected_solution, scalars=u.label)
# plotter.view_xy()
plotter.show_grid()
plotter.show()
del plotter


# %%
#
# Check Contributions of Unresolved Scales
# ----------------------------------------

unresolved_q = (
    grad_exact(projected_solution.points[:, 0], projected_solution.points[:, 1])
    + projected_solution.point_data[q.label]
)
unresolved_u = (
    u_exact(projected_solution.points[:, 0], projected_solution.points[:, 1])
    - projected_solution.point_data[u.label]
)

unresolved_advection = np.sum(
    unresolved_q
    * advection_field(projected_solution.points[:, 0], projected_solution.points[:, 1]),
    axis=-1,
)


plotter = pv.Plotter(off_screen=False, window_size=(900, 900))
projected_solution.points[:, 2] = 0
projected_solution.point_data["A2"] = unresolved_advection**2
plotter.add_mesh(projected_solution, scalars="A2", log_scale=True)
plotter.view_xy()
plotter.show_grid()
plotter.show()
del plotter


def compute_field(
    grid: pv.UnstructuredGrid,
    label: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray:
    """Compute solution from PyVista's grid at given points."""
    xv = 0 * y + x
    yv = y + 0 * x

    point_set = pv.PolyData(
        np.stack((xv.flatten(), yv.flatten(), np.zeros(xv.size)), axis=-1)
    )
    interpolated = point_set.sample(grid)
    del point_set
    result_data = interpolated.point_data[label]
    del interpolated

    return np.reshape(result_data, xv.shape + result_data.shape[2:])


integrals = integrate_over_elements(
    mesh, partial(compute_field, projected_solution, "A2"), orders=10
)
projected_solution.cell_data["integrated"] = integrals


plotter = pv.Plotter(off_screen=False, window_size=(1800, 900), shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(projected_solution, scalars="integrated", log_scale=True)
plotter.view_xy()
plotter.show_grid()

plotter.subplot(0, 1)
galerkin_solution.points[:, 2] = 0
plotter.add_mesh(galerkin_solution, scalars="error_estimate", log_scale=True)
plotter.view_xy()
plotter.show_grid()


plotter.show()
del plotter
