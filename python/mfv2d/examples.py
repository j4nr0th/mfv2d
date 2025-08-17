"""Example setups that are commonly used."""

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from mfv2d._mfv2d import Mesh
from mfv2d.mimetic2d import mesh_create


def unit_square_mesh(
    nh: int,
    nv: int,
    orders: int | Sequence[int],
    deformation: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]],
        tuple[npt.ArrayLike, npt.ArrayLike],
    ]
    | None = None,
) -> Mesh:
    r"""Create a mesh based on the unit square.

    Parameters
    ----------
    nh : int
        Number of elements in the horizontal direction.

    nv : int
        Number of elements in the vertical direction.

    orders : int or Sequence of int
        Order(s) of all elements in the mesh.

    deformation : Callable (array, array) -> (array, array)
        Deformation applied to points of the mesh. Should accept
        positions of in the reference domain that are given on the
        domain :math:`(\xi, \eta) \in [-1, +1]` and map it to real
        geometry :math:`(x, y)`.
    """
    xi, eta = np.meshgrid(np.linspace(-1, +1, nh + 1), np.linspace(-1, +1, nv + 1))
    if deformation is not None:
        p_xi, p_eta = deformation(xi, eta)
        xi[:] = p_xi
        eta[:] = p_eta

    lines_h = [
        ((nh + 1) * j + i + 1, (nh + 1) * j + i + 2)
        for j in range(nv + 1)
        for i in range(nh)
    ]
    lines_v = [
        ((nh + 1) * j + i + 1, (nh + 1) * j + i + nh + 2)
        for j in range(nv)
        for i in range(nh + 1)
    ]
    surfaces = [
        (
            i + nh * j + 1,
            nh * (nv + 1) + j * (nh + 1) + (i + 1) + 1,
            -(i + nh * j + 1 + nh),
            -(nh * (nv + 1) + j * (nh + 1) + i + 1),
        )
        for j in range(nv)
        for i in range(nh)
    ]

    return mesh_create(
        orders,
        np.stack((xi.flatten(), eta.flatten()), axis=-1),
        lines_h + lines_v,
        surfaces,
    )
