"""Check that pre-computed matrices make sense."""

import numpy as np
import pytest
from interplib.mimetic.mimetic2d import BasisCache


@pytest.mark.parametrize(("n", "m"), ((1, 3), (2, 10), (4, 20), (10, 31)))
def test_mix_01(n: int, m: int) -> None:
    """Check that precomputed mixed matrix of 0-form and 1-form basis works."""
    c = BasisCache(n, m)

    mat = c.mass_mix01_precomp

    for i0 in range((c.basis_order + 1)):
        for j0 in range((c.basis_order + 1)):
            basis_nodal = c.nodal_1d[:, i0][:, None] * c.nodal_1d[:, j0][None, :]

            for i1 in range(c.basis_order + 1):
                for j1 in range(c.basis_order):
                    basis_edge = c.nodal_1d[:, i1][:, None] * c.edge_1d[:, j1][None, :]

                    # x = 1 * c.int_nodes_1d[None, :] + 0 * c.int_nodes_1d[:, None]
                    # y = 0 * c.int_nodes_1d[None, :] + 1 * c.int_nodes_1d[:, None]
                    # xgrd = 1 * c.nodes_1d[None, :] + 0 * c.nodes_1d[:, None]
                    # ygrd = 0 * c.nodes_1d[None, :] + 1 * c.nodes_1d[:, None]

                    v = basis_edge * basis_nodal * c.int_weights_2d

                    err = np.abs(
                        v - mat[i0 * (c.basis_order + 1) + j0, i1 * c.basis_order + j1]
                    )
                    # print(
                    #     "Max error for nodal basis"
                    #     f" {int(i0 * (c.basis_order + 1) + j0)}"
                    #     f" and edge basis eta {int(i1 * c.basis_order + j1)}: "
                    #     f"{np.max(err):.3e}."
                    # )
                    assert 0 == pytest.approx(err.max())

            for i1 in range(c.basis_order):
                for j1 in range(c.basis_order + 1):
                    basis_edge = c.edge_1d[:, i1][:, None] * c.nodal_1d[:, j1][None, :]

                    # x = 1 * c.int_nodes_1d[None, :] + 0 * c.int_nodes_1d[:, None]
                    # y = 0 * c.int_nodes_1d[None, :] + 1 * c.int_nodes_1d[:, None]
                    # xgrd = 1 * c.nodes_1d[None, :] + 0 * c.nodes_1d[:, None]
                    # ygrd = 0 * c.nodes_1d[None, :] + 1 * c.nodes_1d[:, None]

                    v = basis_edge * basis_nodal * c.int_weights_2d

                    err = np.abs(
                        v
                        - mat[
                            i0 * (c.basis_order + 1) + j0,
                            (c.basis_order * (c.basis_order + 1))
                            + i1 * (c.basis_order + 1)
                            + j1,
                        ]
                    )
                    # print(
                    #     "Max error for nodal basis "
                    #     f"{int(i0 * (c.basis_order + 1) + j0)}"
                    #     f" and edge basis xi {int(i1 * (c.basis_order + 1) + j1)}: "
                    #     f"{np.max(err):.3e}."
                    # )
                    assert 0 == pytest.approx(err.max())

                    # plt.figure()
                    # plt.contourf(
                    #     x,
                    #     y,
                    #     np.log10(err if err.min() != 0 else (err + 1e-15)),
                    # )
                    # plt.colorbar()
                    # plt.scatter(xgrd, ygrd, color="red")
                    # plt.gca().set(aspect="equal", xlabel="$x$", ylabel="$y$")
                    # plt.show()


@pytest.mark.parametrize(("n", "m"), ((2, 10), (4, 20), (10, 31)))
def test_mix_12(n: int, m: int) -> None:
    """Check that precomputed mixed matrix of 1-form and 2-form basis works."""
    c = BasisCache(n, m)

    mat = c.mass_mix12_precomp

    for i0 in range((c.basis_order)):
        for j0 in range((c.basis_order)):
            basis_surf = c.edge_1d[:, i0][:, None] * c.edge_1d[:, j0][None, :]

            for i1 in range(c.basis_order + 1):
                for j1 in range(c.basis_order):
                    basis_edge = c.nodal_1d[:, i1][:, None] * c.edge_1d[:, j1][None, :]

                    # x = 1 * c.int_nodes_1d[None, :] + 0 * c.int_nodes_1d[:, None]
                    # y = 0 * c.int_nodes_1d[None, :] + 1 * c.int_nodes_1d[:, None]
                    # xgrd = 1 * c.nodes_1d[None, :] + 0 * c.nodes_1d[:, None]
                    # ygrd = 0 * c.nodes_1d[None, :] + 1 * c.nodes_1d[:, None]

                    v = basis_edge * basis_surf * c.int_weights_2d

                    err = np.abs(
                        v - mat[i1 * c.basis_order + j1, i0 * (c.basis_order) + j0]
                    )
                    # print(
                    #     "Max error for nodal basis"
                    #     f" {int(i0 * (c.basis_order + 1) + j0)}"
                    #     f" and edge basis eta {int(i1 * c.basis_order + j1)}: "
                    #     f"{np.max(err):.3e}."
                    # )
                    assert 0 == pytest.approx(err.max())

            for i1 in range(c.basis_order):
                for j1 in range(c.basis_order + 1):
                    basis_edge = c.edge_1d[:, i1][:, None] * c.nodal_1d[:, j1][None, :]

                    # x = 1 * c.int_nodes_1d[None, :] + 0 * c.int_nodes_1d[:, None]
                    # y = 0 * c.int_nodes_1d[None, :] + 1 * c.int_nodes_1d[:, None]
                    # xgrd = 1 * c.nodes_1d[None, :] + 0 * c.nodes_1d[:, None]
                    # ygrd = 0 * c.nodes_1d[None, :] + 1 * c.nodes_1d[:, None]

                    v = basis_edge * basis_surf * c.int_weights_2d

                    err = np.abs(
                        v
                        - mat[
                            (c.basis_order * (c.basis_order + 1))
                            + i1 * (c.basis_order + 1)
                            + j1,
                            i0 * (c.basis_order) + j0,
                        ]
                    )
                    # print(
                    #     "Max error for nodal basis "
                    #     f"{int(i0 * (c.basis_order + 1) + j0)}"
                    #     f" and edge basis xi {int(i1 * (c.basis_order + 1) + j1)}: "
                    #     f"{np.max(err):.3e}."
                    # )
                    assert 0 == pytest.approx(err.max())

                    # plt.figure()
                    # plt.contourf(
                    #     x,
                    #     y,
                    #     np.log10(err if err.min() != 0 else (err + 1e-15)),
                    # )
                    # plt.colorbar()
                    # plt.scatter(xgrd, ygrd, color="red")
                    # plt.gca().set(aspect="equal", xlabel="$x$", ylabel="$y$")
                    # plt.show()
