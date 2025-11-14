"""Check extension works as intended."""

import sys


def compute_l2_of_laplace_equation(mfv2d, nh: int, nv: int, order: int) -> float:
    """Solve test Laplace equation and compute the L2 norm of the solution."""
    mesh = mfv2d.examples.unit_square_mesh(nh, nv, order)

    def u_exact(x, y):
        return (x * y) ** 2

    u = mfv2d.KFormUnknown("u", mfv2d.UnknownFormOrder.FORM_ORDER_2)
    q = mfv2d.KFormUnknown("q", mfv2d.UnknownFormOrder.FORM_ORDER_1)
    v = u.weight
    p = q.weight

    system = mfv2d.KFormSystem(
        p @ q + p.derivative @ u == p ^ u_exact,
        v @ q.derivative == 0,
    )

    sol, _, _ = mfv2d.solve_system_2d(mesh, mfv2d.SystemSettings(system))

    return float(sol[-1].integrate_data().point_data[u.label][0])


def test_subinterpreters():
    """Check that the module isolation works."""
    NH = 3
    NV = 4
    P = 3

    import mfv2d

    mod1 = mfv2d
    del sys.modules["mfv2d"]
    del mfv2d

    import mfv2d

    mod2 = mfv2d
    del sys.modules["mfv2d"]
    del mfv2d

    assert mod1 != mod2
    assert compute_l2_of_laplace_equation(
        mod1, NH, NV, P
    ) == compute_l2_of_laplace_equation(mod2, NH, NV, P)
