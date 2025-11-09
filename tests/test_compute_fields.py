"""Check that C code can correctly reconstruct fields."""

import numpy as np
import numpy.typing as npt
import pytest
from mfv2d._mfv2d import ElementFemSpace2D, compute_integrating_fields
from mfv2d.kform import Function2D, KFormUnknown, UnknownFormOrder
from mfv2d.mimetic2d import (
    FemCache,
    bilinear_interpolate,
    element_primal_dofs,
    reconstruct,
)
from mfv2d.system import ElementFormSpecification

_ORDER_COMBINATIONS_HOMOGENUS = (
    (1, 1),
    (2, 2),
    (5, 5),
    (7, 7),
)

_ORDER_COMBINATIONS_HETEROGENUS = (
    (1, 2),
    (2, 1),
    (5, 3),
    (3, 5),
    (7, 1),
    (1, 7),
)


@pytest.mark.parametrize(
    ("n1", "n2"), _ORDER_COMBINATIONS_HOMOGENUS + _ORDER_COMBINATIONS_HETEROGENUS
)
def test_explicit(n1: int, n2: int):
    """Check that fields that are results of fields are correct."""
    cache = FemCache(2)

    basis_2d = cache.get_basis2d(n1, n2)
    fem_space = ElementFemSpace2D(
        basis_2d,
        np.array(
            (
                (-1, -1),
                (+1, -1),
                (+1, +1),
                (-1, +1),
            ),
            np.float64,
        ),
    )

    def func_0_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) * np.cos(y**2) - x * y

    def func_2_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) / (2 + np.cos(y**2)) + x * y

    def func_1_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.stack((x + y**2, np.sin(2 * y - x)), axis=-1)

    rule_specs = [
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
    ]

    fake_specs = ElementFormSpecification(
        KFormUnknown("a", UnknownFormOrder.FORM_ORDER_0)
    )
    results = compute_integrating_fields(
        fem_space,
        fake_specs,
        tuple(r[1] for r in rule_specs),
        tuple(r[0] for r in rule_specs),
        np.zeros(fake_specs.total_size(n1, n2), np.float64),
    )
    x = bilinear_interpolate(
        fem_space.corners[:, 0],
        fem_space.basis_xi.rule.nodes[None, :],
        fem_space.basis_eta.rule.nodes[:, None],
    )
    y = bilinear_interpolate(
        fem_space.corners[:, 1],
        fem_space.basis_xi.rule.nodes[None, :],
        fem_space.basis_eta.rule.nodes[:, None],
    )
    for r, v in zip(rule_specs, results, strict=True):
        reconstructed = r[0](x, y)
        assert v == pytest.approx(reconstructed)


@pytest.mark.parametrize(
    ("n1", "n2"), _ORDER_COMBINATIONS_HOMOGENUS + _ORDER_COMBINATIONS_HETEROGENUS
)
def test_solution_based(n1: int, n2: int):
    """Check that fields that are results of fields are correct."""
    cache = FemCache(2)

    basis_2d = cache.get_basis2d(n1, n2)
    fem_space = ElementFemSpace2D(
        basis_2d,
        np.array(
            (
                (-1, -1),
                (+1, -1),
                (+1, +1),
                (-1, +1),
            ),
            np.float64,
        ),
    )

    def func_0_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) * np.cos(y**2) - x * y

    def func_2_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) / (2 + np.cos(y**2)) + x * y

    def func_1_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.stack(((x + y**2), np.sin(2 * y - x)), axis=-1)

    rule_specs = [
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1),
    ]

    dofs = [
        element_primal_dofs(order, fem_space, function)
        for (function, order) in rule_specs
    ]

    merged_dofs = np.concatenate(
        dofs,
        axis=-1,
    )

    form_specs = ElementFormSpecification(
        *(KFormUnknown(f"form-{i}", order) for i, (_, order) in enumerate(rule_specs))
    )

    results = compute_integrating_fields(
        fem_space,
        form_specs,
        tuple(order for _, order in rule_specs),
        tuple(label for label, _ in form_specs),
        merged_dofs,
    )

    for (function, order), v, v_dof in zip(rule_specs, results, dofs, strict=True):
        reconstructed = reconstruct(
            fem_space,
            order,
            v_dof,
            fem_space.basis_xi.rule.nodes[None, :],
            fem_space.basis_eta.rule.nodes[:, None],
        )
        assert v == pytest.approx(reconstructed)


@pytest.mark.parametrize(
    ("n1", "n2"), _ORDER_COMBINATIONS_HOMOGENUS + _ORDER_COMBINATIONS_HETEROGENUS
)
def test_mixed(n1: int, n2: int):
    """Mix both solution and explicit ones cause fuck it."""
    cache = FemCache(2)

    basis_2d = cache.get_basis2d(n1, n2)
    fem_space = ElementFemSpace2D(
        basis_2d,
        np.array(
            (
                (-1, -1),
                (+1, -1),
                (+1, +1),
                (-1, +1),
            ),
            np.float64,
        ),
    )

    def func_0_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) * np.cos(y**2) - x * y

    def func_2_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.sin(x) / (2 + np.cos(y**2)) + x * y

    def func_1_form_test(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.ArrayLike:
        """Test function for 0-forms."""
        return np.stack(((x + y**2), np.sin(2 * y - x)), axis=-1)

    rule_specs = [
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0, True),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2, False),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0, True),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1, True),
        (func_2_form_test, UnknownFormOrder.FORM_ORDER_2, True),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1, False),
        (func_0_form_test, UnknownFormOrder.FORM_ORDER_0, True),
        (func_1_form_test, UnknownFormOrder.FORM_ORDER_1, True),
    ]

    dofs = list()

    field_specs: list[int | Function2D] = []
    i = 0
    unknowns: list[KFormUnknown] = list()
    for function, order, sol_based in rule_specs:
        if sol_based:
            field_specs.append(i)
            dofs.append(element_primal_dofs(order, fem_space, function))
            unknowns.append(KFormUnknown(f"form-{i}", order))
            i += 1
        else:
            field_specs.append(function)

    merged_dofs = np.concatenate(dofs, axis=-1)
    form_specs = ElementFormSpecification(*unknowns)
    reversed_labels = [f.label for f in reversed(unknowns)]

    results = compute_integrating_fields(
        fem_space,
        form_specs,
        tuple(order for _, order, _ in rule_specs),
        tuple(
            func if not sol_based else reversed_labels.pop()
            for (func, _, sol_based) in rule_specs
        ),
        merged_dofs,
    )

    xi_nodes = fem_space.basis_xi.rule.nodes[None, :]
    eta_nodes = fem_space.basis_eta.rule.nodes[:, None]
    x = bilinear_interpolate(fem_space.corners[:, 0], xi_nodes, eta_nodes)
    y = bilinear_interpolate(fem_space.corners[:, 1], xi_nodes, eta_nodes)

    i = 0
    for (function, order, sol_based), v in zip(rule_specs, results, strict=True):
        if not sol_based:
            reconstructed = function(x, y)
            assert v == pytest.approx(reconstructed)
        else:
            v_dof = dofs[i]
            i += 1
            reconstructed = reconstruct(fem_space, order, v_dof, xi_nodes, eta_nodes)
            assert v == pytest.approx(reconstructed)
