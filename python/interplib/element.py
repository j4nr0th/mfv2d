"""Implementation of elements."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Concatenate, Generic, ParamSpec, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from interplib.mimetic.mimetic2d import Element2D, ElementLeaf2D, ElementNode2D


@dataclass(frozen=True)
class ArrayCom:
    """Shared array coordination type."""

    element_cnt: int


_ElementDataType = TypeVar("_ElementDataType", bound=np.generic)
_IntegerDataType = TypeVar("_IntegerDataType", bound=np.integer)
_FuncArgs = ParamSpec("_FuncArgs")


@dataclass(frozen=True)
class ElementArrayBase(Generic[_ElementDataType]):
    """Base of element arrays."""

    com: ArrayCom
    values: tuple[npt.NDArray[_ElementDataType], ...]
    dtype: type[_ElementDataType]

    def __getitem__(self, i: SupportsIndex, /) -> npt.NDArray[_ElementDataType]:
        """Return element's values."""
        return self.values[i]

    def __setitem__(self, i: SupportsIndex, val: npt.ArrayLike, /) -> None:
        """Set value for the current element."""
        self.values[int(i)][:] = val

    def __len__(self) -> int:
        """Return number of elements."""
        return self.com.element_cnt

    def unique(self) -> npt.NDArray[_ElementDataType]:
        """Find unique values within the array."""
        return np.unique(self.values)


@dataclass(frozen=True)
class FixedElementArray(ElementArrayBase[_ElementDataType]):
    """Array with values and fixed shape per element spread over elements."""

    shape: tuple[int, ...]

    def __init__(
        self,
        com: ArrayCom,
        shape: int | Sequence[int],
        dtype: type[_ElementDataType],
        /,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        values = tuple(np.zeros(shape, dtype=dtype) for _ in range(com.element_cnt))
        object.__setattr__(self, "shape", shape)
        super().__init__(com, values, dtype)


@dataclass(frozen=True)
class FlexibleElementArray(
    ElementArrayBase[_ElementDataType], Generic[_ElementDataType, _IntegerDataType]
):
    """Array with values and variable shape per element."""

    com: ArrayCom
    shapes: FixedElementArray[_IntegerDataType]
    values: tuple[npt.NDArray[_ElementDataType], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate memory for values."""
        vals = tuple(
            np.zeros(self.shapes[i], dtype=self.dtype)
            for i in range(self.com.element_cnt)
        )
        object.__setattr__(self, "values", vals)

    def resize_entry(self, i: int, new_size: int | Sequence[int], /) -> None:
        """Change size of the entry, which also zeroes it."""
        self.shapes[i] = new_size
        object.__setattr__(
            self,
            "values",
            tuple(
                (np.zeros(self.shapes[i], self.dtype) if j == i else v)
                for j, v in enumerate(self.values)
            ),
        )


class ElementSide(IntEnum):
    """Enum specifying the side of an element."""

    SIDE_BOTTOM = 1
    SIDE_RIGHT = 2
    SIDE_TOP = 3
    SIDE_LEFT = 4


def _order_elements(
    ie: int,
    i_current: int,
    ordering: FixedElementArray[np.uint32],
    child_array: FlexibleElementArray[np.uint32, np.uint32],
) -> int:
    """Order elements."""
    ordering[ie] = i_current
    i_current += 1
    for child in child_array[ie]:
        i_current = _order_elements(int(child), i_current, ordering, child_array)

    return i_current


class ElementCollection:
    """Element tree which contains relations and information about elements."""

    com: ArrayCom
    orders_array: FixedElementArray[np.uint32]
    corners_array: FixedElementArray[np.float64]
    parent_array: FixedElementArray[np.uint32]
    child_count_array: FixedElementArray[np.uint32]
    child_array: FlexibleElementArray[np.uint32, np.uint32]
    ordering: FixedElementArray[np.uint32]

    def __init__(self, elements: Sequence[Element2D]) -> None:
        com = ArrayCom(len(elements))
        orders_array = FixedElementArray(com, 2, np.uint32)
        corners_array = FixedElementArray(com, (4, 2), np.float64)
        parent_array = FixedElementArray(com, 1, np.uint32)
        child_count_array = FixedElementArray(com, 1, np.uint32)
        ordering = FixedElementArray(com, 1, np.uint32)

        self.com = com
        self.orders_array = orders_array
        self.corners_array = corners_array
        self.parent_array = parent_array
        self.child_count_array = child_count_array
        self.ordering = ordering

        for ie, element in enumerate(elements):
            if type(element) is ElementNode2D:
                if element.maximum_order is None:
                    order = (0, 0)
                else:
                    order = (element.maximum_order, element.maximum_order)

                child_count = 4

            elif type(element) is ElementLeaf2D:
                order = (element.order, element.order)
                child_count = 0

            else:
                raise TypeError(f"Unknown element type {type(element)}")

            orders_array[ie] = order
            corners_array[ie] = (
                element.bottom_left,
                element.bottom_right,
                element.top_right,
                element.top_left,
            )
            parent_array[ie] = (
                0 if element.parent is None else elements.index(element.parent) + 1
            )
            child_count_array[ie] = child_count

        child_array = FlexibleElementArray(com, np.uint32, child_count_array)
        self.child_array = child_array

        for ie, element in enumerate(elements):
            if type(element) is not ElementNode2D:
                continue
            child_array[ie] = [elements.index(e) for e in element.children()]

        i_current = 0
        for ie in range(len(elements)):
            if parent_array[ie] != 0:
                continue
            i_current = _order_elements(ie, i_current, ordering, child_array)

    def get_element_children(self, i: int, /) -> tuple[int, ...]:
        """Get children of an element."""
        return tuple(int(j) for j in self.child_array[i])

    def get_element_order_on_side(self, i: int, side: ElementSide, /) -> int:
        """Return the order of an element on the side."""
        side = ElementSide(side)
        if self.orders_array[i] != 0:
            return int(self.orders_array[i][1 - (side.value & 1)])

        children = self.child_array[i]
        c1 = int(children[side.value - 1])
        c2 = int(children[side.value & 3])

        return self.get_element_order_on_side(c1, side) + self.get_element_order_on_side(
            c2, side
        )


def call_per_element_flex(
    com: ArrayCom,
    dims: int,
    dtype: type[_ElementDataType],
    fn: Callable[Concatenate[int, _FuncArgs], npt.ArrayLike],
    *args: _FuncArgs.args,
    **kwargs: _FuncArgs.kwargs,
) -> FlexibleElementArray[_ElementDataType, np.uint32]:
    """Call function for each element and return result as a flexible array."""
    shapes = FixedElementArray(com, dims, np.uint32)
    out = FlexibleElementArray(com, dtype, shapes)
    for ie in range(com.element_cnt):
        res = np.asarray(fn(ie, *args, **kwargs), dtype, copy=None)
        out.resize_entry(ie, res.shape)
        out[ie] = res

    return out


def call_per_element_fix(
    com: ArrayCom,
    dtype: type[_ElementDataType],
    shape: int | Sequence[int],
    fn: Callable[Concatenate[int, _FuncArgs], npt.ArrayLike],
    *args: _FuncArgs.args,
    **kwargs: _FuncArgs.kwargs,
) -> FixedElementArray[_ElementDataType]:
    """Call function for each element and return results."""
    results = FixedElementArray(com, shape, dtype)
    for ie in range(com.element_cnt):
        results[ie] = fn(ie, *args, **kwargs)
    return results


class UnknownFormOrder(IntEnum):
    """Orders of unknown differential forms."""

    FORM_ORDER_0 = 1
    FORM_ORDER_1 = 2
    FORM_ORDER_2 = 3


@dataclass(frozen=True)
class UnknownOrderings:
    """Type for storing ordering of unknowns within an element."""

    form_orders: tuple[UnknownFormOrder, ...]

    @property
    def count(self) -> int:
        """Count of forms."""
        return len(self.form_orders)

    def __init__(self, *orders: int) -> None:
        object.__setattr__(
            self, "form_order", tuple(UnknownFormOrder(i + 1) for i in orders)
        )


def _compute_element_dofs(
    ie: int, ordering: UnknownOrderings, elements: ElementCollection
) -> npt.NDArray[np.uint32]:
    sizes = np.zeros(ordering.count, np.uint32)
    if elements.child_count_array[ie] == 0:
        # This is not a child-less array
        order_1: int
        order_2: int
        order_1, order_2 = elements.orders_array[ie]
        for i_f, form in enumerate(ordering.form_orders):
            if form == UnknownFormOrder.FORM_ORDER_0:
                sizes[i_f] = (order_1 + 1) * (order_2 + 1)
            elif form == UnknownFormOrder.FORM_ORDER_1:
                sizes[i_f] = (order_1 + 1) * (order_2) + (order_1) * (order_2 + 1)
            elif form == UnknownFormOrder.FORM_ORDER_2:
                sizes[i_f] = (order_1 + 1) * (order_2) + (order_1) * (order_2 + 1)
            else:
                raise ValueError(f"Unknown form order values {form}.")

    else:
        for i_f, form in enumerate(ordering.form_orders):
            if (
                form == UnknownFormOrder.FORM_ORDER_0
                or form == UnknownFormOrder.FORM_ORDER_1
            ):
                sizes[i_f] = sum(
                    elements.get_element_order_on_side(ie, side) for side in ElementSide
                )
            elif form == UnknownFormOrder.FORM_ORDER_2:
                sizes[i_f] = 0
            else:
                raise ValueError(f"Unknown form order values {form}.")

    return sizes


def _compute_element_lagrange_multipliers(
    ie: int, elements: ElementCollection, ordering: UnknownOrderings
) -> int:
    """Compute number of lagrange multipliers required per element."""
    if elements.child_count_array[ie] == 0:
        return 0

    child_bl, child_br, child_tr, child_tl = (int(i) for i in elements.child_array[ie])

    # There's always the same number of parent-child as the order of the child
    # on that boundary
    n_btm = elements.get_element_order_on_side(
        child_bl, ElementSide.SIDE_BOTTOM
    ) + elements.get_element_order_on_side(child_br, ElementSide.SIDE_BOTTOM)
    n_rth = elements.get_element_order_on_side(
        child_br, ElementSide.SIDE_RIGHT
    ) + elements.get_element_order_on_side(child_tr, ElementSide.SIDE_RIGHT)
    n_top = elements.get_element_order_on_side(
        child_tr, ElementSide.SIDE_TOP
    ) + elements.get_element_order_on_side(child_tl, ElementSide.SIDE_TOP)
    n_lft = elements.get_element_order_on_side(
        child_tl, ElementSide.SIDE_LEFT
    ) + elements.get_element_order_on_side(child_bl, ElementSide.SIDE_LEFT)

    n_bl_br = max(
        elements.get_element_order_on_side(child_bl, ElementSide.SIDE_RIGHT),
        elements.get_element_order_on_side(child_br, ElementSide.SIDE_LEFT),
    )
    n_br_tr = max(
        elements.get_element_order_on_side(child_br, ElementSide.SIDE_TOP),
        elements.get_element_order_on_side(child_tr, ElementSide.SIDE_BOTTOM),
    )
    n_tr_tl = max(
        elements.get_element_order_on_side(child_tr, ElementSide.SIDE_LEFT),
        elements.get_element_order_on_side(child_tl, ElementSide.SIDE_RIGHT),
    )
    n_tl_bl = max(
        elements.get_element_order_on_side(child_tl, ElementSide.SIDE_BOTTOM),
        elements.get_element_order_on_side(child_bl, ElementSide.SIDE_TOP),
    )

    n_first_order = (n_bl_br + n_br_tr + n_tr_tl + n_tl_bl) + (
        n_btm + n_rth + n_top + n_lft
    )

    n_lagrange = 0
    for order in ordering.form_orders:
        if order == 2:
            continue

        n_lagrange += n_first_order

        if order == 0:
            # Add the center node connectivity relations (but not cyclical)
            n_lagrange += 3

    raise NotImplementedError


def compute_dof_sizes(
    elements: ElementCollection, ordering: UnknownOrderings
) -> FixedElementArray[np.uint32]:
    """Compute number of DoFs for each element."""
    return call_per_element_fix(
        elements.com, np.uint32, ordering.count, _compute_element_dofs, ordering, elements
    )


def compute_lagrange_sizes(
    elements: ElementCollection, ordering: UnknownOrderings
) -> FixedElementArray[np.uint32]:
    """Compute number of Lagrange multipliers present within each element."""
    return call_per_element_fix(
        elements.com,
        np.uint32,
        ordering.count,
        _compute_element_lagrange_multipliers,
        elements,
        ordering,
    )


def compute_total_element_sizes(
    elements: ElementCollection,
    dof_sizes: FixedElementArray[np.uint32],
    lagrange_counts: FixedElementArray[np.uint32],
) -> FixedElementArray[np.uint32]:
    """Compute total number of unknowns per element.

    For leaf elements this only means own DoFs, but for any parent elements it
    means the sum of all children.
    """
    sizes = FixedElementArray(elements.com, 1, np.uint32)

    for ie in (int(ie) for ie in reversed(elements.orders_array)):
        children = elements.child_array[ie]
        self_size = dof_sizes[ie]
        lag_size = lagrange_counts[ie]
        child_size = 0
        for c in (int(c) for c in children):
            child_size += int(sizes[c][0])

        sizes[ie] = self_size + lag_size + child_size

    return sizes


def jacobian(
    corners: npt.NDArray[np.float64], nodes_1: npt.ArrayLike, nodes_2: npt.ArrayLike
) -> tuple[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """Compute jacobian of the element with given corners."""
    t0 = np.asarray(nodes_1)
    t1 = np.asarray(nodes_2)

    x0, y0 = corners[0, :]
    x1, y1 = corners[1, :]
    x2, y2 = corners[2, :]
    x3, y3 = corners[3, :]

    dx_dxi = np.astype(
        ((x1 - x0) * (1 - t1) + (x2 - x3) * (1 + t1)) / 4, np.float64, copy=False
    )
    dx_deta = np.astype(
        ((x3 - x0) * (1 - t0) + (x2 - x1) * (1 + t0)) / 4, np.float64, copy=False
    )
    dy_dxi = np.astype(
        ((y1 - y0) * (1 - t1) + (y2 - y3) * (1 + t1)) / 4, np.float64, copy=False
    )
    dy_deta = np.astype(
        ((y3 - y0) * (1 - t0) + (y2 - y1) * (1 + t0)) / 4, np.float64, copy=False
    )
    return ((dx_dxi, dy_dxi), (dx_deta, dy_deta))


def poly_x(
    corner_x: npt.NDArray[np.float64], xi: npt.ArrayLike, eta: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute the x-coordiate of (xi, eta) points."""
    t0 = np.asarray(xi)
    t1 = np.asarray(eta)
    b11 = (1 - t0) / 2
    b12 = (1 + t0) / 2
    b21 = (1 - t1) / 2
    b22 = (1 + t1) / 2
    return np.astype(
        (corner_x[0] * b11 + corner_x[1] * b12) * b21
        + (corner_x[3] * b11 + corner_x[2] * b12) * b22,
        np.float64,
        copy=False,
    )


def poly_y(
    corner_y: npt.NDArray[np.float64], xi: npt.ArrayLike, eta: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compute the y-coordiate of (xi, eta) points."""
    t0 = np.asarray(xi)
    t1 = np.asarray(eta)
    b11 = (1 - t0) / 2
    b12 = (1 + t0) / 2
    b21 = (1 - t1) / 2
    b22 = (1 + t1) / 2
    return np.astype(
        (corner_y[0] * b11 + corner_y[1] * b12) * b21
        + (corner_y[3] * b11 + corner_y[2] * b12) * b22,
        np.float64,
        copy=False,
    )
