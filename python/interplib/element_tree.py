"""Implementation of the element tree type."""

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import interplib.kforms as kforms
from interplib.mimetic.mimetic2d import Element2D, ElementLeaf2D

OrderDivisionFunction = Callable[
    [int, int, int], tuple[int | None, tuple[int, int, int, int]]
]


def check_and_refine(
    pred: Callable[[ElementLeaf2D, int], bool] | None,
    order_div: OrderDivisionFunction,
    e: ElementLeaf2D,
    level: int,
    max_level: int,
) -> list[Element2D]:
    """Return element and potentially its children."""
    out: list[Element2D]
    if level < max_level and pred is not None and pred(e, level):
        # TODO: Make this nicer without this stupid Method mumbo jumbo
        parent_order, child_orders = order_div(e.order, level, max_level)
        new_e, ((ebl, ebr), (etl, etr)) = e.divide(*child_orders, parent_order)
        cbl = check_and_refine(pred, order_div, ebl, level + 1, max_level)
        cbr = check_and_refine(pred, order_div, ebr, level + 1, max_level)
        ctl = check_and_refine(pred, order_div, etl, level + 1, max_level)
        ctr = check_and_refine(pred, order_div, etr, level + 1, max_level)
        object.__setattr__(new_e, "child_bl", cbl[0])
        object.__setattr__(new_e, "child_br", cbr[0])
        object.__setattr__(new_e, "child_tl", ctl[0])
        object.__setattr__(new_e, "child_tr", ctr[0])
        out = cbl + cbr + ctl + ctr + [new_e]

    else:
        out = [e]

    return out


@dataclass(init=False)
class ElementTree:
    """Container for mesh elements."""

    # Tuple of all elements in the tree.
    elements: tuple[Element2D, ...]
    # Offset of the first DoF of a variable within an element
    dof_offsets: tuple[npt.NDArray[np.uint32], ...]
    # Total number of degrees of freedom
    n_dof: int
    # Total number of degrees of freedom in the leaf nodes
    n_dof_leaves: int
    # Total number of level-zero elements
    n_base_elements: int
    # Unique orders of leaf elements.
    unique_orders: tuple[int, ...]
    # Indices of elements on the highest level
    top_indices: tuple[int, ...]
    # Counts of elements of the specific order
    order_counts: dict[int, int]

    def __init__(
        self,
        elements: Sequence[ElementLeaf2D],
        predicate: None | Callable[[ElementLeaf2D, int], bool],
        division_function: OrderDivisionFunction,
        max_levels: int,
        unknowns: Sequence[kforms.KFormUnknown],
    ) -> None:
        # Divide the elements as long as predicate is true.
        all_elems: list[Element2D] = list()
        self.n_base_elements = len(elements)

        # Depth-first refinement
        top_indices: list[int] = list()

        for e in elements:
            top_indices.append(len(all_elems))
            all_elems += check_and_refine(predicate, division_function, e, 0, max_levels)

        self.elements = tuple(all_elems)
        del all_elems

        # Check if elements have children
        n_total = len(self.elements)

        n_dof_leaves = 0
        list_dof_sizes: list[tuple[int, ...]] = list()
        order_cnts: dict[int, int] = dict()
        # Compute DoF offsets within elements
        for element in self.elements:
            n = element.dof_sizes([form.order for form in unknowns])

            if isinstance(element, ElementLeaf2D):
                if element.order in order_cnts:
                    order_cnts[element.order] += 1
                else:
                    order_cnts[element.order] = 1
                n_dof_leaves += sum(n)

            list_dof_sizes.append(n)

        dof_sizes = np.array(list_dof_sizes, np.uint32).T
        del list_dof_sizes

        self.dof_offsets = (
            np.zeros(n_total, np.uint32),
            *np.cumsum(dof_sizes, axis=0),
        )

        self.n_dof_leaves = n_dof_leaves
        self.n_dof = int(np.sum(self.dof_offsets[-1]))
        self.unique_orders = tuple(sorted(order_cnts.keys()))
        self.top_indices = tuple(top_indices)
        self.order_counts = {order: order_cnts[order] for order in self.unique_orders}

    def iter_leaves(self) -> Generator[ElementLeaf2D]:
        """Iterate over leaves."""
        for e in self.elements:
            if isinstance(e, ElementLeaf2D):
                yield e

    def iter_top(self) -> Generator[Element2D]:
        """Iterate top level elements."""
        for e in self.elements:
            if e.parent is None:
                yield e

    def element_edge_dofs(self, index: int, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of edge DoFs on the boundary of an element."""
        return self.elements[index].element_edge_dofs(bnd_idx)

    def element_node_dofs(self, index: int, bnd_idx: int, /) -> npt.NDArray[np.uint32]:
        """Get non-offset indices of nodal DoFs on the boundary of an element."""
        return self.elements[index].element_node_dofs(bnd_idx)

    def leaf_indices(self) -> npt.NDArray[np.uint32]:
        """Return indices of leaf elements."""
        where: list[int] = list()
        for i, e in enumerate(self.elements):
            if isinstance(e, ElementLeaf2D):
                where.append(i)

        return np.array(where, np.uint32)

    @property
    def n_elements(self) -> int:
        """Number of elements in the tree."""
        return len(self.elements)
