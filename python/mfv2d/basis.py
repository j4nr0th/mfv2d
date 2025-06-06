"""Basis and FEM space types."""

from dataclasses import dataclass

from mfv2d._mfv2d import Basis1D


@dataclass(frozen=True)
class Basis2D:
    """Type used to store 2D basis information."""

    basis_xi: Basis1D
    basis_eta: Basis1D

    def __post_init__(self) -> None:
        """Just check that we can support it for now."""
        if self.basis_xi.order != self.basis_eta.order:
            raise NotImplementedError
