from __future__ import annotations
from abc import ABC

from tenpy.linalg.backends.abstract_backend import AbstractBackend, AbstractBlockBackend, BackendArray


# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy
from ..symmetries import Symmetry, AbelianGroup


class AbstractAbelianBackend(AbstractBackend, AbstractBlockBackend, ABC):

    def __init__(self, symmetry: Symmetry):
        assert symmetry.is_abelian
        super().__init__(symmetry=symmetry)

    def tdot(self, a: BackendArray, b: BackendArray, a_axes: list[int], b_axes: list[int]
             ) -> BackendArray:
        ...  # TODO
