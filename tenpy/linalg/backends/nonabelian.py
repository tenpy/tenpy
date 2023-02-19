# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from abc import ABC, abstractmethod

from tenpy.linalg.backends.abstract_backend import AbstractBackend, AbstractBlockBackend, Data

__all__ = ['AbstractNonabelianBackend']

# TODO eventually remove AbstractBlockBackend inheritance, it is not needed,
#  jakob only keeps it around to make his IDE happy
class AbstractNonabelianBackend(AbstractBackend, AbstractBlockBackend, ABC):

    def tdot(self, a: Data, b: Data, a_axes: list[int], b_axes: list[int]
             ) -> Data:
        ...  # TODO
