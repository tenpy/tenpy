# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations

from .backends import AbstractBackend
from .symmetries import VectorSpace
from .tensors import Tensor

__all__ = ['uniform', 'normal']

def uniform() -> Tensor:
    # TODO is there even a useful and basis-independent notion of a uniform probability distribution?
    raise NotImplementedError


def normal(backend: AbstractBackend, *some_args) -> Tensor:
    """
    Generate a sample from the probability distribution p(T) ~ exp(-.5 * (norm(T) / sigma) ** 2)
    """
    raise NotImplementedError  # TODO


# TODO random orthogonal and unitary maps, like in old tenpy
