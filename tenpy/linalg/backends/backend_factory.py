# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations

import logging

from .abstract_backend import Backend
from .numpy import NoSymmetryNumpyBackend, AbelianNumpyBackend
from .torch import NoSymmetryTorchBackend, AbelianTorchBackend
from ..groups import Symmetry, no_symmetry, AbelianGroup

__all__ = ['get_backend', 'todo_get_backend']

logger = logging.getLogger(__name__)


_backend_lookup = dict(
    no_symmetry=dict(
        numpy=(NoSymmetryNumpyBackend, {}),
        torch=(NoSymmetryTorchBackend, {}),
        tensorflow=None,  # TODO
        jax=None,  # TODO
        cpu=(NoSymmetryNumpyBackend, {}),
        gpu=(NoSymmetryTorchBackend, dict(device='cuda')),
        tpu=None,  # TODO
    ),
    #
    abelian=dict(
        numpy=(AbelianNumpyBackend, {}),
        torch=(AbelianTorchBackend, {}),
        tensorflow=None,  # FUTURE
        jax=None,  # FUTURE
        cpu=(AbelianNumpyBackend, {}),
        gpu=(AbelianTorchBackend, dict(device='cuda')),
        tpu=None,  # FUTURE
    ),
    #
    non_abelian=dict(
        numpy=None,  # FUTURE
        torch=None,  # FUTURE
        tensorflow=None,  # FUTURE
        jax=None,  # FUTURE
        cpu=None,  # FUTURE
        gpu=None,  # FUTURE
        tpu=None,  # FUTURE
    ),
)

_instantiated_backends = {}  # keys: (symmetry_backend, block_backend, kwargs)


def get_backend(symmetry: Symmetry | str = None, block_backend: str = None) -> Backend:
    """
    Parameters
    ----------
    symmetry : {'no_symmetry', 'abelian', 'nonabelian'} | Symmetry
    block_backend : {None, 'numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu'}
    """
    # TODO these are a dummies, in the future we should have some mechanism to store the default
    # values in some state-ful global config of tenpy
    if symmetry is None:
        symmetry = 'abelian'
    if block_backend is None:
        block_backend = 'numpy'

    if isinstance(symmetry, Symmetry):
        # figure out minimal symmetry_backend that supports that symmetry
        if symmetry == no_symmetry:
            symmetry_backend = 'no_symmetry'
        elif isinstance(symmetry, AbelianGroup):
            symmetry_backend = 'abelian'
        else:
            symmetry_backend = 'nonabelian'
    else:
        symmetry_backend = symmetry

    assert block_backend in ['numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu']
    assert symmetry_backend in ['no_symmetry', 'abelian', 'nonabelian']

    res = _backend_lookup[symmetry_backend][block_backend]
    if res is None:
        raise NotImplementedError(f'Backend not implemented {symmetry_backend} & {block_backend}')
    cls, kwargs = res

    key = (symmetry_backend, block_backend, tuple(kwargs.items()))
    if key not in _instantiated_backends:
        backend = cls(**kwargs)
        _instantiated_backends[key] = backend
    else:
        backend = _instantiated_backends[key]

    if isinstance(symmetry, Symmetry):
        assert backend.supports_symmetry(symmetry)
        
    return backend


def todo_get_backend():
    """temporary tool during development. Allows to get a backend.

    TODO revisit usages and decide if backends should be passed around through inits or a
    global state of tenpy
    """
    return get_backend(block_backend='numpy', symmetry_backend='abelian')
