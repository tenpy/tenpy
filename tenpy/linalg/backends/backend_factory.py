# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations

import logging

from .abstract_backend import TensorBackend
from .no_symmetry import NoSymmetryBackend
from .abelian import AbelianBackend
from .fusion_tree_backend import FusionTreeBackend
from .numpy import NumpyBlockBackend
from .torch import TorchBlockBackend
from ..symmetries import Symmetry, no_symmetry, AbelianGroup

__all__ = ['get_backend', 'todo_get_backend']

logger = logging.getLogger(__name__)

_tensor_backend_classes = dict(  # values: (cls, kwargs)
    no_symmetry=(NoSymmetryBackend, {}),
    abelian=(AbelianBackend, {}),
    fusion_tree=(FusionTreeBackend, {})
)
_block_backends = dict(  # values: (cls, kwargs)
    numpy=(NumpyBlockBackend, {}),
    torch=(TorchBlockBackend, {}),
    tensorflow=None,  # TODO
    jax=None,  # TODO
    cpu=(NumpyBlockBackend, {}),
    gpu=(TorchBlockBackend, dict(device='cuda')),
    apple_silicon=(TorchBlockBackend, dict(device='mps')),
    tpu=None,  # TODO
)
_instantiated_backends = {}  # keys: (tensor_backend: str, block_backend: str)


def get_backend(symmetry: Symmetry | str = None, block_backend: str = None) -> TensorBackend:
    """
    Parameters
    ----------
    symmetry : {'no_symmetry', 'abelian', 'fusion_tree'} | Symmetry
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
            tensor_backend = 'no_symmetry'
        elif isinstance(symmetry, AbelianGroup):
            tensor_backend = 'abelian'
        else:
            tensor_backend = 'fusion_tree'
    else:
        tensor_backend = symmetry

    key = (tensor_backend, block_backend)
    backend = _instantiated_backends.get(key, None)
    if backend is not None:
        return backend

    BlockBackendCls, block_kwargs = _block_backends[block_backend]
    TensorBackendCls, tensor_kwargs = _tensor_backend_classes[tensor_backend]
    backend = TensorBackendCls(block_backend=BlockBackendCls(**block_kwargs), **tensor_kwargs)

    if isinstance(symmetry, Symmetry):
        assert backend.supports_symmetry(symmetry)

    _instantiated_backends[key] = backend
    return backend


def todo_get_backend():
    """temporary tool during development. Allows to get a backend.

    TODO revisit usages and decide if backends should be passed around through inits or a
    global state of tenpy
    """
    return get_backend(block_backend='numpy', symmetry_backend='abelian')
