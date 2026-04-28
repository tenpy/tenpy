"""Utility functions to access backend instances."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import logging

from ..block_backends import NumpyBlockBackend, TorchBlockBackend
from ..dummy_config import config
from ..symmetries import AbelianGroup, Symmetry, no_symmetry
from ._backend import TensorBackend
from .abelian import AbelianBackend
from .fusion_tree_backend import FusionTreeBackend
from .no_symmetry import NoSymmetryBackend

logger = logging.getLogger(__name__)

_tensor_backend_classes = dict(  # values: (cls, kwargs)
    no_symmetry=(NoSymmetryBackend, {}), abelian=(AbelianBackend, {}), fusion_tree=(FusionTreeBackend, {})
)
_block_backends = dict(  # values: (cls, kwargs)
    numpy=(NumpyBlockBackend, {}),
    torch=(TorchBlockBackend, {}),
    tensorflow=None,
    jax=None,
    cpu=(NumpyBlockBackend, {}),
    gpu=(TorchBlockBackend, dict(default_device='cuda')),
    apple_silicon=(TorchBlockBackend, dict(default_device='mps')),
    tpu=None,
)
_instantiated_backends = {}  # keys: (tensor_backend: str, block_backend: str)


def get_backend(symmetry: Symmetry | str = None, block_backend: str = None) -> TensorBackend:
    """Get an instance of an appropriate backend.

    Backends are instantiated only once and then cached. If a suitable backend instance is in
    the cache, that same instance is returned.

    Parameters
    ----------
    symmetry : {'no_symmetry', 'abelian', 'fusion_tree'} | Symmetry
        Specifies which subclass of :class:`TensorBackend` to use, either directly via string,
        or as the minimal version which supports the given symmetry.
    block_backend : {None, 'numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu'}
        Specify which block backend to use.

    """
    if symmetry is None:
        symmetry = config.default_symmetry_backend
    if block_backend is None:
        block_backend = config.default_block_backend

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
