"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import torch
from tenpy.linalg.backends.torch import TorchBlockBackend

import tdot_tenpy


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'torch') in ['torch', 'gpu']
    a, b, legs1, legs2 = tdot_tenpy.setup_benchmark(**kwargs)
    assert isinstance(a.backend, TorchBlockBackend)
    res = a.to_dense_block(), b.to_dense_block(), (tuple(legs1), tuple(legs2))

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all GPU kernels to complete

    return res


def benchmark(data):
    a, b, axes = data
    _ = torch.tensordot(a, b, axes)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all GPU kernels to complete
