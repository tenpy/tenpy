"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import torch
from math import prod
from tenpy.linalg.backends.torch import TorchBlockBackend

import svd_tenpy


def setup_benchmark(**kwargs):
    kwargs = kwargs.copy()
    kwargs['block_backend'] = 'torch'
    a, q_legs, r_legs = svd_tenpy.setup_benchmark(**kwargs)
    assert isinstance(a.backend, TorchBlockBackend)
    return a.to_dense_block(), q_legs, r_legs


def benchmark(data):
    a, q_legs, r_legs = data
    a = torch.permute(a, q_legs + r_legs)
    q_dims = list(a.shape[:len(q_legs)])
    r_dims = list(a.shape[len(q_dims):])
    a = torch.reshape(a, (prod(q_dims), -1))
    u, s, vh = torch.linalg.svd(a)
    u = torch.reshape(u, q_dims + [len(s)])
    vh = torch.reshape(vh, [len(s)] + r_dims)
    try:
        torch.cuda.synchronize()  # wait for all GPU kernels to complete
    except AssertionError:
        pass  # synchronize raises if no GPU is available.

