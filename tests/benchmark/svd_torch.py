"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import torch
from math import prod
from tenpy.linalg.backends.torch import TorchBlockBackend

import svd_tenpy


def setup_benchmark(**kwargs):
    kwargs = kwargs.copy()
    kwargs['block_backend'] = 'torch'
    a, u_legs, vh_legs = svd_tenpy.setup_benchmark(**kwargs)
    assert isinstance(a.backend, TorchBlockBackend)
    return a.to_dense_block(), u_legs, vh_legs


def benchmark(data):
    a, u_legs, vh_legs = data
    a = torch.permute(a, u_legs + vh_legs)
    u_dims = list(a.shape[:len(u_legs)])
    v_dims = list(a.shape[len(u_dims):])
    a = torch.reshape(a, (prod(u_dims), -1))
    u, s, vh = torch.linalg.svd(a)
    u = torch.reshape(u, u_dims + [len(s)])
    vh = torch.reshape(vh, [len(s)] + v_dims)
    try:
        torch.cuda.synchronize()  # wait for all GPU kernels to complete
    except AssertionError:
        pass  # synchronize raises if no GPU is available.

