"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import torch
from math import prod
from tenpy.linalg.backends.torch import TorchBlockBackend

import qr_tenpy


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'torch') in ['torch', 'gpu']
    a, q_legs, r_legs = qr_tenpy.setup_benchmark(**kwargs)
    assert isinstance(a.backend, TorchBlockBackend)
    res = a.to_dense_block(), q_legs, r_legs

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all GPU kernels to complete

    return res


def benchmark(data):
    a, q_legs, r_legs = data
    a = torch.permute(a, q_legs + r_legs)
    q_dims = list(a.shape[:len(q_legs)])
    r_dims = list(a.shape[len(q_dims):])
    a = torch.reshape(a, (prod(q_dims), -1))
    q, r = torch.qr(a, some=True)
    new_dim = q.shape[-1]
    q = torch.reshape(q, q_dims + [new_dim])
    r = torch.reshape(r, [new_dim] + r_dims)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all GPU kernels to complete
