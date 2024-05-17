"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np
import qr_tenpy


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, q_legs, r_legs = qr_tenpy.setup_benchmark(**kwargs)
    return a.to_numpy(), q_legs, r_legs


def benchmark(data):
    a, q_legs, r_legs = data
    a = np.transpose(a, q_legs + r_legs)
    q_dims = list(a.shape[:len(q_legs)])
    v_dims = list(a.shape[len(q_legs):])
    a = np.reshape(a, (np.prod(q_dims), -1))
    q, r = np.linalg.qr(a, mode='reduced')
    new_dim = q.shape[-1]
    q = np.reshape(q, q_dims + [new_dim])
    r = np.reshape(r, [new_dim] + v_dims)
