"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np
import svd_tenpy


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, u_legs, vh_legs = svd_tenpy.setup_benchmark(**kwargs)
    return a.to_numpy(), u_legs, vh_legs


def benchmark(data):
    a, u_legs, vh_legs = data
    a = np.transpose(a, u_legs + vh_legs)
    u_dims = list(a.shape[:len(u_legs)])
    v_dims = list(a.shape[len(u_legs):])
    a = np.reshape(a, (np.prod(u_dims), -1))
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    u = np.reshape(u, u_dims + [len(s)])
    vh = np.reshape(vh, [len(s)] + v_dims)
