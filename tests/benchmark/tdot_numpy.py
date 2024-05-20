"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np
import tdot_tenpy


def setup_benchmark(**kwargs):
    assert kwargs.get('block_backend', 'numpy') == 'numpy'
    a, b, legs1, legs2 = tdot_tenpy.setup_benchmark(**kwargs)
    return a.to_numpy(), b.to_numpy(), (legs1, legs2)


def benchmark(data):
    a, b, axes = data
    _ = np.tensordot(a, b, axes)
