"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.linalg.tensors import tdot
from tenpy.linalg.backends.backend_factory import get_backend
from tenpy.tools.misc import to_iterable

from misc import parse_symmetry, get_random_tensor


def setup_benchmark(symmetry_backend='abelian',  # no_symmetry, abelian, fusion_tree
                    block_backend='numpy',  # numpy, torch
                    symmetry='u1_symmetry',
                    legs=2,
                    size=20,
                    sectors=3,
                    **kwargs
                    ):
    if sectors > size:
        sectors = size
    symmetry = parse_symmetry(to_iterable(symmetry))
    if sectors > symmetry.num_sectors:
        sectors = symmetry.num_sectors
    backend = get_backend(symmetry=symmetry_backend, block_backend=block_backend)
    legs1 = np.random.choice(2 * legs, legs, replace=False)
    legs2 = np.random.choice(2 * legs, legs, replace=False)
    a = get_random_tensor(symmetry=symmetry, backend=backend, legs=[None] * (2 * legs), leg_dim=size,
                          sectors_per_leg=sectors)
    a.test_sanity()
    b_legs = [None] * (2 * legs)
    # make sure legs on b are contractible with those on a
    for l1, l2 in zip(legs1, legs2):
        b_legs[l2] = a.legs[l1].dual
    b = get_random_tensor(symmetry=symmetry, backend=backend, legs=b_legs, leg_dim=size,
                          sectors_per_leg=sectors)
    b.test_sanity()
    a.backend.synchronize()
    return a, b, legs1, legs2


def benchmark(data):
    a, b, legs1, legs2 = data
    _ = tdot(a, b, legs1, legs2)
    a.backend.synchronize()
