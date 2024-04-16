"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

from tenpy.linalg.matrix_operations import svd
from tenpy.tools.misc import to_iterable

from tdot_tenpy import parse_symmetry, get_backend, get_random_tensor


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
    a = get_random_tensor(symmetry=symmetry, backend=backend, legs=[None] * (2 * legs), leg_dim=size,
                          sectors_per_leg=sectors)
    a.test_sanity()
    u_legs = list(range(legs))
    vh_legs = list(range(legs, 2* legs))
    a.backend.synchronize()
    return a, u_legs, vh_legs


def benchmark(data):
    a, u_legs, vh_legs = data
    _ = svd(a, u_legs, vh_legs)
    a.backend.synchronize()
