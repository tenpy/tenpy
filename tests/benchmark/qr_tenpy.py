"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

from tenpy.linalg.matrix_operations import qr
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
    a = get_random_tensor(symmetry=symmetry, backend=backend, legs=[None] * (2 * legs), leg_dim=size,
                          sectors_per_leg=sectors)
    a.test_sanity()
    q_legs = list(range(legs))
    r_legs = list(range(legs, 2* legs))

    a.backend.synchronize()
    return a, q_legs, r_legs


def benchmark(data):
    a, q_legs, r_legs = data
    _ = qr(a, q_legs, r_legs)
    a.backend.synchronize()
