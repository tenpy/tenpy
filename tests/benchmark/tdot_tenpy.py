"""To be used in the `-m` argument of benchmark.py."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.linalg.symmetries import groups, spaces
from tenpy.linalg.tensors import Tensor, tdot
from tenpy.linalg.backends.abstract_backend import AbstractBackend
from tenpy.linalg.backends.backend_factory import get_backend
from tenpy.tools.misc import to_iterable

from tests.linalg.conftest import random_symmetry_sectors


def _parse_symmetry(symmetry: list[str]) -> groups.Symmetry:
    """Translate --symmetry argparse argument"""
    symmetry = [getattr(groups, s) for s in symmetry]
    symmetry = [s() if isinstance(s, type) else s for s in symmetry]
    assert all(isinstance(s, groups.Symmetry) for s in symmetry)
    if len(symmetry) == 0:
        return groups.no_symmetry
    if len(symmetry) == 1:
        return symmetry[0]
    return groups.ProductSymmetry(symmetry)


def rand_distinct_int(a, b, n):
    """returns n distinct integers from a to b inclusive."""
    if n < 0:
        raise ValueError
    if n > b - a + 1:
        raise ValueError
    return np.sort((np.random.random_integers(a, b - n + 1, size=n))) + np.arange(n)


def rand_partitions(a, b, n):
    """return [a] + `cuts` + [b], where `cuts` are ``n-1`` (strictly ordered) values inbetween."""
    if b - a <= n:
        return np.array(range(a, b + 1))
    else:
        return np.concatenate(([a], rand_distinct_int(a + 1, b - 1, n - 1), [b]))


def get_random_multiplicities(dim: int, num_sectors: int):
    slices = rand_partitions(0, dim, num_sectors)
    assert len(slices) == num_sectors + 1, f'{len(slices)=}, {num_sectors=}'
    return slices[1:] - slices[:-1]


def get_random_leg(symmetry: groups.Symmetry, dim: int, num_sectors: int):
    assert num_sectors <= dim
    multiplicities = get_random_multiplicities(dim=dim, num_sectors=num_sectors)
    assert len(multiplicities) == num_sectors
    sectors = random_symmetry_sectors(symmetry=symmetry, np_random=np.random, len_=num_sectors)
    assert len(sectors) == num_sectors
    return spaces.VectorSpace(symmetry=symmetry, sectors=sectors, multiplicities=multiplicities)


def get_compatible_leg(legs: list[spaces.VectorSpace]) -> spaces.VectorSpace:
    """return a leg such that a tensor with ``legs + [result]`` allows a non-zero # of blocks."""
    fully_compatible = spaces.ProductSpace(legs).dual
    num_sectors = legs[0].num_sectors
    dim = legs[0].dim

    from_compatible = np.random.randint(num_sectors // 2, num_sectors)
    which = np.random.choice(fully_compatible.num_sectors, size=from_compatible, replace=False)
    rest = np.asarray([i for i in range(fully_compatible.num_sectors) if i not in which])
    sectors = fully_compatible.sectors[which]

    from_rest_or_random = num_sectors - from_compatible
    random_sectors = random_symmetry_sectors(symmetry=fully_compatible.symmetry, np_random=np.random,
                                             len_=len(rest))
    rest_and_random = np.concatenate([fully_compatible.sectors[rest], random_sectors])
    rest_and_random = np.unique(rest_and_random, axis=0)
    is_duplicate = np.any(np.all(rest_and_random[:, None, :] == sectors[None, :, :], axis=2), axis=1)
    rest_and_random = rest_and_random[np.logical_not(is_duplicate)]
    which = np.random.choice(len(rest_and_random), size=from_rest_or_random, replace=False)
    sectors = np.concatenate([sectors, rest_and_random[which]])
    assert len(np.unique(sectors, axis=0)) == len(sectors)

    assert sectors.shape == (num_sectors, fully_compatible.symmetry.sector_ind_len)
    return spaces.VectorSpace(
        symmetry=fully_compatible.symmetry, sectors=sectors,
        multiplicities=get_random_multiplicities(dim=dim, num_sectors=num_sectors),
    )


def get_random_tensor(symmetry: groups.Symmetry, backend: AbstractBackend,
                      legs: list[spaces.VectorSpace | None], leg_dim: int, sectors_per_leg: int,
                      real: bool = False):
    assert sectors_per_leg <= leg_dim

    # determine legs
    legs = legs[:]
    missing_legs = [i for i, leg in enumerate(legs) if leg is None]
    while len(missing_legs) > 1:
        legs[missing_legs[0]] = get_random_leg(symmetry=symmetry, dim=leg_dim, num_sectors=sectors_per_leg)
        missing_legs = [i for i, leg in enumerate(legs) if leg is None]
    if len(missing_legs) > 0:
        which, = missing_legs
        legs[which] = get_compatible_leg(legs[:which] + legs[which + 1:])
    
    def random_block(size):
        res = np.random.normal(size=size)
        if real:
            res = res + 1.j * np.random.normal(size=size)
        return res

    return Tensor.from_numpy_func(func=random_block, legs=legs, backend=backend)


def setup_benchmark(symmetry_backend='abelian',  # no_symmetry, abelian, nonabelian
                    block_backend='numpy',  # numpy, torch
                    symmetry='u1_symmetry',
                    legs=2,
                    size=20,
                    sectors=3,
                    **kwargs
                    ):
    if sectors > size:
        sectors = size
    symmetry = _parse_symmetry(to_iterable(symmetry))
    if sectors > symmetry.num_sectors:
        sectors = symmetry.num_sectors
    backend = get_backend(symmetry=symmetry, block_backend=block_backend, symmetry_backend=symmetry_backend)
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
    return (a, b, legs1, legs2)


def benchmark(data):
    a, b, legs1, legs2 = data
    _ = tdot(a, b, legs1, legs2)
    a.backend.synchronize()
