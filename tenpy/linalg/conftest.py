"""Provide test configuration for backends etc."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from . import backends
from .symmetries import groups, spaces


@pytest.fixture(params=[groups.no_symmetry,
                        groups.u1_symmetry,
                        groups.ZNSymmetry(3, "My_Z3_symmetry"),
                        groups.ProductSymmetry([groups.u1_symmetry, groups.z3_symmetry]),
                        # groups.su2_symmetry,  # TODO reintroduce once SU2 is implemented
                        ],
                ids=repr)
def some_symmetry(request):
    return request.param

def _get_example_symmetry_sectors(some_symmetry: groups.Symmetry) -> groups.SectorArray:
    # TODO: should this be sorted? Unique entries?  fixed number of entries? or just <= 10?
    if isinstance(some_symmetry, groups.SU2Symmetry):
        return np.arange(0, 8, 2, dtype=int)[:, None]
    elif isinstance(some_symmetry, groups.U1Symmetry):
        return np.array([-2, 4, 5, 1, 0, 492])[:, np.newaxis]
    elif isinstance(some_symmetry, groups.ProductSymmetry):
        factor_sectors = [_get_example_symmetry_sectors(factor) for factor in some_symmetry.factors]
        combs = np.indices([len(s) for s in factor_sectors]).T.reshape((-1, len(factor_sectors)))
        keep = [3, 2, 5, 0, 1, 7, 9, 30, 11, 12]
        return np.array([combs[i] for i in keep if i < len(combs)])
    elif some_symmetry.num_sectors < np.inf:
        res = some_symmetry.all_sectors()
        if len(res) >= 10:
            res = res[0, 3, 2, 6, 7, 8]
        elif len(res) >= 4:
            res = res[:4]
        else:
            assert len(res) > 0
        assert res.shape[1] == some_symmetry.sector_ind_len
        return res
    pytest.skip("don't know how to get symmetry sectors")  # raises Skipped


@pytest.fixture
def some_symmetry_sectors(some_symmetry: groups.Symmetry) -> groups.SectorArray:
    # separate function to allow recursion - pytest does not allow to call fixtures directly
    return _get_example_symmetry_sectors(some_symmetry)

@pytest.fixture
def some_symmetry_multiplicities(some_symmetry_sectors):
    mults = np.array([2, 1, 3, 5, 1, 3, 2, 2, 1, 3, 4, 5, 4, 3, 1, 10])
    assert len(some_symmetry_sectors) <= len(mults) , "more sectors than expected!"
    return mults[:len(some_symmetry_sectors)]


@pytest.fixture(params=[spaces.VectorSpace, backends.abelian.AbelianBackendVectorSpace], ids=repr)
def VectorSpace(request):
    return request.param


@pytest.fixture
def ProductSpace(VectorSpace):
    # TODO (JH) requires VectorSpace.ProductSpace class attribute!!!
    if VectorSpace == backends.abelian.AbelianBackendVectorSpace:
        return backends.abelian.AbelianBackendProductSpace
    return spaces.ProductSpace


@pytest.fixture
def default_backend():
    return backends.backend_factory.get_default_backend()


@pytest.fixture(params=['numpy', 'torch'])
def some_block_backend(request):
    if request.name == 'torch':
        torch = pytest.importorskip('torch', reason='torch not installed')
    return request.name


@pytest.fixture
def some_backend(some_symmetry, some_block_backend):
    return backends.backend_factory.get_backend(some_symmetry, some_block_backend)


