"""Provide test configuration for backends etc.

TODO format this summary of fixtures:

np_random : np.random.Generator
symmetry : Symmetry
symmetry_sectors_rng : function (size: int, sort: bool = False) -> SectorArray
vector_space_rng : function (max_num_blocks: int = 4, max_block_size: int = 8) -> VectorSpace
default_backend : AbstractBackend
block_backend : str
backend : AbstractBackend
block_rng : function (size: list[int], real: bool = True) -> Block
backend_data_rng : function (legs: list[VectorSpace], real: bool = True) -> BackendData
tensor_rng : function (legs: list[VectorSpace] = None, num_legs: int = 2, labels: list[str] = None,
                       max_num_block: int = 5, max_block_size: int = 5, real: bool = True) -> Tensor
"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from tenpy.linalg import backends, groups, spaces, tensors


@pytest.fixture
def np_random() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture(params=['numpy']) # TODO: reintroduce 'torch'])
def block_backend(request):
    if request.param == 'torch':
        torch = pytest.importorskip('torch', reason='torch not installed')
    return request.param


@pytest.fixture(params=['no_symmetry', 'abelian'])  # TODO include nonabelian
def symmetry_backend(request):
    return request.param


@pytest.fixture
def backend(symmetry_backend, block_backend):
    return backends.backend_factory.get_backend(symmetry_backend, block_backend)


@pytest.fixture
def default_backend():
    return backends.backend_factory.get_backend()


@pytest.fixture(params=[groups.no_symmetry,
                        groups.u1_symmetry,
                        groups.z2_symmetry,
                        groups.ZNSymmetry(4, "My_Z4_symmetry"),
                        groups.ProductSymmetry([groups.u1_symmetry, groups.z3_symmetry]),
                        # groups.su2_symmetry,  # TODO reintroduce once SU2 is implemented
                        ],
                ids=['NoSymm', 'U(1)', 'Z2', 'Z4', 'U(1)xZ3'])
def symmetry(request, backend):
    symm = request.param
    if not backend.supports_symmetry(symm):
        # TODO find a way to hide this in the report, i.e. to not show it as skipped.
        #      hope and pray that pytest merges https://github.com/pytest-dev/pytest/issues/3730
        #      i guess?
        #
        #      I also found approaches that use the pytest_collection_modifyitems hook
        #      but it looks impossible to get the fixture values at collect time
        pytest.skip('Backend does not support symmetry')
    return symm


def random_symmetry_sectors(symmetry: groups.Symmetry, np_random: np.random.Generator, len_=None,
                            sort: bool = False) -> groups.SectorArray:
    """random unique symmetry sectors, optionally sorted"""
    if len_ is None:
        len_ = np_random.integers(3,7)
    if isinstance(symmetry, groups.SU2Symmetry):
        res = np.arange(0, 2*len_, 2, dtype=int)[:, np.newaxis]
    elif isinstance(symmetry, groups.U1Symmetry):
        vals = list(range(-len_, len_)) + [123]
        res = np_random.choice(vals, replace=False, size=(len_, 1))
    elif symmetry.num_sectors < np.inf:
        if symmetry.num_sectors <= len_:
            res = np_random.permutation(symmetry.all_sectors())
        else:
            which = np_random.choice(symmetry.num_sectors, replace=False, size=len_)
            res = symmetry.all_sectors()[which, :]
    elif isinstance(symmetry, groups.ProductSymmetry):
        factor_len = max(3, len_ // len(symmetry.factors))
        factor_sectors = [random_symmetry_sectors(factor, np_random, factor_len)
                          for factor in symmetry.factors]
        combs = np.indices([len(s) for s in factor_sectors]).T.reshape((-1, len(factor_sectors)))
        if len(combs) > len_:
            combs = np_random.choice(combs, replace=False, size=len_)
        res = np.hstack([fs[i] for fs, i in zip(factor_sectors, combs.T)])
    else:
        pytest.skip("don't know how to get symmetry sectors")  # raises Skipped
    if sort:
        order = np.lexsort(res.T)
        res = res[order]
    return res


@pytest.fixture
def symmetry_sectors_rng(symmetry, np_random):
    def generator(size: int, sort: bool = False):
        """generate random symmetry sectors"""
        return random_symmetry_sectors(symmetry, np_random, len_=size, sort=sort)
    return generator


def random_vector_space(symmetry, max_num_blocks=5, max_block_size=5, np_random=None):
    if np_random is None:
        np_random = np.random.default_rng()
    len_ = np_random.integers(1, max_num_blocks, endpoint=True)
    sectors = random_symmetry_sectors(symmetry, np_random, len_, sort=True)
    # if there are very few sectors, e.g. for symmetry==NoSymmetry(), dont let them be one-dimensional
    min_mult = min(max_block_size, max(4 - len(sectors), 1))
    mults = np_random.integers(min_mult, max_block_size, size=(len(sectors),), endpoint=True)
    dim = np.sum(symmetry.batch_sector_dim(sectors) * mults)
    basis_perm = np_random.permutation(dim) if np_random.random() < 0.7 else None
    res = spaces.VectorSpace(
        symmetry, sectors, mults, basis_perm=basis_perm, is_real=False
    )
    if np_random.random() < 0.5:
        res = res.dual
    res.test_sanity()
    return res


@pytest.fixture
def vector_space_rng(symmetry, np_random):
    def generator(max_num_blocks: int = 4, max_block_size=8):
        """generate random spaces.VectorSpace instances."""
        return random_vector_space(symmetry, max_num_blocks, max_block_size, np_random)
    return generator


@pytest.fixture
def block_rng(backend, np_random):
    def generator(size, real=True):
        block = np_random.normal(size=size)
        if not real:
            block = block + 1.j * np_random.normal(size=size)
        return backend.block_from_numpy(block)
    return generator


@pytest.fixture
def backend_data_rng(backend, block_rng, np_random):
    def generator(legs, real=True, empty_ok=False):
        data = backend.from_block_func(block_rng, legs, func_kwargs=dict(real=real))
        if isinstance(backend, backends.abelian.AbstractAbelianBackend):
            if np_random.random() < 0.5:  # with 50% probability
                # keep roughly half of the blocks
                keep = (np_random.random(len(data.blocks)) < 0.5)
                if not np.any(keep):
                    # but keep at least one
                    # note that using [0:1] instead of [0] is robust in case ``keep.shape == (0,)``
                    keep[0:1] = True
                data.blocks = [block for block, k in zip(data.blocks, keep) if k]
                data.block_inds = data.block_inds[keep]
            if (not empty_ok) and (len(data.blocks) == 0):
                raise ValueError('Empty data was generated. If thats ok, supress with `empty_ok=True`')
        return data
    return generator


@pytest.fixture
def tensor_rng(backend, backend_data_rng, vector_space_rng, symmetry, np_random):
    def generator(legs=None, num_legs=None, labels=None, max_num_blocks=5, max_block_size=5,
                  real=True, empty_ok=False):
        if num_legs is None:
            if legs is not None:
                num_legs = len(legs)
            elif labels is not None:
                num_legs = len(labels)
            else:
                num_legs = 2
        if labels is not None:
            assert len(labels) == num_legs
        if legs is None:
            legs = [None] * num_legs
        else:
            assert len(legs) == num_legs
        legs = list(legs)
        missing_legs = [i for i, leg in enumerate(legs) if leg is None]
        last_missing = missing_legs[-1] if len(missing_legs) > 0 and len(legs) > 1 else -1
        for i, leg in enumerate(legs):
            if leg is None:
                if i != last_missing:
                    legs[i] = vector_space_rng(max_num_blocks, max_block_size)
            else:
                legs[i] = backend.add_leg_metadata(leg)
        if len(legs) == len(missing_legs) == 1:
            # the recipe below assumes that there are some non-missing legs.
            # so we need to deal with this special case first.
            compatible_leg = vector_space_rng(max_num_blocks, max_block_size)
            # ensure that the leg has the trivial sector, so we can have blocks
            if not np.any(np.all(compatible_leg.sectors == symmetry.trivial_sector[None, :], axis=1)):
                compatible_leg._non_dual_sectors[0, :] = symmetry.trivial_sector
            legs[last_missing] = compatible_leg
        elif last_missing != -1:
            # generate compatible leg such that tensor can have non-zero blocks given the charges
            compatible = legs[:]
            compatible.pop(last_missing)
            compatible_leg = spaces.ProductSpace(compatible, backend=backend).as_VectorSpace().dual
            if compatible_leg.num_sectors > max_num_blocks:
                keep = np_random.choice(compatible_leg.num_sectors, max_num_blocks, replace=False)
                keep = np.sort(keep)
                compatible_leg = spaces.VectorSpace(
                    symmetry=compatible_leg.symmetry,
                    sectors=compatible_leg._non_dual_sectors[keep, :],
                    multiplicities=np.maximum(compatible_leg.multiplicities[keep], max_block_size),
                    is_real=compatible_leg.is_real,
                    _is_dual=compatible_leg.is_dual
                )
            legs[last_missing] = compatible_leg
        data = backend_data_rng(legs, real=real, empty_ok=empty_ok)
        return tensors.Tensor(data, backend=backend, legs=legs, labels=labels)
    return generator
