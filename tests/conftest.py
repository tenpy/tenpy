"""Provide test configuration for backends etc.

TODO format this summary of fixtures:

np_random : np.random.Generator
symmetry : Symmetry
symmetry_sectors_rng : function (size: int, sort: bool = False) -> SectorArray
vector_space_rng : function (max_num_blocks: int = 4, max_block_size: int = 8) -> VectorSpace
default_backend : Backend
block_backend : str
backend : Backend
block_rng : function (size: list[int], real: bool = True) -> Block
backend_data_rng : function (legs: list[VectorSpace], real: bool = True) -> BackendData
tensor_rng : function (legs: list[VectorSpace] = None, num_legs: int = 2, labels: list[str] = None,
                       max_num_block: int = 5, max_block_size: int = 5, real: bool = True,
                       cls: Type[T] = Tensor) -> T
"""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from tenpy.linalg import backends, spaces, symmetries, tensors


def pytest_addoption(parser):
    parser.addoption(
        "--run-nonabelian", action="store_true", default=False, help="run nonabelian tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-nonabelian"):
        skip_nonabelian = pytest.mark.skip(reason='need --run-nonabelian to run.')
        for item in items:
            if 'nonabelian' in item.keywords:
                item.add_marker(skip_nonabelian)

# FIXTURES:


@pytest.fixture
def np_random() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture(params=['numpy']) # TODO: reintroduce 'torch'])
def block_backend(request):
    if request.param == 'torch':
        torch = pytest.importorskip('torch', reason='torch not installed')
    return request.param


# Note: nonabelian backend is disabled by default. Use ``--run-nonabelian`` CL option to run them.
_nonabelian_param = pytest.param('nonabelian', marks=pytest.mark.nonabelian)
@pytest.fixture(params=['no_symmetry', 'abelian', _nonabelian_param])
def symmetry_backend(request):
    return request.param


@pytest.fixture
def backend(symmetry_backend, block_backend):
    return backends.backend_factory.get_backend(symmetry_backend, block_backend)


@pytest.fixture
def default_backend():
    return backends.backend_factory.get_backend()


@pytest.fixture(params=[symmetries.no_symmetry,
                        symmetries.u1_symmetry,
                        symmetries.z2_symmetry,
                        symmetries.ZNSymmetry(4, "My_Z4_symmetry"),
                        symmetries.ProductSymmetry([symmetries.u1_symmetry, symmetries.z3_symmetry]),
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


def random_symmetry_sectors(symmetry: symmetries.Symmetry, np_random: np.random.Generator, len_=None,
                            sort: bool = False) -> symmetries.SectorArray:
    """random unique symmetry sectors, optionally sorted"""
    if len_ is None:
        len_ = np_random.integers(3,7)
    if isinstance(symmetry, symmetries.SU2Symmetry):
        res = np.arange(0, 2*len_, 2, dtype=int)[:, np.newaxis]
    elif isinstance(symmetry, symmetries.U1Symmetry):
        vals = list(range(-len_, len_)) + [123]
        res = np_random.choice(vals, replace=False, size=(len_, 1))
    elif symmetry.num_sectors < np.inf:
        if symmetry.num_sectors <= len_:
            res = np_random.permutation(symmetry.all_sectors())
        else:
            which = np_random.choice(symmetry.num_sectors, replace=False, size=len_)
            res = symmetry.all_sectors()[which, :]
    elif isinstance(symmetry, symmetries.ProductSymmetry):
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


def random_vector_space(symmetry, max_num_blocks=5, max_block_size=5, is_dual=None, np_random=None):
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
    if (is_dual is None and np_random.random() < 0.5) or (is_dual is True):
        res = res.dual
    res.test_sanity()
    return res


@pytest.fixture
def vector_space_rng(symmetry, np_random):
    def generator(max_num_blocks: int = 4, max_block_size=8, is_dual=None):
        """generate random spaces.VectorSpace instances."""
        return random_vector_space(symmetry, max_num_blocks, max_block_size, is_dual=is_dual,
                                   np_random=np_random)
    return generator


@pytest.fixture
def block_rng(backend, np_random):
    def generator(size, real=True):
        block = np_random.normal(size=size)
        if not real:
            block = block + 1.j * np_random.normal(size=size)
        return backend.block_from_numpy(block)
    return generator


def abelian_data_drop_some_blocks(data, np_random, empty_ok=False, all_blocks=False):
    """Randomly drop some blocks, in-place"""
    # unless all_blocks=True, drop some blocks with 50% probability
    if (not all_blocks) and (np_random.random() < .5):
        keep = (np_random.random(len(data.blocks)) < 0.5)
        if not np.any(keep):
            # keep at least one
            # note that using [0:1] instead of [0] is robust in case ``keep.shape == (0,)``
            keep[0:1] = True
        data.blocks = [block for block, k in zip(data.blocks, keep) if k]
        data.block_inds = data.block_inds[keep]
    if (not empty_ok) and (len(data.blocks) == 0):
        raise ValueError('Empty data was generated. If thats ok, suppress with `empty_ok=True`')
    return data


@pytest.fixture
def backend_data_rng(backend, block_rng, np_random):
    """Generate random data for a Tensor"""
    def generator(legs, real=True, empty_ok=False, all_blocks=False, num_domain_legs=0):
        data = backend.from_block_func(block_rng, legs, num_domain_legs=num_domain_legs,
                                       func_kwargs=dict(real=real))
        if isinstance(backend, backends.abelian.AbelianBackend):
            data = abelian_data_drop_some_blocks(data, np_random=np_random, empty_ok=empty_ok,
                                                 all_blocks=all_blocks)
        return data
    return generator


@pytest.fixture
def tensor_rng(backend, symmetry, np_random, block_rng, vector_space_rng, symmetry_sectors_rng):
    """TODO proper documentation

    ChargedTensor: only creates one-dimensional dummy legs
    """
    def generator(legs=None, num_legs=None, labels=None, max_num_blocks=5, max_block_size=5,
                  real=True, empty_ok=False, all_blocks=False, cls=tensors.BlockDiagonalTensor,
                  num_domain_legs=0):
        # parse legs
        if num_legs is None:
            if legs is not None:
                num_legs = len(legs)
            elif labels is not None:
                num_legs = len(labels)
            else:
                num_legs = 2
        if labels is None:
            labels = [None] * num_legs
        else:
            assert len(labels) == num_legs
        if legs is None:
            legs = [None] * num_legs
        else:
            assert len(legs) == num_legs
        legs = list(legs)

        # deal with other classes
        if cls is tensors.DiagonalTensor:
            second_leg_dual = True
            if len(legs) == 1:
                leg = legs[0]
            elif len(legs) == 2:
                if legs[0] is None:
                    leg = legs[1]
                elif legs[1] is None:
                    leg = legs[0]
                else:
                    leg = legs[0]
                    assert leg.is_equal_or_dual(legs[1])
                    second_leg_dual = (leg.is_dual != legs[1].is_dual)
            else:
                raise ValueError('Invalid legs. Expected none, one or two')
            if leg is None:
                leg = vector_space_rng(max_num_blocks, max_block_size)
            assert num_legs in [None, 2]
            data = backend.diagonal_from_block_func(block_rng, leg=leg, func_kwargs=dict(real=real))
            if isinstance(backend, backends.abelian.AbelianBackend):
                data = abelian_data_drop_some_blocks(data, np_random=np_random, empty_ok=empty_ok,
                                                     all_blocks=all_blocks)
            return tensors.DiagonalTensor(data, first_leg=leg, second_leg_dual=second_leg_dual,
                                          backend=backend, labels=labels)
        elif cls is tensors.ChargedTensor:
            sectors = symmetry_sectors_rng(1)
            dummy_leg = spaces.VectorSpace(symmetry=symmetry, sectors=sectors, multiplicities=[1])
            inv_part = generator(legs=legs + [dummy_leg], labels=labels + ['!'],
                                 max_num_blocks=max_num_blocks, max_block_size=max_block_size,
                                 real=real, empty_ok=empty_ok, all_blocks=all_blocks,
                                 cls=tensors.BlockDiagonalTensor)
            return tensors.ChargedTensor(inv_part)
        elif cls is tensors.Mask:
            assert len(legs) == 1
            if legs[0] is None:
                leg = vector_space_rng(max_num_blocks, max_block_size)
            else:
                leg = legs[0]
            leg: spaces.VectorSpace
            blockmask = np_random.choice([True, False], size=leg.dim)
            if np.all(blockmask):
                blockmask[len(blockmask) // 2] = False
            if not np.any(blockmask):
                blockmask[len(blockmask) // 2] = True
            if isinstance(backend, backends.abelian.AbelianBackend):
                # "drop" some blocks, i.e. set them to False
                if (not all_blocks) and (np_random.random() < .5):
                    drop = (np_random.random(leg.num_sectors) < 0.5)
                    if np.all(drop):  # keep at least one
                        drop[0:1] = False
                    for slc in leg.slices[drop]:
                        blockmask[leg.basis_perm[slice(*slc)]] = False
                if (not empty_ok) and (not np.any(blockmask)):
                    msg = 'Empty data was generated. If thats ok, suppress with `empty_ok=True`'
                    raise ValueError(msg)
            return tensors.Mask.from_blockmask(blockmask, large_leg=leg, backend=backend)
        elif cls is not tensors.BlockDiagonalTensor:
            raise ValueError(f'Illegal tensor cls: {cls}')

        # fill in missing legs
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
        
        data = backend.from_block_func(block_rng, legs, num_domain_legs=num_domain_legs,
                                       func_kwargs=dict(real=real))
        if isinstance(backend, backends.abelian.AbelianBackend):
            data = abelian_data_drop_some_blocks(data, np_random=np_random, empty_ok=empty_ok,
                                                 all_blocks=all_blocks)

        return tensors.BlockDiagonalTensor(data, backend=backend, legs=legs,
                                           num_domain_legs=num_domain_legs, labels=labels)
    return generator
