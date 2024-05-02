"""Provide test configuration for backends etc.

TODO format this summary of fixtures:

The following table summarizes the available fixtures.
There are three groups; miscellaneous independent fixtures, unconstrained fixtures
(with ``any`` in their name) and constrained fixtures (with ``compatible`` in their name).
The latter two groups are similar in terms of signature.
The unconstrained fixtures are intended if a test should be parametrized over possible
symmetry backends *or* over possible symmetries, but not both.
The constrained ("compatible") fixtures are intended if a test should be parametrized over
possible combinations of a symmetry backend *and* a symmetry it is compatible with.
They should not be mixed in any single test, as that would generate unnecessarily many tests.
Whenever applicable, the unconstrained fixtures should be preferred, since e.g. most symmetries
appear multiple times as values of ``compatible_symmetry`` (same argument for ``compatible_backend``).


=============================  ======================  ===========================================
Fixture                        Depends on / # cases    Description
=============================  ======================  ===========================================
np_random                      -                       A numpy random Generator. Use this for
                                                       reproducibility.
-----------------------------  ----------------------  -------------------------------------------
block_backend                  Generates ~2 cases      Goes over all block backends, as str
                                                       descriptions, valid for ``get_backend``.
=============================  ======================  ===========================================
any_symmetry_backend           Generates 3 cases       Goes over all symmetry backends, as str
                                                       descriptions, valid for ``get_backend``.
-----------------------------  ----------------------  -------------------------------------------
any_backend                    block_backend           Goes over all backends.
                               any_symmetry_backend
-----------------------------  ----------------------  -------------------------------------------
any_symmetry                   Generates ~10 cases     Goes over some representative symmetries.
-----------------------------  ----------------------  -------------------------------------------
make_any_sectors               any_symmetry            RNG for sectors with ``any_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_any_space                 any_symmetry            RNG for spaces with ``any_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_any_block                 any_backend             RNG for blocks of ``any_backend``.
                                                       ``make(size, real=False)``
=============================  ======================  ===========================================
compatible_pairs               Generates ~20 cases     Not an public fixture, only generates
                                                       the cases. Compatible pairs are built like
                                                       combinations of ``any_symmetry_backend``
                                                       and ``any_symmetry``, constrained by
                                                       compatibility.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry_backend    compatible_pairs        The symmetry backend of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry            compatible_pairs        The symmetry of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_backend             compatible_pairs        A backend that is compatible with
                               block_backend           ``compatible_symmetry``.
-----------------------------  ----------------------  -------------------------------------------
make_compatible_sectors        compatible_pairs        RNG for sectors with ``compatible_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_space          compatible_pairs        RNG for spaces with ``compatible_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_block          compatible_backend      RNG for blocks with ``compatible_backend``.
                                                       ``make(size, real=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_tensor         compatible_backend      RNG for tensors with ``compatible_backend``.
=============================  ======================  ===========================================

The signature for ``make_compatible_tensor`` is
``make(legs=None, num_legs=None, labels=None, max_blocks=5, max_block_size=5, real=False,
       empty_ok=False, all_blocks=False, cls=tensors.BlockDiagonalTensor, num_domain_legs=0)``

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import pytest
from typing import Callable

from tenpy.linalg import backends, spaces, symmetries, tensors


# QUICK CONFIGURATION

_block_backends = ['numpy']  # TODO reactivate 'torch'
_symmetries = {
    'NoSymm': symmetries.no_symmetry,
    'U1': symmetries.u1_symmetry,
    'Z2': symmetries.z2_symmetry,
    'Z4_named': symmetries.ZNSymmetry(4, "My_Z4_symmetry"),
    'U1xZ3': symmetries.ProductSymmetry([symmetries.u1_symmetry, symmetries.z3_symmetry]),
    'SU2': symmetries.SU2Symmetry(),
}


# "UNCONSTRAINED" FIXTURES  ->  independent (mostly) of the other features. no compatibility guarantees.

@pytest.fixture
def np_random() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture(params=_block_backends)
def block_backend(request) -> str:
    if request.param == 'torch':
        torch = pytest.importorskip('torch', reason='torch not installed')
    return request.param


@pytest.fixture(params=['no_symmetry', 'abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def any_symmetry_backend(request) -> str:
    return request.param


@pytest.fixture
def any_backend(block_backend, any_symmetry_backend) -> backends.Backend:
    return backends.backend_factory.get_backend(any_symmetry_backend, block_backend)


@pytest.fixture(params=list(_symmetries.values()), ids=list(_symmetries.keys()))
def any_symmetry(request) -> symmetries.Symmetry:
    return request.param


@pytest.fixture
def make_any_sectors(any_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> symmetries.SectorArray:
        # return SectorArray
        return random_symmetry_sectors(any_symmetry, num, sort, np_random=np_random)
    return make


@pytest.fixture
def make_any_space(any_symmetry, np_random):
    def make(max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None) -> spaces.VectorSpace:
        # return VectorSpace
        return random_vector_space(any_symmetry, max_sectors, max_mult, is_dual, np_random=np_random)
    return make


@pytest.fixture
def make_any_block(any_backend, np_random):
    def make(size: tuple[int, ...], real=False) -> backends.Block:
        # return Block
        return random_block(any_backend, size, real=real, np_random=np_random)
    return make


# "COMPATIBLE" FIXTURES  ->  only go over those pairings of backend and symmetry that are compatible

# build the compatible pairs
_compatible_pairs = {'NoSymmetry': ('no_symmetry', symmetries.no_symmetry)}  # {id: param}
for _sym_name, _sym in _symmetries.items():
    if isinstance(_sym, symmetries.AbelianGroup):
        _compatible_pairs[f'AbelianBackend-{_sym_name}'] = ('abelian', _sym)
    _compatible_pairs[f'FusionTreeBackend-{_sym_name}'] = pytest.param(
        ('fusion_tree', _sym), marks=pytest.mark.FusionTree
    )

@pytest.fixture(params=list(_compatible_pairs.values()), ids=list(_compatible_pairs.keys()))
def _compatible_backend_symm_pairs(request) -> tuple[str, symmetries.Symmetry]:
    """Helper fixture that allows us to generate the *compatible* fixtures.

    Values are pairs (symmetry_backend: str, symmetry: Symmetry)
    """
    return request.param


@pytest.fixture
def compatible_symmetry_backend(_compatible_backend_symm_pairs) -> str:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry_backend


@pytest.fixture
def compatible_backend(compatible_symmetry_backend, block_backend) -> backends.Backend:
    return backends.backend_factory.get_backend(compatible_symmetry_backend, block_backend)


@pytest.fixture
def compatible_symmetry(_compatible_backend_symm_pairs) -> symmetries.Symmetry:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry


@pytest.fixture
def make_compatible_sectors(compatible_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> symmetries.SectorArray:
        # returns SectorArray
        return random_symmetry_sectors(compatible_symmetry, num, sort, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_space(compatible_symmetry, np_random):
    def make(max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None) -> spaces.VectorSpace:
        # returns VectorSpace
        return random_vector_space(compatible_symmetry, max_sectors, max_mult, is_dual, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_block(compatible_backend, np_random):
    def make(size: tuple[int, ...], real: bool = False) -> backends.Block:
        # returns Block
        return random_block(compatible_backend, size, real=real, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_tensor(compatible_backend, compatible_symmetry, make_compatible_block,
                           make_compatible_space, np_random):
    """Tensor RNG.

    legs may contain any or all ``None`` entries.
    Those will be filled randomly, but tuned such that the result can have free parameters.
    """
    def make(legs=None, num_legs=None, labels=None,
             max_blocks=5, max_block_size=5, real=False, empty_ok=False, all_blocks=False,
             cls=tensors.BlockDiagonalTensor, num_domain_legs=0):
        # return tensor of type cls
        
        # deal with tensor classes with constrained legs first
        if cls is tensors.DiagonalTensor:
            if legs is None:
                legs = [None, None]
            assert len(legs) == 2
            if legs[0] is not None:
                leg = legs[0]
                assert legs[1] is None or legs[0].can_contract_with(legs[1])
            elif legs[1] is not None:
                leg = legs[1].dual
            else:
                leg = make_compatible_space(max_sectors=max_blocks, max_mult=max_block_size)
            res = tensors.DiagonalTensor.from_block_func(
                make_compatible_block, leg, backend=compatible_backend, func_kwargs=dict(real=real)
            )
            if not all_blocks:
                res = randomly_drop_blocks(res, max_blocks=max_blocks, empty_ok=empty_ok,
                                        np_random=np_random)
            res.test_sanity()
            return res
        
        if cls is tensors.Mask:
            if legs is None:
                legs = [None, None]
            assert len(legs) == 2
            if legs[0] is None and legs[1] is None:
                large_leg = make_compatible_space(max_sectors=max_blocks, max_mult=max_block_size)
                small_leg = None
            elif legs[1] is None:
                large_leg = legs[0].dual
                small_leg = None
            elif legs[0] is None:
                raise NotImplementedError  # TODO need to generate a larger leg that "contains" legs[1]
            else:
                raise NotImplementedError  # TODO need to generate random mask that *fits* legs[1]
            blockmask = np_random.choice([True, False], large_leg.dim)
            res = tensors.Mask.from_blockmask(blockmask, large_leg, compatible_backend, labels)
            res.test_sanity()
            return res

        # parse legs
        if legs is None:
            if num_legs is None and labels is None:
                raise ValueError('Need to specify number of legs via ``legs``, ``num_legs`` or ``labels``')
            elif num_legs is None:
                num_legs = len(labels)
            elif labels is None:
                labels = [None] * num_legs
            else:
                assert num_legs == len(labels)
            legs = [None] * num_legs
        else:
            if num_legs is None:
                num_legs = len(legs)
            assert num_legs == len(legs)
            if labels is None:
                labels = [None] * num_legs
            assert len(labels) == num_legs
        
        # fill in missing legs
        missing_leg_pos = list(np_random.permuted([i for i, l in enumerate(legs) if l is None]))
        while len(missing_leg_pos) > 1:
            which = missing_leg_pos.pop()
            legs[which] = make_compatible_space(max_sectors=max_blocks, max_mult=max_block_size)
        if len(missing_leg_pos) > 0:
            which, = missing_leg_pos
            if len(legs) == 1:
                new_leg = make_compatible_space(max_sectors=max_blocks, max_mult=max_block_size)
                # make sure leg has the trivial space, so we can allow some blocks
                if new_leg.sector_multiplicity(compatible_symmetry.trivial_sector) == 0:
                    sectors = new_leg._non_dual_sectors
                    where = np_random.choice(len(sectors))
                    sectors[where] = compatible_symmetry.trivial_sector
                    # have potentially replaced higher-dimensional sectors with one-dimensional trivial sectors
                    # this would reduce dim and make basis_perm invalid.
                    # correct for that by increasing the multiplicities of the trivial sectors.
                    mults = new_leg.multiplicities
                    mults[where] *= compatible_symmetry.sector_dim(sectors[where])  
                    new_leg = spaces.VectorSpace.from_sectors(
                        new_leg.symmetry, sectors, mults, new_leg.basis_perm
                    )
            else:
                new_leg = find_compatible_leg(legs[:which] + legs[which + 1:],
                                              max_sectors=max_blocks, max_mult=max_block_size)
            legs[which] = new_leg
        
        if cls is tensors.BlockDiagonalTensor:
            res = tensors.BlockDiagonalTensor.from_block_func(
                make_compatible_block, legs, compatible_backend, labels,
                func_kwargs=dict(real=real), num_domain_legs=num_domain_legs
            )
            if not all_blocks:
                res = randomly_drop_blocks(res, max_blocks, empty_ok=empty_ok, np_random=np_random)
            res.test_sanity()
            return res
        
        if cls is tensors.ChargedTensor:
            dummy_leg = make_compatible_space(max_sectors=1, max_mult=1, is_dual=False)
            res = tensors.ChargedTensor.from_block_func(
                make_compatible_block, legs, dummy_leg, compatible_backend, labels,
                func_kwargs=dict(real=real), num_domain_legs=num_domain_legs
            )
            if not all_blocks:
                res.invariant_part = randomly_drop_blocks(res.invariant_part, max_blocks,
                                                          empty_ok=empty_ok, np_random=np_random)
            res.test_sanity()
            return res
        raise ValueError(f'Invalid tensor cls: {cls}')
    return make

# RANDOM GENERATION

def random_block(backend, size, real=False, np_random=np.random.default_rng(0)):
    block = np_random.normal(size=size)
    if not real:
        block = block + 1.j * np_random.normal(size=size)
    return backend.block_from_numpy(block)


def random_symmetry_sectors(symmetry: symmetries.Symmetry, num: int, sort: bool = False,
                            np_random=np.random.default_rng()) -> symmetries.SectorArray:
    """random unique symmetry sectors, optionally sorted"""
    if isinstance(symmetry, symmetries.SU2Symmetry):
        res = np_random.choice(int(1.3 * num), replace=False, size=(num, 1))
    elif isinstance(symmetry, symmetries.U1Symmetry):
        vals = list(range(-num, num)) + [123]
        res = np_random.choice(vals, replace=False, size=(num, 1))
    elif symmetry.num_sectors < np.inf:
        if symmetry.num_sectors <= num:
            res = np_random.permutation(symmetry.all_sectors())
        else:
            which = np_random.choice(symmetry.num_sectors, replace=False, size=num)
            res = symmetry.all_sectors()[which, :]
    elif isinstance(symmetry, symmetries.ProductSymmetry):
        factor_len = max(3, num // len(symmetry.factors))
        factor_sectors = [random_symmetry_sectors(factor, factor_len, np_random=np_random)
                          for factor in symmetry.factors]
        combs = np.indices([len(s) for s in factor_sectors]).T.reshape((-1, len(factor_sectors)))
        if len(combs) > num:
            combs = np_random.choice(combs, replace=False, size=num)
        res = np.hstack([fs[i] for fs, i in zip(factor_sectors, combs.T)])
    else:
        pytest.skip("don't know how to get symmetry sectors")  # raises Skipped
    if sort:
        order = np.lexsort(res.T)
        res = res[order]
    return res


def random_vector_space(symmetry, max_num_blocks=5, max_block_size=5, is_dual=None, np_random=None):
    if np_random is None:
        np_random = np.random.default_rng()
    num_sectors = np_random.integers(1, max_num_blocks, endpoint=True)
    sectors = random_symmetry_sectors(symmetry, num_sectors, sort=True, np_random=np_random)
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


def randomly_drop_blocks(res: tensors.BlockDiagonalTensor | tensors.DiagonalTensor,
                         max_blocks: int | None, empty_ok: bool, np_random=np.random.default_rng()):
    
    if isinstance(res.backend, backends.NoSymmetryBackend):
        # nothing to do
        return res

    num_blocks = len(res.data.blocks)
    min_blocks = 0 if empty_ok else 1
    if max_blocks is None:
        max_blocks = num_blocks
    else:
        max_blocks = min(num_blocks, max_blocks)
    if max_blocks < min_blocks:
        return res
    
    if np_random.uniform() < 0.5:
        # with 50% chance, keep maximum number
        num_keep = max_blocks
    else:
        num_keep = np_random.integers(min_blocks, max_blocks, endpoint=True)
    if num_keep == num_blocks:
        return res
    which = np_random.choice(num_blocks, size=num_keep, replace=False, shuffle=False)
    which = np.sort(which)

    if isinstance(res.backend, backends.AbelianBackend):
        res.data = backends.AbelianBackendData(
            dtype=res.dtype,
            blocks=[res.data.blocks[n] for n in which],
            block_inds=res.data.block_inds[which],
            is_sorted=True
        )
        return res

    if isinstance(res.backend, backends.FusionTreeBackend):
        res.data = backends.FusionTreeData(
            coupled_sectors=res.data.coupled_sectors[which, :],
            blocks=[res.data.blocks[n] for n in which],
            domain=res.data.domain, codomain=res.data.codomain, dtype=res.data.dtype
        )
        return res

    raise ValueError('Backend not recognized')


def find_compatible_leg(others, max_sectors: int, max_mult: int, extra_sectors=None,
                        np_random=np.random.default_rng()):
    """Find a leg such that ``[*others, new_leg]`` allows non-zero tensors."""
    prod = spaces.ProductSpace(others).as_VectorSpace()
    sectors = prod.symmetry.dual_sectors(prod.sectors)
    mults = prod.multiplicities
    if len(sectors) > max_sectors:
        which = np_random.choice(len(sectors), size=max_sectors, replace=False, shuffle=False)
        sectors = sectors[which, :]
        mults = mults[which]
    mults = np.minimum(mults, max_mult)
    if extra_sectors is not None:
        # replace some sectors by extra_sectors
        duplicates = np.any(np.all(extra_sectors[None, :, :] == sectors[:, None, :], axis=2), axis=0)
        extra_sectors = extra_sectors[np.logical_not(duplicates)]
        # replace some sectors
        min_replace = max(1, int(.2 * len(sectors)))
        max_replace = min(int(.5 * len(sectors)), len(extra_sectors))
        if max_replace >= min_replace:
            num_replace = np_random.integers(min_replace, max_replace, endpoint=True)
            which = np_random.choice(len(sectors), size=num_replace, replace=False)
            sectors[which, :] = extra_sectors[:num_replace, :]
    # guarantee sorting
    order = np.lexsort(sectors.T)
    sectors = sectors[order]
    mults = mults[order]
    
    res = spaces.VectorSpace(prod.symmetry, sectors, mults)

    # check that it actually worked
    assert spaces.ProductSpace([*others, res]).sector_multiplicity(prod.symmetry.trivial_sector) > 0
    res.test_sanity()

    return res
