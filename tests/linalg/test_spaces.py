# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy import testing as npt

from tenpy.linalg import spaces, backends, symmetries


some_symmetries = dict(
    no_symmetry=symmetries.no_symmetry,
    z4=symmetries.z4_symmetry,
    z4_named=symmetries.ZNSymmetry(4, 'foo'),
    z4_z5=symmetries.z4_symmetry * symmetries.z5_symmetry,
    z4_z5_named=symmetries.ZNSymmetry(4, 'foo') * symmetries.ZNSymmetry(5, 'bar'),
    # su2=groups.su2_symmetry,  # TODO (JU) : reintroduce once n symbol is implemented
)


# TODO (JU) unsused?
def _get_four_sectors(symm: symmetries.Symmetry) -> symmetries.SectorArray:
    if isinstance(symm, symmetries.SU2Symmetry):
        res = np.arange(0, 8, 2, dtype=int)[:, None]
    elif symm.num_sectors >= 8:
        res = symm.all_sectors()[:8:2]
    elif symm.num_sectors >= 4:
        res = symm.all_sectors()[:4]
    else:
        res = np.tile(symm.all_sectors()[:, 0], 4)[:4, None]
    assert res.shape == (4, symm.sector_ind_len)
    return res


def test_vector_space(symmetry, symmetry_sectors_rng, np_random):
    sectors = symmetry_sectors_rng(10)
    sectors = sectors[np.lexsort(sectors.T)]
    mults = np_random.integers(1, 10, size=len(sectors))

    # TODO (JU) test real (as in "not complex") vectorspaces

    s1 = spaces.VectorSpace(symmetry=symmetry, sectors=sectors, multiplicities=mults)
    s2 = spaces.VectorSpace.from_trivial_sector(dim=8)

    print('checking VectorSpace.sectors')
    npt.assert_array_equal(s2.sectors, symmetries.no_symmetry.trivial_sector[None, :])
    npt.assert_array_equal(s1.dual.sectors, symmetry.dual_sectors(s1.sectors))

    print('checking str and repr')
    _ = str(s1)
    _ = str(s2)
    _ = repr(s1)
    _ = repr(s2)

    print('checking duality and equality')
    assert s1 == s1
    assert s1 != s1.dual
    assert s1 != s2
    wrong_mults = mults.copy()
    if len(mults) > 2:
        wrong_mults[-2] += 1
    else:
        wrong_mults[0] += 1
    assert s1 != spaces.VectorSpace(symmetry=symmetry, sectors=sectors, multiplicities=wrong_mults)
    assert s1.dual == spaces.VectorSpace(symmetry=symmetry, sectors=sectors, multiplicities=mults, _is_dual=True)
    assert s1.can_contract_with(s1.dual)
    assert not s1.can_contract_with(s1)
    assert not s1.can_contract_with(s2)

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert spaces.VectorSpace.from_trivial_sector(dim=1).is_trivial
    assert spaces.VectorSpace(symmetry=symmetry, sectors=symmetry.trivial_sector[np.newaxis, :]).is_trivial

    print('checking is_subspace_of')
    same_sectors_less_mults = spaces.VectorSpace(
        symmetry=symmetry, sectors=sectors, multiplicities=[max(1, m - 1) for m in mults]
    )
    same_sectors_different_mults = spaces.VectorSpace(
       symmetry=symmetry, sectors=sectors,
       multiplicities=[max(1, m + (+1 if i % 2 == 0 else -1)) for i, m in enumerate(mults)]
    )  # but at least one mult is larger than for s1
    if len(sectors) > 2:
        which1 = [0, -1]
        which2 = [1, -2]
    else:
        # if there are only two sectors, we cant have different sets of sectors,
        # both of which have multiple entries
        which1 = [0]
        which2 = [-1]
    fewer_sectors1 = spaces.VectorSpace(symmetry=symmetry, sectors=[sectors[i] for i in which1],
                                        multiplicities=[mults[i] for i in which1])
    fewer_sectors2 = spaces.VectorSpace(symmetry=symmetry, sectors=[sectors[i] for i in which2],
                                        multiplicities=[mults[i] for i in which2])
    assert s1.is_subspace_of(s1)
    assert not s1.dual.is_subspace_of(s1)
    assert same_sectors_less_mults.is_subspace_of(s1)
    assert not s1.is_subspace_of(same_sectors_less_mults)
    assert not same_sectors_different_mults.is_subspace_of(s1)
    assert len(sectors) == 1 or not s1.is_subspace_of(same_sectors_different_mults)
    assert fewer_sectors1.is_subspace_of(s1)
    if len(sectors) == 1:
        # if there is only one sector, the "fewer_sectors*" spaces dont actually have fewer sectors
        # and are both equal to s1
        assert s1.is_subspace_of(fewer_sectors1)
        assert fewer_sectors1.is_subspace_of(fewer_sectors2)
        assert fewer_sectors2.is_subspace_of(fewer_sectors1)
    else:
        assert not s1.is_subspace_of(fewer_sectors1)
        assert not fewer_sectors1.is_subspace_of(fewer_sectors2)
        assert not fewer_sectors2.is_subspace_of(fewer_sectors1)

    # TODO (JU) test num_parameters when ready

    print('check idx_to_sector and parse_idx')
    idx = 0
    for n, s in enumerate(s1.sectors):
        for m in range(s1.multiplicities[n]):
            sector_idx, mult_idx = s1.parse_index(idx)
            assert sector_idx == n
            assert mult_idx == m
            assert np.all(s1.idx_to_sector(idx) == s)
            idx += 1

    print('check sector lookup')
    for expect in [2, 3, 4]:
        expect = expect % s1.num_sectors
        assert s1.sectors_where(s1.sectors[expect]) == expect
        assert s1._non_dual_sectors_where(s1._non_dual_sectors[expect]) == expect
        assert s1.sector_multiplicity(s1.sectors[expect]) == s1.multiplicities[expect]

    print('check from_basis')
    if symmetry.num_sectors == 1:
        which_sectors = np.array([0] * 9)
        expect_basis_perm = np.arange(9)
        expect_sectors = sectors[:1]
    elif symmetry.num_sectors == 2:
        #                         0  1  2  3  4  5  6  7  8
        which_sectors = np.array([1, 0, 0, 1, 1, 0, 1, 1, 1])
        expect_basis_perm = np.array([1, 2, 5, 0, 3, 4, 6, 7, 8])
        expect_sectors = sectors[:2]
    else:
        assert len(np.unique(sectors[:3], axis=0)) == 3
        #                         0  1  2  3  4  5  6  7  8  9
        which_sectors = np.array([2, 0, 1, 2, 2, 2, 0, 0, 1, 2])
        expect_basis_perm = np.array([1, 6, 7, 2, 8, 0, 3, 4, 5, 9])
        expect_sectors = sectors[:3]
    expect_mults = np.sum(which_sectors[:, None] == np.arange(len(expect_sectors))[None, :], axis=0)
    sectors_of_basis = sectors[which_sectors]
    space = spaces.VectorSpace.from_basis(symmetry=symmetry, sectors_of_basis=sectors_of_basis)
    npt.assert_array_equal(space.sectors, expect_sectors)
    npt.assert_array_equal(space.multiplicities, expect_mults)
    npt.assert_array_equal(space.basis_perm, expect_basis_perm)
    # also check sectors_of_basis property
    npt.assert_array_equal(space.sectors_of_basis, sectors_of_basis)


def test_take_slice(vector_space_rng, np_random):
    space: spaces.VectorSpace = vector_space_rng()
    mask = np_random.choice([True, False], size=space.dim)
    small = space.take_slice(mask)
    #
    npt.assert_array_equal(small.sectors_of_basis, space.sectors_of_basis[mask])
    #
    internal_mask = mask[space.basis_perm]
    x = np.arange(space.dim)
    npt.assert_array_equal(x[mask][small.basis_perm], x[space.basis_perm][internal_mask])


def test_product_space(symmetry, symmetry_sectors_rng, np_random):
    sectors = symmetry_sectors_rng(10)
    mults = np_random.integers(1, 10, size=len(sectors))

    # TODO (JU) test real (as in "not complex") vectorspaces

    s1 = spaces.VectorSpace.from_sectors(symmetry=symmetry, sectors=sectors, multiplicities=mults)
    s2 = spaces.VectorSpace.from_sectors(symmetry=symmetry, sectors=sectors[:2], multiplicities=mults[:2])
    s3 = spaces.VectorSpace.from_sectors(symmetry=symmetry, sectors=sectors[::2], multiplicities=mults[::2])

    p1 = spaces.ProductSpace([s1, s2, s3])
    p2 = spaces.ProductSpace([s1, s2])
    p3 = spaces.ProductSpace([spaces.ProductSpace([s1, s2]), s3])

    npt.assert_array_equal(p1.sectors, p3.sectors)

    _ = str(p1)
    _ = str(p3)
    _ = repr(p1)
    _ = repr(p3)

    assert p1 == p1
    assert p1 != s1
    assert s1 != p1
    assert p1 != p3
    assert p2 == spaces.ProductSpace([s1.dual, s2.dual], _is_dual=True).dual
    for p in [p1, p2, p3]:
        assert p.can_contract_with(p.dual)
    assert p2 == spaces.ProductSpace([s1, s2], _is_dual=True).flip_is_dual()
    assert p2.can_contract_with(spaces.ProductSpace([s1.dual, s2.dual], _is_dual=False).flip_is_dual())
    assert p2.can_contract_with(spaces.ProductSpace([s1.dual, s2.dual]))  # check default _is_dual
    assert p2.can_contract_with(spaces.ProductSpace([s1.dual, s2.dual], _is_dual=True))
    p1_s = p1.as_VectorSpace()
    assert isinstance(p1_s, spaces.VectorSpace)
    assert np.all(p1_s.sectors == p1.sectors)
    assert np.all(p1_s.multiplicities == p1.multiplicities)
    assert not p1_s.is_equal_or_dual(p1)
    assert p1_s != p1
    assert p1 != p1_s


def test_get_basis_transformation(default_backend):
    # TODO expand this
    even, odd = [0], [1]
    spin1 = spaces.VectorSpace.from_basis(symmetries.z2_symmetry, [even, odd, even])
    assert np.array_equal(spin1.sectors, [even, odd])
    assert np.array_equal(spin1.basis_perm, [0, 2, 1])
    backend = backends.get_backend(block_backend='numpy', symmetry='abelian')
    product_space = spaces.ProductSpace([spin1, spin1], backend=backend)

    perm = product_space._get_fusion_outcomes_perm()
    # internal order of spin1: + - 0
    # internal uncoupled:  ++  +-  +0  -+  --  -0  0+  0-  00
    # coupled:  ++  +-  -+  --  00  +0  -0  0+  0-
    expect = np.array([0, 1, 3, 4, 8, 2, 5, 6, 7])
    print(perm)
    assert np.all(perm == expect)

    perm = product_space.get_basis_transformation_perm()
    # public order of spin1 : + 0 -
    # public coubpled : ++  +0  +-  0+  00  0-  -+  -0  --
    # coupled:  ++  +-  -+  --  00  +0  -0  0+  0-
    expect = np.array([0, 2, 6, 8, 4, 1, 7, 3, 5])
    print(perm)
    assert np.all(perm == expect)


# TODO systematically test VectorSpace class methods


def test_direct_sum(vector_space_rng, max_block_size=5, max_num_blocks=5):
    a = vector_space_rng(max_block_size=max_block_size, max_num_blocks=max_num_blocks)
    b = vector_space_rng(max_block_size=max_block_size, max_num_blocks=max_num_blocks,
                         is_dual=a.is_dual)
    c = vector_space_rng(max_block_size=max_block_size, max_num_blocks=max_num_blocks,
                         is_dual=a.is_dual)
    assert a == spaces.VectorSpace.direct_sum(a)
    d = a.direct_sum(b, c)
    expect = np.concatenate([leg.sectors_of_basis for leg in [a, b, c]], axis=0)
    npt.assert_array_equal(d.sectors_of_basis, expect)
    

def all_str_repr_demos():
    # python -c "import test_spaces; test_spaces.all_str_repr_demos()"
    print()
    print('----------------------')
    print('VectorSpace.__repr__()')
    print('----------------------')
    demo_VectorSpace_repr(repr)
    
    print()
    print('---------------------')
    print('VectorSpace.__str__()')
    print('---------------------')
    demo_VectorSpace_repr(str)
    
    print()
    print('-----------------------')
    print('ProductSpace.__repr__()')
    print('-----------------------')
    demo_ProductSpace_repr(repr)
    
    print()
    print('----------------------')
    print('ProductSpace.__str__()')
    print('----------------------')
    demo_ProductSpace_repr(str)


def demo_VectorSpace_repr(fun=repr):
    from tests import conftest
    for symmetry in conftest.symmetry._pytestfixturefunction.params:
        space = conftest.random_vector_space(symmetry, max_num_blocks=20)
        print()
        print(fun(space))


def demo_ProductSpace_repr(fun=repr):
    from tests import conftest
    for symmetry in conftest.symmetry._pytestfixturefunction.params:
        num = 1 + np.random.choice(3)
        is_dual = np.random.choice([True, False, None])
        spaces_ = [conftest.random_vector_space(symmetry) for _ in range(num)]
        space = spaces.ProductSpace(spaces_, _is_dual=is_dual)
        print()
        print(fun(space))
