# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy import testing as npt

from tenpy.linalg import spaces, backends, symmetries


def test_vector_space(any_symmetry, make_any_sectors, np_random):
    sectors = make_any_sectors(10)
    sectors = sectors[np.lexsort(sectors.T)]
    mults = np_random.integers(1, 10, size=len(sectors))

    # TODO (JU) test real (as in "not complex") vectorspaces

    s1 = spaces.VectorSpace(symmetry=any_symmetry, sectors=sectors, multiplicities=mults)
    s2 = spaces.VectorSpace.from_trivial_sector(dim=8)

    print('checking VectorSpace.sectors')
    npt.assert_array_equal(s2.sectors, symmetries.no_symmetry.trivial_sector[None, :])
    npt.assert_array_equal(s1.dual.sectors, any_symmetry.dual_sectors(s1.sectors))

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
    assert s1 != spaces.VectorSpace(symmetry=any_symmetry, sectors=sectors, multiplicities=wrong_mults)
    assert s1.dual == spaces.VectorSpace(symmetry=any_symmetry, sectors=sectors, multiplicities=mults, _is_dual=True)
    assert s1.can_contract_with(s1.dual)
    assert not s1.can_contract_with(s1)
    assert not s1.can_contract_with(s2)

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert spaces.VectorSpace.from_trivial_sector(dim=1).is_trivial
    assert spaces.VectorSpace(symmetry=any_symmetry, sectors=any_symmetry.trivial_sector[np.newaxis, :]).is_trivial

    print('checking is_subspace_of')
    same_sectors_less_mults = spaces.VectorSpace(
        symmetry=any_symmetry, sectors=sectors, multiplicities=[max(1, m - 1) for m in mults]
    )
    same_sectors_different_mults = spaces.VectorSpace(
       symmetry=any_symmetry, sectors=sectors,
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
    fewer_sectors1 = spaces.VectorSpace(symmetry=any_symmetry, sectors=[sectors[i] for i in which1],
                                        multiplicities=[mults[i] for i in which1])
    fewer_sectors2 = spaces.VectorSpace(symmetry=any_symmetry, sectors=[sectors[i] for i in which2],
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

    if not any_symmetry.is_abelian:
        # TODO
        pytest.xfail('parse_idx (and probly more methods) are wrong if any sector_dim > 1.')

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
    if any_symmetry.num_sectors == 1:
        which_sectors = np.array([0] * 9)
        expect_basis_perm = np.arange(9)
        expect_sectors = sectors[:1]
    elif any_symmetry.num_sectors == 2:
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
    space = spaces.VectorSpace.from_basis(symmetry=any_symmetry, sectors_of_basis=sectors_of_basis)
    npt.assert_array_equal(space.sectors, expect_sectors)
    npt.assert_array_equal(space.multiplicities, expect_mults)
    npt.assert_array_equal(space.basis_perm, expect_basis_perm)
    # also check sectors_of_basis property
    npt.assert_array_equal(space.sectors_of_basis, sectors_of_basis)


def test_take_slice(make_any_space, np_random):
    space: spaces.VectorSpace = make_any_space()
    mask = np_random.choice([True, False], size=space.dim)
    small = space.take_slice(mask)

    if not space.symmetry.is_abelian:
        # TODO
        pytest.xfail('sectors_of_basis not implemented')
        
    #
    npt.assert_array_equal(small.sectors_of_basis, space.sectors_of_basis[mask])
    #
    internal_mask = mask[space.basis_perm]
    x = np.arange(space.dim)
    npt.assert_array_equal(x[mask][small.basis_perm], x[space.basis_perm][internal_mask])


def test_product_space(any_symmetry, make_any_sectors, np_random):
    sectors = make_any_sectors(10)  # note: may be fewer, if symmetry doesnt have enough
    mults = np_random.integers(1, 10, size=len(sectors))

    # TODO (JU) test real (as in "not complex") vectorspaces

    s1 = spaces.VectorSpace.from_sectors(symmetry=any_symmetry, sectors=sectors, multiplicities=mults)
    s2 = spaces.VectorSpace.from_sectors(symmetry=any_symmetry, sectors=sectors[:2], multiplicities=mults[:2])
    s3 = spaces.VectorSpace.from_sectors(symmetry=any_symmetry, sectors=sectors[::2], multiplicities=mults[::2])

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

    # check empty product
    empty_product = spaces.ProductSpace([], symmetry=any_symmetry, is_real=s1.is_real)
    monoidal_unit = spaces.VectorSpace.from_trivial_sector(1, any_symmetry)
    _ = str(empty_product)
    _ = repr(empty_product)
    assert empty_product != p1
    assert empty_product != monoidal_unit
    assert empty_product.as_VectorSpace() == monoidal_unit


def test_ProductSpace_SU2():
    sym = symmetries.SU2Symmetry()
    a = spaces.VectorSpace(sym, [[0], [3], [2]], [2, 3, 4])
    b = spaces.VectorSpace(sym, [[1], [4]], [5, 6])
    c = spaces.VectorSpace(sym, [[0], [3], [1]], [3, 1, 2])

    ab = spaces.ProductSpace([a, b])
    # a     b           fusion
    #                   0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2            10
    # 0     2                          12
    # 3/2   1/2                15      15
    # 3/2   2              18      18      18      18
    # 1     1/2            20      20
    # 1     2                  24      24      24
    # COUNTS
    npt.assert_array_equal(ab.sectors, np.array([1, 2, 3, 4, 5, 6, 7])[:, None])
    npt.assert_array_equal(ab.multiplicities, np.array([48, 39, 38, 51, 18, 24, 18]))

    bc = spaces.ProductSpace([b, c])
    # c     b      mult     fusion
    #                       0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2    3*5         15
    # 0     2      3*6                     18
    # 3/2   1/2    1*5              5       5
    # 3/2   2      1*6          6       6       6       6
    # 1/2   1/2    2*5     10      10
    # 1/2   2      2*6                 12      12
    # COUNTS
    npt.assert_array_equal(bc.sectors, np.array([0, 1, 2, 3, 4, 5, 7])[:, None])
    npt.assert_array_equal(bc.multiplicities, np.array([10, 21, 15, 18, 23, 18, 6]))

    abc = spaces.ProductSpace([a, b, c])
    # ab    c   mult    fusion
    #                   0   1/2   1   3/2   2   5/2   3   7/2   4   9/2   5
    # 1/2   0   48*3        144
    # 1/2 3/2   48*1             48        48
    # 1/2 1/2   48*2   96        96
    # 1     0   39*3            117
    # 1   3/2   39*1         39        39        39
    # 1   1/2   39*2         78        78
    # 3/2   0   38*3                  114 
    # 3/2 3/2   38*1   38        38        38        38
    # 3/2 1/2   38*2             76        76
    # 2     0   51*3                      153
    # 2   3/2   51*1         51        51        51        51
    # 2   1/2   51*2                  102       102
    # 5/2   0   18*3                             54
    # 5/2 3/2   18*1             18        18        18        18
    # 5/2 1/2   18*2                       36        36
    # 3     0   24*3                                 72
    # 3   3/2   24*1                   24        24        24       24
    # 3   1/2   24*2                             48        48
    # 7/2   0   18*3                                       54
    # 7/2 3/2   18*1                       18        18        18        18
    # 7/2 1/2   18*2                                 36        36
    expect_mults = [96+38, 144+39+78+51, 48+96+117+38+76+18, 39+78+114+51+102+24,
                    48+38+76+153+18+36+18, 39+51+102+54+24+48, 38+18+36+72+18+36, 51+24+48+54,
                    18+18+36, 24, 18]
    npt.assert_array_equal(abc.sectors, np.arange(11)[:, None])
    npt.assert_array_equal(abc.multiplicities, np.array(expect_mults))


def test_get_basis_transformation():
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
    # public coupled : ++  +0  +-  0+  00  0-  -+  -0  --
    # coupled:  ++  +-  -+  --  00  +0  -0  0+  0-
    expect = np.array([0, 2, 6, 8, 4, 1, 7, 3, 5])
    print(perm)
    assert np.all(perm == expect)


# TODO systematically test VectorSpace class methods


def test_direct_sum(make_any_space, max_mult=5, max_sectors=5):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=a.is_dual)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=a.is_dual)
    assert a == spaces.VectorSpace.direct_sum(a)
    d = a.direct_sum(b, c)

    if not a.symmetry.is_abelian:
        # TODO
        pytest.xfail('sectors_of_basis not implemented')
    
    expect = np.concatenate([leg.sectors_of_basis for leg in [a, b, c]], axis=0)
    npt.assert_array_equal(d.sectors_of_basis, expect)


def test_str_repr(make_any_space, str_max_lines=20, repr_max_lines=20):
    """Check if str and repr work. Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run ``pytest -rP -k test_str_repr`` to see the output.
    """
    terminal_width = 80
    str_max_len = terminal_width * str_max_lines
    repr_max_len = terminal_width * str_max_lines
    # TODO output is a bit long, should we force shorter? -> consider config.printoptions!
    
    space = make_any_space(max_sectors=20)

    print('----------------------')
    print('VectorSpace.__repr__()')
    print('----------------------')
    res = repr(space)
    assert len(res) <= repr_max_len
    assert res.count('\n') <= repr_max_lines
    print(res)
    
    print()
    print()
    print('----------------------')
    print('VectorSpace.__str__() ')
    print('----------------------')
    res = str(space)
    assert len(res) <= str_max_len
    assert res.count('\n') <= str_max_lines
    print(res)

    product_space = spaces.ProductSpace([make_any_space(max_sectors=3)])
    while len(product_space.spaces) <= 3:
        product_space = spaces.ProductSpace([*product_space.spaces, make_any_space(max_sectors=3)])
        print()
        print()
        print('-----------------------')
        print(f'ProductSpace.__repr__()  {len(product_space.spaces)} spaces')
        print('-----------------------')
        res = repr(product_space)
        assert len(res) <= repr_max_len
        assert res.count('\n') <= repr_max_lines
        print(res)
        
        print()
        print()
        print('-----------------------')
        print(f'ProductSpace.__str__()  {len(product_space.spaces)} spaces')
        print('-----------------------')
        res = str(product_space)
        assert len(res) <= str_max_len
        assert res.count('\n') <= str_max_lines
        print(res)
