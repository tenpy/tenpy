# Copyright (C) TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy import testing as npt

from tenpy.linalg import spaces, backends, symmetries, SymmetryError


def test_vector_space(any_symmetry, make_any_sectors, np_random):
    sectors = make_any_sectors(10)
    sectors = sectors[np.lexsort(sectors.T)]
    dual_sectors = any_symmetry.dual_sectors(sectors)
    dual_sectors_sort = np.lexsort(dual_sectors.T)
    mults = np_random.integers(1, 10, size=len(sectors))

    # TODO (JU) test real (as in "not complex") vectorspaces

    s1 = spaces.ElementarySpace(symmetry=any_symmetry, sectors=sectors, multiplicities=mults)
    s2 = spaces.ElementarySpace.from_trivial_sector(dim=8)

    print('checking ElementarySpace.sectors')
    npt.assert_array_equal(s2.sectors, symmetries.no_symmetry.trivial_sector[None, :])

    print('checking str and repr')
    _ = str(s1)
    _ = str(s2)
    _ = repr(s1)
    _ = repr(s2)

    print('checking duality and equality')
    assert_spaces_equal(s1, s1)
    s1_dual = s1.dual
    assert s1 != s1_dual
    assert s1 != s2
    wrong_mults = mults.copy()
    if len(mults) > 2:
        wrong_mults[-2] += 1
    else:
        wrong_mults[0] += 1
    assert s1 != spaces.ElementarySpace(symmetry=any_symmetry, sectors=sectors, multiplicities=wrong_mults)
    npt.assert_array_equal(s1_dual.sectors, dual_sectors[dual_sectors_sort])
    npt.assert_array_equal(s1_dual.multiplicities, s1.multiplicities[dual_sectors_sort])
    assert s1_dual.symmetry == s1.symmetry
    assert s1_dual.is_dual is True
    #
    s1_modified = spaces.ElementarySpace(s1.symmetry, sectors=s1.sectors, multiplicities=s1.multiplicities,
                                         is_dual=not s1.is_dual, basis_perm=s1._basis_perm)
    assert s1 != s1_modified
    assert s1_modified == s1.with_opposite_duality()

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert spaces.ElementarySpace.from_trivial_sector(dim=1).is_trivial
    assert spaces.ElementarySpace(symmetry=any_symmetry, sectors=any_symmetry.trivial_sector[np.newaxis, :]).is_trivial

    print('checking is_subspace_of')
    same_sectors_less_mults = spaces.ElementarySpace(
        symmetry=any_symmetry, sectors=sectors, multiplicities=[max(1, m - 1) for m in mults]
    )
    same_sectors_different_mults = spaces.ElementarySpace(
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
    fewer_sectors1 = spaces.ElementarySpace(symmetry=any_symmetry, sectors=[sectors[i] for i in which1],
                                            multiplicities=[mults[i] for i in which1])
    fewer_sectors2 = spaces.ElementarySpace(symmetry=any_symmetry, sectors=[sectors[i] for i in which2],
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
    if any_symmetry.can_be_dropped:
        idx = 0  # will step this up during the loop
        for n_sector, sector in enumerate(s1.sectors):
            d = any_symmetry.sector_dim(sector)
            for m in range(s1.multiplicities[n_sector]):
                for mu in range(d):
                    sector_idx, mult_idx = s1.parse_index(idx)
                    assert sector_idx == n_sector
                    assert mult_idx == m * d + mu
                    assert np.all(s1.idx_to_sector(idx) == sector)
                    idx += 1

    print('check sector lookup')
    for expect in [2, 3, 4]:
        expect = expect % s1.num_sectors
        assert s1.sectors_where(s1.sectors[expect]) == expect
        assert s1.sectors_where(s1.sectors[expect]) == expect
        assert s1.sector_multiplicity(s1.sectors[expect]) == s1.multiplicities[expect]

    print('check from_basis')
    if any_symmetry.can_be_dropped:
        if isinstance(any_symmetry, symmetries.SU2Symmetry):
            with pytest.raises(ValueError, match='Sectors must appear in whole multiplets'):
                bad_sectors = np.array([0, 1, 1, 1, 2, 2, 2])[:, None]
                # have three basis vectors for 2-dimensional spin-1/2
                _ = spaces.ElementarySpace.from_basis(symmetry=any_symmetry, sectors_of_basis=bad_sectors)
            
            # spins 0, 1/2 and 1, each two times
            #                         0  1  2  3  4  5  6  7  8  9  10 11
            sectors_of_basis = np.array([0, 2, 2, 1, 2, 1, 2, 2, 0, 2, 1, 1])[:, None]
            expect_basis_perm = np.array([0, 8, 3, 5, 10, 11, 1, 2, 4, 6, 7, 9])
            expect_sectors = np.array([0, 1, 2])[:, None]
            expect_mults = np.array([2, 2, 2])
        else:
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
        space = spaces.ElementarySpace.from_basis(symmetry=any_symmetry, sectors_of_basis=sectors_of_basis)
        npt.assert_array_equal(space.sectors, expect_sectors)
        npt.assert_array_equal(space.multiplicities, expect_mults)
        npt.assert_array_equal(space.basis_perm, expect_basis_perm)
        # also check sectors_of_basis property
        npt.assert_array_equal(space.sectors_of_basis, sectors_of_basis)


def test_ElementarySpace_from_sectors(any_symmetry, make_any_sectors, np_random):
    sectors = np.concatenate([make_any_sectors(5) for _ in range(5)])
    multiplicities = np_random.integers(1, 5, size=len(sectors))
    if any_symmetry.can_be_dropped:
        dim = np.sum(multiplicities * any_symmetry.batch_sector_dim(sectors))
        basis_perm = np_random.permutation(dim)
    else:
        basis_perm = None
    #
    # call from_sectors
    res = spaces.ElementarySpace.from_sectors(symmetry=any_symmetry, sectors=sectors,
                                              multiplicities=multiplicities, basis_perm=basis_perm)
    res.test_sanity()
    #
    # check sectors and multiplicities
    expect_sectors = np.unique(sectors, axis=0)
    expect_sectors = expect_sectors[np.lexsort(expect_sectors.T)]
    mult_contributions = np.where(
        np.all(sectors[None, :, :] == expect_sectors[:, None, :], axis=2),
        multiplicities[None, :],
        0
    )
    expect_mults = np.sum(mult_contributions, axis=1)
    assert np.all(res.sectors == expect_sectors)
    assert np.all(res.multiplicities == expect_mults)
    #
    # check basis perm
    if any_symmetry.can_be_dropped:
        expect_internal_basis = []
        for s, m in zip(sectors, multiplicities):
            expect_internal_basis.extend([s] * m * any_symmetry.sector_dim(s))
        expect_internal_basis = np.array(expect_internal_basis)
        expect_public_basis = np.zeros_like(expect_internal_basis)
        expect_public_basis[basis_perm] = expect_internal_basis
        #
        internal_basis = []
        for s, m in zip(res.sectors, res.multiplicities):
            internal_basis.extend([s] * m * any_symmetry.sector_dim(s))
        internal_basis = np.array(internal_basis)
        public_basis = np.zeros_like(internal_basis)
        public_basis[res.basis_perm] = internal_basis
        #
        npt.assert_array_equal(public_basis, expect_public_basis)


def test_take_slice(make_any_space, any_symmetry, np_random):
    if not any_symmetry.can_be_dropped:
        space: spaces.ElementarySpace = make_any_space()
        with pytest.raises(SymmetryError, match='take_slice is meaningless for .*.'):
            _ = space.take_slice([True])
        return
    
    if isinstance(any_symmetry, symmetries.SU2Symmetry):
        sectors = np.array([0, 1, 2, 4])[:, None]
        mults = np.array([3, 1, 2, 2])
        basis_perm = np.array([19, 20, 17, 2, 9, 16, 8, 3, 0, 4, 11, 13, 5, 15, 12, 14, 10, 7, 1, 18, 6])
        space = spaces.ElementarySpace(any_symmetry, sectors, mults, basis_perm)

        # build an allowed and an illegal mask in the internal basis order
        keep_states = []
        illegal_keep_states = []
        for sect, mult in zip(sectors, mults):
            dim = (sect + 1).item()
            for keep in np_random.choice([True, False], size=mult):
                keep_states.extend([keep] * dim)
                print(f'{keep=} {keep_states=}')
                illegal_keep_states.extend([keep] * (dim // 2) + [not keep] * (dim - dim // 2))
        mask = np.array(keep_states)[space.inverse_basis_perm]
        illegal_mask = np.array(illegal_keep_states)[space.inverse_basis_perm]

        with pytest.raises(ValueError, match='Multiplets need to be kept or discarded as a whole.'):
            _ = space.take_slice(illegal_mask)
    else:
        assert any_symmetry.is_abelian, 'Need to design test differently for non-abelian symm'
        space: spaces.ElementarySpace = make_any_space()
        mask = np_random.choice([True, False], size=space.dim)

    small = space.take_slice(mask)
    npt.assert_array_equal(small.sectors_of_basis, space.sectors_of_basis[mask])
    #
    internal_mask = mask[space.basis_perm]
    x = np.arange(space.dim)
    npt.assert_array_equal(x[mask][small.basis_perm], x[space.basis_perm][internal_mask])


def test_ProductSpace(make_any_space):
    """Test TensorProduct and ProductSpace"""
    V1, V2, V3 = [make_any_space() for _ in range(3)]

    examples = [
        spaces.ProductSpace([V1, V2, V3]),  # (V1 x V2 x V3)
        spaces.ProductSpace([V1, V2]),  # (V1 x V2)
        spaces.ProductSpace([spaces.ProductSpace([V1, V2]), V3]),  # ((V1 x V2) x V3)
        spaces.ProductSpace([V2.dual, V3]),  # V2* x V3
        spaces.ProductSpace([V3.dual, V2]),  # V3* x V2
        spaces.ProductSpace([V2]),  # V2
        spaces.ProductSpace([], symmetry=V1.symmetry),  # Cbb
    ]
    expected_duals = [
        spaces.ProductSpace([V3.dual, V2.dual, V1.dual]),
        spaces.ProductSpace([V2.dual, V1.dual]),  # (V1 x V2)
        spaces.ProductSpace([V3.dual, spaces.ProductSpace([V2.dual, V1.dual])]),
        spaces.ProductSpace([V3.dual, V2]),  # V3* x V2
        spaces.ProductSpace([V2.dual, V3]),  # V2* x V3
        spaces.ProductSpace([V2.dual]),  # V2
        spaces.ProductSpace([], symmetry=V1.symmetry),  # Cbb
    ]
    with pytest.raises(ValueError, match='If spaces is empty, the symmetry arg is required.'):
        _ = spaces.ProductSpace([])
    with pytest.raises(symmetries.SymmetryError, match='Incompatible symmetries.'):
        weird_symmetry = symmetries.u1_symmetry * symmetries.u1_symmetry
        _ = spaces.ProductSpace([V1], symmetry=weird_symmetry)

    for n, p1 in enumerate(examples):
        print(f'{n}=')
        print(p1)
        
        p1.test_sanity()
        _ = str(p1)
        _ = repr(p1)

        print('  checking __eq__')
        for m, p2 in enumerate(examples):
            # by construction, expect equality exactly if m == n
            assert (p1 == p2) == (n == m)
        assert p1 != V1
        assert p1 != V2
        assert p1 != V3

        print('  checking .dual')
        p1_dual = p1.dual
        for m, p2 in enumerate(expected_duals):
            # by construction, expect equality exactly if m == n
            assert (p1_dual == p2) == (n == m)
        
        _ = p1.as_ElementarySpace()

    print('empty product is monoidal unit?')
    empty_product = spaces.ProductSpace([], symmetry=V1.symmetry)
    assert np.all(empty_product.sectors == V1.symmetry.trivial_sector)
    assert np.all(empty_product.multiplicities == np.ones(1, dtype=int))
    monoidal_unit = spaces.ElementarySpace.from_trivial_sector(dim=1, symmetry=V1.symmetry)
    assert empty_product.as_ElementarySpace() == monoidal_unit
        

def test_ProductSpace_SU2():
    sym = symmetries.SU2Symmetry()
    a = spaces.ElementarySpace(sym, [[0], [3], [2]], [2, 3, 4])
    b = spaces.ElementarySpace(sym, [[1], [4]], [5, 6])
    c = spaces.ElementarySpace(sym, [[0], [3], [1]], [3, 1, 2])

    ab = spaces.ProductSpace([a, b])
    # a     b           fusion
    #                   0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2            10
    # 0     2                          12
    # 3/2   1/2                15      15
    # 3/2   2              18      18      18      18
    # 1     1/2            20      20
    # 1     2                  24      24      24
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
    spin1 = spaces.ElementarySpace.from_basis(symmetries.z2_symmetry, [even, odd, even])
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


# TODO systematically test ElementarySpace class methods


def test_direct_sum(make_any_space, max_mult=5, max_sectors=5):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=a.is_dual)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=a.is_dual)
    assert a == spaces.ElementarySpace.direct_sum(a)
    d = spaces.ElementarySpace.direct_sum(a, b, c)
    if a.symmetry.can_be_dropped:
        expect = np.concatenate([leg.sectors_of_basis for leg in [a, b, c]], axis=0)
        npt.assert_array_equal(d.sectors_of_basis, expect)
    sector2mult = {}
    for leg in [a, b, c]:
        for s, m in zip(leg.sectors, leg.multiplicities):
            key = tuple(s)
            sector2mult[key] = sector2mult.get(key, 0) + m
    sectors = np.array(list(sector2mult.keys()))
    mults = np.array(list(sector2mult.values()))
    sort = np.lexsort(sectors.T)
    sectors = sectors[sort]
    mults = mults[sort]
    assert np.all(d.sectors == sectors)
    assert np.all(d.multiplicities == mults)


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
    print('ElementarySpace.__repr__()')
    print('----------------------')
    res = repr(space)
    assert len(res) <= repr_max_len
    assert res.count('\n') <= repr_max_lines
    print(res)
    
    print()
    print()
    print('----------------------')
    print('ElementarySpace.__str__() ')
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


# TODO move to some testing tools module?
def assert_spaces_equal(space1: spaces.Space, space2: spaces.Space):
    if isinstance(space1, spaces.ElementarySpace):
        assert isinstance(space2, spaces.ElementarySpace), 'mismatching types'
        assert space1.is_dual == space2.is_dual, 'mismatched is_dual'
        assert space1.symmetry == space2.symmetry, 'mismatched symmetry'
        assert space1.num_sectors == space2.num_sectors, 'mismatched num_sectors'
        assert np.all(space1.multiplicities == space2.multiplicities), 'mismatched multiplicities'
        assert np.all(space1.sectors == space2.sectors), 'mismatched sectors'
        if (space1._basis_perm is not None) or (space2._basis_perm is not None):
            # otherwise both are trivial and this match
            assert np.all(space1.basis_perm == space2.basis_perm), 'mismatched basis_perm'
    elif isinstance(space1, spaces.ProductSpace):
        assert isinstance(space2, spaces.ProductSpace), 'mismatching types'
        assert space1.num_spaces == space2.num_spaces
        for n, (s1, s2) in enumerate(zip(space1.spaces, space2.spaces)):
            try:
                assert_spaces_equal(s1, s2)
            except AssertionError as e:
                raise AssertionError(f'Mismatched spaces[{n}]: {str(e)}') from None
    else:
        raise ValueError('Not a known Space type')

    # should have checked all conditions already, but just to be sure do this again
    assert space1 == space2
