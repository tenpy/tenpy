# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import pytest
from numpy import testing as npt

from cyten import SymmetryError
from cyten.block_backends import NumpyBlockBackend
from cyten.symmetries import _symmetries, spaces, trees
from cyten.testing import random_ElementarySpace
from cyten.tools import is_permutation

# TODO test all cases of Space.as_ElementarySpace


def test_ElementarySpace(any_symmetry, make_any_sectors, np_random):
    sectors = make_any_sectors(10)
    sectors = sectors[np.lexsort(sectors.T)]
    dual_sectors = any_symmetry.dual_sectors(sectors)
    dual_sectors_sort = np.lexsort(dual_sectors.T)
    mults = np_random.integers(1, 10, size=len(sectors))

    s1 = spaces.ElementarySpace(symmetry=any_symmetry, defining_sectors=sectors, multiplicities=mults)
    s2 = spaces.ElementarySpace.from_trivial_sector(dim=8)

    print('checking ElementarySpace.sector_decomposition')
    npt.assert_array_equal(s2.sector_decomposition, _symmetries.no_symmetry.trivial_sector[None, :])

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
    assert s1 != spaces.ElementarySpace(symmetry=any_symmetry, defining_sectors=sectors, multiplicities=wrong_mults)
    npt.assert_array_equal(s1_dual.defining_sectors, s1.defining_sectors)
    npt.assert_array_equal(s1_dual.sector_decomposition, dual_sectors)
    npt.assert_array_equal(s1_dual.multiplicities, s1.multiplicities)
    assert s1_dual.symmetry == s1.symmetry
    assert s1_dual.is_dual is True

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert spaces.ElementarySpace.from_trivial_sector(dim=1).is_trivial
    assert spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=any_symmetry.trivial_sector[np.newaxis, :]
    ).is_trivial

    print('checking is_subspace_of')
    same_sectors_less_mults = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=sectors, multiplicities=[max(1, m - 1) for m in mults]
    )
    same_sectors_different_mults = spaces.ElementarySpace(
        symmetry=any_symmetry,
        defining_sectors=sectors,
        multiplicities=[max(1, m + (+1 if i % 2 == 0 else -1)) for i, m in enumerate(mults)],
    )  # but at least one mult is larger than for s1
    if len(sectors) > 2:
        which1 = [0, -1]
        which2 = [1, -2]
    else:
        # if there are only two sectors, we cant have different sets of sectors,
        # both of which have multiple entries
        which1 = [0]
        which2 = [-1]
    fewer_sectors1 = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=[sectors[i] for i in which1], multiplicities=[mults[i] for i in which1]
    )
    fewer_sectors2 = spaces.ElementarySpace(
        symmetry=any_symmetry, defining_sectors=[sectors[i] for i in which2], multiplicities=[mults[i] for i in which2]
    )
    assert s1.is_subspace_of(s1)
    expect_dual_is_subspace = np.all(s1.sector_decomposition == dual_sectors)
    assert s1_dual.is_subspace_of(s1) == expect_dual_is_subspace

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

    print('check idx_to_sector and parse_idx')
    if any_symmetry.can_be_dropped:
        idx = 0  # will step this up during the loop
        for n_sector, sector in enumerate(s1.sector_decomposition):
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
        assert s1.sector_decomposition_where(s1.sector_decomposition[expect]) == expect
        assert s1.sector_decomposition_where(s1.sector_decomposition[expect]) == expect
        assert s1.sector_multiplicity(s1.sector_decomposition[expect]) == s1.multiplicities[expect]

    print('check from_basis')
    if any_symmetry.can_be_dropped:
        if isinstance(any_symmetry, _symmetries.SU2Symmetry):
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
        npt.assert_array_equal(space.sector_decomposition, expect_sectors)
        npt.assert_array_equal(space.multiplicities, expect_mults)
        npt.assert_array_equal(space.basis_perm, expect_basis_perm)
        # also check sectors_of_basis property
        npt.assert_array_equal(space.sectors_of_basis, sectors_of_basis)


def test_ElementarySpace_from_defining_sectors(any_symmetry, make_any_sectors, np_random):
    # TODO also test from_sector_decomposition
    sectors = np.concatenate([make_any_sectors(5) for _ in range(5)])
    multiplicities = np_random.integers(1, 5, size=len(sectors))
    if any_symmetry.can_be_dropped:
        dim = np.sum(multiplicities * any_symmetry.batch_sector_dim(sectors))
        basis_perm = np_random.permutation(dim)
    else:
        basis_perm = None
    #
    # call from_defining_sectors
    res = spaces.ElementarySpace.from_defining_sectors(
        symmetry=any_symmetry, defining_sectors=sectors, multiplicities=multiplicities, basis_perm=basis_perm
    )
    res.test_sanity()
    #
    # check sectors and multiplicities
    expect_sectors = np.unique(sectors, axis=0)
    expect_sectors = expect_sectors[np.lexsort(expect_sectors.T)]
    mult_contributions = np.where(
        np.all(sectors[None, :, :] == expect_sectors[:, None, :], axis=2), multiplicities[None, :], 0
    )
    expect_mults = np.sum(mult_contributions, axis=1)
    assert np.all(res.sector_decomposition == expect_sectors)
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
        for s, m in zip(res.sector_decomposition, res.multiplicities):
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

    if isinstance(any_symmetry, _symmetries.SU2Symmetry):
        sectors = np.array([0, 1, 2, 4])[:, None]
        mults = np.array([3, 1, 2, 2])
        basis_perm = np.array([19, 20, 17, 2, 9, 16, 8, 3, 0, 4, 11, 13, 5, 15, 12, 14, 10, 7, 1, 18, 6])
        space = spaces.ElementarySpace(
            any_symmetry, defining_sectors=sectors, multiplicities=mults, basis_perm=basis_perm
        )

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


def check_basis_perm(perm, inv_perm=None):
    """Check that `perm` is a valid permutation, and optionally that `inv_perm` is its inverse."""
    perm = np.asarray(perm)
    assert perm.ndim == 1
    assert is_permutation(perm)

    if inv_perm is not None:
        check_basis_perm(perm=inv_perm, inv_perm=None)
        npt.assert_array_equal(perm[inv_perm], np.arange(len(perm)))
        npt.assert_array_equal(inv_perm[perm], np.arange(len(perm)))


def test_LegPipe_basis_perm(np_random):
    block_backend = NumpyBlockBackend()
    a = spaces.ElementarySpace.from_trivial_sector(dim=5, is_dual=True, basis_perm=np_random.permutation(5))
    b = spaces.ElementarySpace.from_trivial_sector(dim=6, is_dual=False, basis_perm=np_random.permutation(6))
    c = spaces.ElementarySpace.from_trivial_sector(dim=7, is_dual=True, basis_perm=np_random.permutation(7))
    d = spaces.ElementarySpace.from_trivial_sector(dim=8, is_dual=False, basis_perm=np_random.permutation(8))

    data = np.arange(a.dim * b.dim * c.dim * d.dim).reshape([a.dim, b.dim, c.dim, d.dim])

    pipe_ab = spaces.LegPipe([a, b])
    check_basis_perm(pipe_ab.basis_perm, pipe_ab.inverse_basis_perm)
    data1 = block_backend.apply_leg_permutations(
        data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm]
    ).reshape((a.dim * b.dim, c.dim, d.dim))
    data2 = block_backend.apply_leg_permutations(
        data.reshape((a.dim * b.dim, c.dim, d.dim)), [pipe_ab.basis_perm, c.basis_perm, d.basis_perm]
    )
    npt.assert_array_equal(data1, data2)

    pipe_abc = spaces.LegPipe([a, b, c])
    check_basis_perm(pipe_abc.basis_perm, pipe_abc.inverse_basis_perm)
    data1 = block_backend.apply_leg_permutations(
        data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm]
    ).reshape((a.dim * b.dim * c.dim, d.dim))
    data2 = block_backend.apply_leg_permutations(
        data.reshape((a.dim * b.dim * c.dim, d.dim)), [pipe_abc.basis_perm, d.basis_perm]
    )
    npt.assert_array_equal(data1, data2)

    pipe_nested = spaces.LegPipe([pipe_ab, c])
    check_basis_perm(pipe_nested.basis_perm, pipe_nested.inverse_basis_perm)
    data1 = block_backend.apply_leg_permutations(
        data, [a.basis_perm, b.basis_perm, c.basis_perm, d.basis_perm]
    ).reshape((a.dim * b.dim * c.dim, d.dim))
    data2 = block_backend.apply_leg_permutations(
        data.reshape((a.dim * b.dim * c.dim, d.dim)), [pipe_nested.basis_perm, d.basis_perm]
    )
    npt.assert_array_equal(data1, data2)


@pytest.mark.parametrize('num_spaces', [3, 4, 5])
def test_TensorProduct(any_symmetry, make_any_space, make_any_sectors, num_spaces):
    domain = spaces.TensorProduct([make_any_space() for _ in range(num_spaces)], symmetry=any_symmetry)
    domain.test_sanity()

    for coupled in make_any_sectors(10):
        expect1 = sum(
            len(trees.fusion_trees(any_symmetry, uncoupled, coupled)) * np.prod(mults)
            for uncoupled, mults in domain.iter_uncoupled()
        )
        expect2 = sum(domain.forest_block_size(uncoupled, coupled) for uncoupled, _ in domain.iter_uncoupled())
        res = domain.block_size(coupled)
        assert res == expect1
        assert res == expect2


def test_TensorProduct_SU2():
    sym = _symmetries.SU2Symmetry()
    a = spaces.ElementarySpace(sym, [[0], [3], [2]], [2, 3, 4])
    b = spaces.ElementarySpace(sym, [[1], [4]], [5, 6])
    c = spaces.ElementarySpace(sym, [[0], [3], [1]], [3, 1, 2])

    ab = spaces.TensorProduct([a, b])
    # a     b           fusion
    #                   0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2            10
    # 0     2                          12
    # 3/2   1/2                15      15
    # 3/2   2              18      18      18      18
    # 1     1/2            20      20
    # 1     2                  24      24      24
    npt.assert_array_equal(ab.sector_decomposition, np.array([1, 2, 3, 4, 5, 6, 7])[:, None])
    npt.assert_array_equal(ab.multiplicities, np.array([48, 39, 38, 51, 18, 24, 18]))

    bc = spaces.TensorProduct([b, c])
    # c     b      mult     fusion
    #                       0 1/2   1 3/2   2 5/2   3 7/2
    # 0     1/2    3*5         15
    # 0     2      3*6                     18
    # 3/2   1/2    1*5              5       5
    # 3/2   2      1*6          6       6       6       6
    # 1/2   1/2    2*5     10      10
    # 1/2   2      2*6                 12      12
    npt.assert_array_equal(bc.sector_decomposition, np.array([0, 1, 2, 3, 4, 5, 7])[:, None])
    npt.assert_array_equal(bc.multiplicities, np.array([10, 21, 15, 18, 23, 18, 6]))

    abc = spaces.TensorProduct([a, b, c])
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
    expect_mults = [
        96 + 38,
        144 + 39 + 78 + 51,
        48 + 96 + 117 + 38 + 76 + 18,
        39 + 78 + 114 + 51 + 102 + 24,
        48 + 38 + 76 + 153 + 18 + 36 + 18,
        39 + 51 + 102 + 54 + 24 + 48,
        38 + 18 + 36 + 72 + 18 + 36,
        51 + 24 + 48 + 54,
        18 + 18 + 36,
        24,
        18,
    ]
    npt.assert_array_equal(abc.sector_decomposition, np.arange(11)[:, None])
    npt.assert_array_equal(abc.multiplicities, np.array(expect_mults))


@pytest.mark.parametrize('combine_cstyle', [True, False])
@pytest.mark.parametrize('pipe_dual', [False, True])
def test_AbelianLegPipe(abelian_group_symmetry, combine_cstyle, pipe_dual, np_random):
    def iter_combinations(s1, s2):
        # iterate combinations in either C-style or F-style
        if combine_cstyle:
            for a in s1:
                for b in s2:
                    yield a, b
        else:
            for b in s2:
                for a in s1:
                    yield a, b

    leg_1: spaces.ElementarySpace = random_ElementarySpace(symmetry=abelian_group_symmetry, np_random=np_random)
    leg_2: spaces.ElementarySpace = random_ElementarySpace(symmetry=abelian_group_symmetry, np_random=np_random)
    leg_1.basis_perm = np_random.permutation(leg_1.dim)
    leg_2.basis_perm = np_random.permutation(leg_2.dim)

    pipe = spaces.AbelianLegPipe([leg_1, leg_2], is_dual=pipe_dual, combine_cstyle=combine_cstyle)
    pipe.test_sanity()

    # Setup
    # =======================================
    # for each basis element, store its sector and an int as a unique identifier
    public_basis_1 = [(s, i) for i, s in enumerate(leg_1.sectors_of_basis)]
    public_basis_2 = [(s, i) for i, s in enumerate(leg_2.sectors_of_basis)]
    internal_basis_1 = [public_basis_1[n] for n in leg_1.basis_perm]
    internal_basis_2 = [public_basis_2[n] for n in leg_2.basis_perm]

    # make sure we built them correctly
    start = 0
    for sector, mult in zip(leg_1.sector_decomposition, leg_1.multiplicities):
        for b in internal_basis_1[start : start + mult]:
            assert np.all(b[0] == sector)
        start = start + mult
    assert start == leg_1.dim

    # Misc properties of the pipe
    # =======================================
    assert pipe.is_isomorphic_to(spaces.TensorProduct([leg_1, leg_2]))

    # check fusion_outcomes_sort
    # =======================================
    fusion_outcomes = [
        abelian_group_symmetry.fusion_outcomes(s_1, s_2)[0]
        for s_1, s_2 in iter_combinations(leg_1.sector_decomposition, leg_2.sector_decomposition)
    ]
    fusion_outcomes_sorted, expect = _sort_sectors(fusion_outcomes, abelian_group_symmetry, by_duals=pipe.is_dual)
    assert np.all(pipe.fusion_outcomes_sort == expect)

    # check block_ind_map_slices
    # =======================================
    for n, sector in enumerate(pipe.sector_decomposition):
        start = pipe.block_ind_map_slices[n]
        stop = pipe.block_ind_map_slices[n + 1]
        assert np.all(fusion_outcomes_sorted[start:stop, :] == sector)

    # check block_ind_map
    # =======================================
    pass  # completely checked by pipe.test_sanity()

    # check _get_fusion_outcomes_perm()
    # =======================================
    internal_fusion_outcomes = [
        (abelian_group_symmetry.fusion_outcomes(b_1[0], b_2[0])[0], b_1[1], b_2[1])
        for b_1, b_2 in iter_combinations(internal_basis_1, internal_basis_2)
    ]
    _, fusion_outcomes_perm = _sort_sectors(
        [b[0] for b in internal_fusion_outcomes], abelian_group_symmetry, by_duals=pipe.is_dual
    )

    assert np.all(pipe._get_fusion_outcomes_perm(pipe.multiplicities) == fusion_outcomes_perm)

    # check basis_perm
    # =======================================
    assert pipe.basis_perm.shape == (pipe.dim,)
    assert np.all(np.sort(pipe.basis_perm) == np.arange(pipe.dim))
    public_basis_pipe = [
        (abelian_group_symmetry.fusion_outcomes(b_1[0], b_2[0])[0], b_1[1], b_2[1])
        for b_1, b_2 in iter_combinations(public_basis_1, public_basis_2)
    ]
    internal_basis_pipe = [internal_fusion_outcomes[n] for n in fusion_outcomes_perm]

    # want to do ``expect_perm = [public_basis_pipe.index(i) for i in internal_basis_pipe]``
    # but need to deal with array comparison
    expect_perm = []
    for i in internal_basis_pipe:
        for j, p in enumerate(public_basis_pipe):
            if np.all(i[0] == p[0]) and i[1:] == p[1:]:
                expect_perm.append(j)
                break
        else:  # else == "no break occurred"
            raise RuntimeError

    assert np.all(pipe.basis_perm == np.array(expect_perm))


@pytest.mark.parametrize('is_dual', [True, False])
def test_direct_sum(is_dual, make_any_space, max_mult=5, max_sectors=5):
    a = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    b = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    c = make_any_space(max_mult=max_mult, max_sectors=max_sectors, is_dual=is_dual)
    assert a == spaces.ElementarySpace.direct_sum(a)
    d = spaces.ElementarySpace.direct_sum(a, b, c)
    d.test_sanity()
    assert d.is_dual == is_dual
    if a.symmetry.can_be_dropped:
        expect = np.concatenate([leg.sectors_of_basis for leg in [a, b, c]], axis=0)
        npt.assert_array_equal(d.sectors_of_basis, expect)
    sector2mult = {}
    for leg in [a, b, c]:
        for s, m in zip(leg.sector_decomposition, leg.multiplicities):
            key = tuple(s)
            sector2mult[key] = sector2mult.get(key, 0) + m
    sectors = np.array(list(sector2mult.keys()))
    mults = np.array(list(sector2mult.values()))
    sort = np.lexsort(sectors.T)
    sectors = sectors[sort]
    mults = mults[sort]
    if is_dual:
        expected_order = np.lexsort(d.sector_decomposition.T)
    else:
        expected_order = slice(None, None, None)
    assert np.all(d.sector_decomposition[expected_order] == sectors)
    assert np.all(d.multiplicities[expected_order] == mults)


def test_str_repr(make_any_space, any_symmetry, str_max_lines=20, repr_max_lines=20):
    """Check if str and repr work. Automatically, we can only check if they run at all.
    To check if the output is sensible and useful, a human should look at it.
    Run ``pytest -rP -k test_str_repr`` to see the output.
    """
    terminal_width = 80
    str_max_len = terminal_width * str_max_lines
    repr_max_len = terminal_width * str_max_lines
    # TODO output is a bit long, should we force shorter? -> consider config.printoptions!

    instances = {
        'ElementarySpace (short)': make_any_space(max_sectors=3, is_dual=True),
        'ElementarySpace (med)': make_any_space(max_sectors=10, is_dual=False),
        'ElementarySpace (long)': make_any_space(max_sectors=100, is_dual=True),
        'LegPipe (1)': spaces.LegPipe([make_any_space(max_sectors=5)]),
        'LegPipe (3)': spaces.LegPipe([make_any_space(max_sectors=5) for _ in range(3)], is_dual=True),
        'LegPipe (5)': spaces.LegPipe([make_any_space(max_sectors=3) for _ in range(5)]),
        'TensorProduct (1)': spaces.TensorProduct(
            [make_any_space(max_sectors=5)],
        ),
        'TensorProduct (3)': spaces.TensorProduct([make_any_space(max_sectors=5) for _ in range(3)]),
        'TensorProduct (5)': spaces.TensorProduct([make_any_space(max_sectors=3) for _ in range(5)]),
    }
    if any_symmetry.is_abelian and any_symmetry.can_be_dropped:
        more = {
            'AbelianLegPipe (1)': spaces.AbelianLegPipe([make_any_space(max_sectors=5)]),
            'AbelianLegPipe (1F)': spaces.AbelianLegPipe([make_any_space(max_sectors=5)], combine_cstyle=False),
            'AbelianLegPipe (3)': spaces.AbelianLegPipe(
                [make_any_space(max_sectors=5) for _ in range(3)], is_dual=True
            ),
            'AbelianLegPipe (5)': spaces.AbelianLegPipe(
                [make_any_space(max_sectors=3) for _ in range(5)],
            ),
        }
        instances.update(more)

    for name, space in instances.items():
        space.test_sanity()
        print()
        print()
        print('-' * 40)
        print(name)
        print('-' * 40)
        res = repr(space)
        assert len(res) <= repr_max_len
        assert res.count('\n') <= repr_max_lines
        print(res)

        print()
        res = str(space)
        assert len(res) <= str_max_len
        assert res.count('\n') <= str_max_lines
        print(res)


# TODO move to some testing tools module?
def assert_spaces_equal(space1: spaces.Space, space2: spaces.Space):
    # TODO review in light of new spaces classes
    if isinstance(space1, spaces.ElementarySpace):
        assert isinstance(space2, spaces.ElementarySpace), 'mismatching types'
        assert space1.is_dual == space2.is_dual, 'mismatched is_dual'
        assert space1.symmetry == space2.symmetry, 'mismatched symmetry'
        assert space1.num_sectors == space2.num_sectors, 'mismatched num_sectors'
        assert np.all(space1.multiplicities == space2.multiplicities), 'mismatched multiplicities'
        assert np.all(space1.sector_decomposition == space2.sector_decomposition), 'mismatched sectors'
        if (space1._basis_perm is not None) or (space2._basis_perm is not None):
            # otherwise both are trivial and this match
            assert np.all(space1.basis_perm == space2.basis_perm), 'mismatched basis_perm'
    elif isinstance(space1, spaces.TensorProduct):
        assert isinstance(space2, spaces.TensorProduct), 'mismatching types'
        assert space1.num_factors == space2.num_factors
        for n, (s1, s2) in enumerate(zip(space1.factors, space2.factors)):
            try:
                assert_spaces_equal(s1, s2)
            except AssertionError as e:
                raise AssertionError(f'Mismatched spaces[{n}]: {str(e)}') from None
    else:
        raise ValueError('Not a known Space type')

    # should have checked all conditions already, but just to be sure do this again
    assert space1 == space2


def _sort_sectors(sectors, sym: _symmetries.Symmetry, by_duals: bool = False):
    sectors = np.array(sectors)
    sort_by = sym.dual_sectors(sectors) if by_duals else sectors
    perm = np.lexsort(sort_by.T)
    return sectors[perm], perm
