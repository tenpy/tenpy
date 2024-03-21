# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from tenpy.linalg import groups


def common_checks(sym: groups.Symmetry, example_sectors):
    """common consistency checks to be performed on a symmetry instance"""
    assert sym.trivial_sector.shape == (sym.sector_ind_len,)
    assert sym.is_valid_sector(sym.trivial_sector)
    assert not sym.is_valid_sector(np.zeros(shape=(sym.sector_ind_len + 2), dtype=int))
    # objects which are not 1D arrays should not be valid sectors
    for invalid_sector in [0, 1, 42., None, False, 'foo', [0], ['foo'], [None], (), [],
                           np.zeros((1, 1), dtype=int)]:
        assert not sym.is_valid_sector(invalid_sector)
    assert sym.qdim(sym.trivial_sector) in [1, 1.]
    assert sym.num_sectors == np.inf or (isinstance(sym.num_sectors, int) and sym.num_sectors > 0)

    # check all_symmetries
    if sym.num_sectors < np.inf:
        all_sectors = sym.all_sectors()
        assert all_sectors.shape == (sym.num_sectors, sym.sector_ind_len)
        for s in all_sectors:
            assert sym.is_valid_sector(s)

    # just check if they run
    _ = sym.sector_str(sym.trivial_sector)
    _ = repr(sym)
    _ = str(sym)

    # defining property of trivial sector
    for example_sector in example_sectors:
        fusion = sym.fusion_outcomes(example_sector, sym.trivial_sector)
        assert len(fusion) == 1
        assert_array_equal(fusion[0], example_sector)

    # trivial sector is its own dual
    assert_array_equal(sym.dual_sector(sym.trivial_sector), sym.trivial_sector)

    # defining property of dual sector
    try:
        for example_sector in example_sectors:
            assert sym._n_symbol(example_sector, sym.dual_sector(example_sector), sym.trivial_sector) == 1
    except NotImplementedError:
        pytest.xfail("NotImplementedError")
        pass  # TODO SU(2) does not implement n_symbol yet

    # TODO checks of topological data (decide if checking all combinations of all example_sectors is
    # fast enough, if not pick some randomly):
    # - if fusion_tensor available: orthonormal, complete, identity when one input is trivial sector
    # - N symbol consistency:
    #    - associativity: \sum_e N(a, b, e) N(e, c, d) == \sum_f N(b, c, d) N(a, f, d)
    #    - left and right unitor : N(a, 1, b) = delta_{a, b} = N(1, a, b)
    #    - duality N(a, b, 1) == delta_{b, dual(a)}
    # - F symbol consistency:
    #    - triangle equation
    #    - pentagon equation
    #    - unitarity
    # - all data with fallback implementations: compare actual implementation,
    #   e.g. ``sym.frobenius_schur(a) == Symmetry.frobenius_schur(sym, a)``
    # - B symbol normalization
    # - R symbol consistency:
    #   - hexagon equation
    #   - unitarity
    #   - consistency with twist (twist not implemented yet...)
    # - braiding_style
    #   - if trivial, explicitly check the fusion_tensor / CG coefficients
    #   - if symmetric, test the R and C symbols
    # - fusion_style
    #   - check fusion outcomes, N symbol, shape of the other symbols


def test_generic_symmetry(symmetry, symmetry_sectors_rng):
    common_checks(symmetry, symmetry_sectors_rng(10))


def test_no_symmetry():
    sym = groups.NoSymmetry()
    s = np.array([0])
    common_checks(sym, example_sectors=s[np.newaxis, :])

    print('instancecheck and is_abelian')
    assert isinstance(sym, groups.AbelianGroup)
    assert isinstance(sym, groups.GroupSymmetry)
    assert sym.is_abelian

    print('checking valid sectors')
    assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    assert_array_equal(
        sym.fusion_outcomes(s, s),
        np.array([[0]])
    )

    print('checking fusion_outcomes_broadcast')
    many_s = np.stack([s, s, s])
    assert_array_equal(
        sym.fusion_outcomes_broadcast(many_s, many_s),
        many_s
    )

    # print('checking sector dimensions')
    # nothing to do, the dimension of the only sector (trivial) is checked in common_checks

    print('checking equality')
    assert sym == sym
    assert sym == groups.no_symmetry
    assert sym != groups.u1_symmetry
    assert sym != groups.su2_symmetry * groups.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(groups.no_symmetry)
    assert not sym.is_same_symmetry(groups.u1_symmetry)
    assert not sym.is_same_symmetry(groups.su2_symmetry * groups.u1_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s), s)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(many_s), many_s)


def test_product_symmetry():
    sym = groups.ProductSymmetry([
        groups.SU2Symmetry(), groups.U1Symmetry(), groups.FermionParity()
    ])
    sym_with_name = groups.ProductSymmetry([
        groups.SU2Symmetry('foo'), groups.U1Symmetry('bar'), groups.FermionParity()
    ])
    s1 = np.array([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")
    s2 = np.array([3, 2, 0])  # e.g. spin 3/2 , 2 particles , even parity ("bosonic")
    common_checks(sym, example_sectors=np.array([s1, s2]))

    u1_z3 = groups.u1_symmetry * groups.z3_symmetry
    common_checks(u1_z3, example_sectors=np.array([[42, 1], [-1, 2], [-2, 0]]))

    print('instancecheck and is_abelian')
    assert not isinstance(sym, groups.AbelianGroup)
    assert not isinstance(sym, groups.GroupSymmetry)
    assert not sym.is_abelian
    assert isinstance(u1_z3, groups.AbelianGroup)
    assert isinstance(u1_z3, groups.GroupSymmetry)
    assert u1_z3.is_abelian

    print('checking creation via __mul__')
    sym2 = groups.su2_symmetry * groups.u1_symmetry * groups.fermion_parity
    assert sym2 == sym

    print('checking valid sectors')
    assert sym.is_valid_sector(s1)
    assert sym.is_valid_sector(s2)
    assert not sym.is_valid_sector(np.array([-1, 2, 0]))  # negative spin is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 42]))  # parity not in [0, 1] is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 0, 1]))  # too many entries

    print('checking fusion_outcomes')
    outcomes = sym.fusion_outcomes(s1, s2)
    # spin 3/2 and 5/2 can fuse to [1, 2, 3, 4]  ;  U(1) charges  3 + 2 = 5  ;  fermion charges 1 + 0 = 1
    expect = np.array([[2, 5, 1], [4, 5, 1], [6, 5, 1], [8, 5, 1]])
    assert_array_equal(outcomes, expect)

    print('checking fusion_outcomes_broadcast')
    with pytest.raises(AssertionError):
        # sym is not abelian, so this should raise
        _ = sym.fusion_outcomes_broadcast(s1[None, :], s2[None, :])
    outcomes = u1_z3.fusion_outcomes_broadcast(np.array([[42, 2], [-2, 0]]), np.array([[1, 1], [2, 1]]))
    assert_array_equal(outcomes, np.array([[43, 0], [0, 1]]))

    print('checking sector dimensions')
    assert sym.sector_dim(s1) == 6
    assert sym.sector_dim(s2) == 4

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym != groups.su2_symmetry * groups.u1_symmetry
    assert sym != groups.no_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert not sym.is_same_symmetry(groups.su2_symmetry * groups.u1_symmetry)
    assert not sym.is_same_symmetry(groups.no_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s1), np.array([5, -3, 1]))
    assert_array_equal(sym.dual_sector(s2), np.array([3, -2, 0]))

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([s1, s2])), np.array([[5, -3, 1], [3, -2, 0]]))


def test_u1_symmetry():
    sym = groups.U1Symmetry()
    sym_with_name = groups.U1Symmetry('foo')
    s_0 = np.array([0])
    s_1 = np.array([1])
    s_neg1 = np.array([-1])
    s_2 = np.array([2])
    s_42 = np.array([42])
    common_checks(sym, example_sectors=np.array([s_0, s_1, s_neg1, s_2, s_42]))

    print('instancecheck and is_abelian')
    assert isinstance(sym, groups.AbelianGroup)
    assert isinstance(sym, groups.GroupSymmetry)
    assert sym.is_abelian

    print('checking valid sectors')
    for s in [s_0, s_1, s_neg1, s_2, s_42]:
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    assert_array_equal(sym.fusion_outcomes(s_1, s_1), s_2[None, :])
    assert_array_equal(sym.fusion_outcomes(s_neg1, s_1), s_0[None, :])

    print('checking fusion_outcomes_broadcast')
    outcomes = sym.fusion_outcomes_broadcast(np.stack([s_0, s_1, s_0]), np.stack([s_neg1, s_1, s_2]))
    assert_array_equal(outcomes, np.stack([s_neg1, s_2, s_2]))

    print('checking sector dimensions')
    for s in [s_0, s_1, s_neg1, s_2, s_42]:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym == groups.u1_symmetry
    assert sym != groups.no_symmetry
    assert sym != groups.su2_symmetry * groups.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert sym.is_same_symmetry(groups.u1_symmetry)
    assert not sym.is_same_symmetry(groups.no_symmetry)
    assert not sym.is_same_symmetry(groups.su2_symmetry * groups.u1_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s_1), s_neg1)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([s_0, s_1, s_42, s_2])),
                       np.array([0, -1, -42, -2])[:, None])


@pytest.mark.parametrize('N', [2, 3, 4, 42])
def test_ZN_symmetry(N):
    sym = groups.ZNSymmetry(N=N)
    sym_with_name = groups.ZNSymmetry(N, descriptive_name='foo')
    sectors_a = np.array([0, 1, 2, 10])[:, None] % N
    sectors_b = np.array([0, 1, 3, 11])[:, None] % N
    common_checks(sym, example_sectors=sectors_a)

    print('instancecheck and is_abelian')
    assert isinstance(sym, groups.AbelianGroup)
    assert isinstance(sym, groups.GroupSymmetry)
    assert sym.is_abelian

    print('checking valid sectors')
    for s in sectors_a:
        assert sym.is_valid_sector(s)
    assert not sym.is_valid_sector(np.array([0, 0]))
    assert not sym.is_valid_sector(np.array([N]))
    assert not sym.is_valid_sector(np.array([-1]))

    print('checking fusion_outcomes')
    for a in sectors_a:
        for b in sectors_b:
            expect = (a + b)[None, :] % N
            assert_array_equal(sym.fusion_outcomes(a, b), expect)

    print('checking fusion_outcomes_broadcast')
    expect = (sectors_a + sectors_b) % N
    assert_array_equal(sym.fusion_outcomes_broadcast(sectors_a, sectors_b), expect)

    print('checking sector dimensions')
    for s in sectors_a:
        assert sym.sector_dim(s) == 1

    print('checking equality')
    other = {
        2: groups.z2_symmetry,
        3: groups.z3_symmetry,
        4: groups.z4_symmetry,
        5: groups.z5_symmetry,
        42: groups.ZNSymmetry(42),
        43: groups.ZNSymmetry(43),
    }
    assert sym == sym
    assert sym != sym_with_name
    assert sym == other[N]
    assert sym != other[N + 1]
    assert sym != groups.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert sym.is_same_symmetry(other[N])
    assert not sym.is_same_symmetry(other[N + 1])
    assert not sym.is_same_symmetry(groups.u1_symmetry)

    print('checking dual_sector')
    for s in sectors_a:
        assert_array_equal(sym.dual_sector(s), (-s) % N)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(sectors_a), (-sectors_a) % N)


def test_su2_symmetry_common():
    sym = groups.SU2Symmetry()
    common_checks(sym, example_sectors=np.array([3]))


def test_su2_symmetry():
    sym = groups.SU2Symmetry()
    spin_1 = np.array([2])
    spin_3_half = np.array([3])
    sym_with_name = groups.SU2Symmetry('foo')

    print('instancecheck and is_abelian')
    assert not isinstance(sym, groups.AbelianGroup)
    assert isinstance(sym, groups.GroupSymmetry)
    assert not sym.is_abelian

    print('checking valid sectors')
    for valid in [[0], [1], [2], [42]]:
        assert sym.is_valid_sector(np.array(valid))
    for invalid in [[-1], [0, 0]]:
        assert not sym.is_valid_sector(np.array(invalid))

    print('checking fusion_outcomes')
    # 1 x 3/2 = 1/2 + 3/2 + 5/2
    assert_array_equal(
        sym.fusion_outcomes(spin_1, spin_3_half),
        np.array([[1], [3], [5]])
    )

    print('checking fusion_outcomes_broadcast')
    with pytest.raises(AssertionError):
        # sym does not have FusionStyle.single, so this should raise
        _ = sym.fusion_outcomes_broadcast(spin_1[None, :], spin_3_half[None, :])

    print('checking sector dimensions')
    assert sym.sector_dim(spin_1) == 3
    assert sym.sector_dim(spin_3_half) == 4

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym == groups.su2_symmetry
    assert sym != groups.fermion_parity

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(spin_1), spin_1)
    assert_array_equal(sym.dual_sector(spin_3_half), spin_3_half)

    print('checking dual_sectors')
    assert_array_equal(
        sym.dual_sectors(np.stack([spin_1, spin_3_half])),
        np.stack([spin_1, spin_3_half])
    )


def test_fermion_parity():
    sym = groups.FermionParity()
    even = np.array([0])
    odd = np.array([1])
    common_checks(sym, example_sectors=np.array([even, odd]))

    print('instancecheck and is_abelian')
    assert not isinstance(sym, groups.AbelianGroup)
    assert not isinstance(sym, groups.GroupSymmetry)
    assert sym.is_abelian

    print('checking valid sectors')
    assert sym.is_valid_sector(odd)
    assert not sym.is_valid_sector(np.array([2]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion_outcomes')
    assert_array_equal(sym.fusion_outcomes(odd, odd), even[None, :])
    assert_array_equal(sym.fusion_outcomes(odd, even), odd[None, :])

    print('checking fusion_outcomes_broadcast')
    assert_array_equal(
        sym.fusion_outcomes_broadcast(np.stack([even, even, odd]), np.stack([even, odd, odd])),
        np.stack([even, odd, even])
    )

    print('checking equality')
    assert sym == sym
    assert sym == groups.fermion_parity
    assert sym != groups.no_symmetry
    assert sym != groups.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert not sym.is_same_symmetry(groups.no_symmetry)
    assert not sym.is_same_symmetry(groups.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(odd), odd)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([odd, even, odd])),
                       np.stack([odd, even, odd]))


@pytest.mark.parametrize('handedness', ['left', 'right'])
def test_fibonacci_grading(handedness):
    sym = groups.FibonacciGrading(handedness)
    vac = np.array([0])
    tau = np.array([1])
    common_checks(sym, example_sectors=sym.all_sectors())

    print('instancecheck and is_abelian')
    assert not isinstance(sym, groups.AbelianGroup)
    assert not isinstance(sym, groups.GroupSymmetry)
    assert not sym.is_abelian

    print('checking valid sectors')
    assert sym.is_valid_sector(tau)
    assert not sym.is_valid_sector(np.array([3]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion rules')
    assert_array_equal(sym.fusion_outcomes(vac, tau), tau[None, :])
    assert_array_equal(sym.fusion_outcomes(tau, tau), np.stack([vac, tau]))

    print('checking equality')
    assert sym == sym
    assert (sym == groups.fibonacci_grading) == (handedness == 'left')
    assert sym != groups.no_symmetry
    assert sym != groups.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(groups.fibonacci_grading) == (handedness == 'left')
    assert not sym.is_same_symmetry(groups.no_symmetry)
    assert not sym.is_same_symmetry(groups.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(tau), tau)


@pytest.mark.parametrize('nu', [*range(1, 16, 2)])
def test_ising_grading(nu):
    sym = groups.IsingGrading(nu)
    vac = np.array([0])
    anyon = np.array([1])
    fermion = np.array([2])
    common_checks(sym, example_sectors=sym.all_sectors())

    print('instancecheck and is_abelian')
    assert not isinstance(sym, groups.AbelianGroup)
    assert not isinstance(sym, groups.GroupSymmetry)
    assert not sym.is_abelian

    print('checking valid sectors')
    assert sym.is_valid_sector(anyon)
    assert sym.is_valid_sector(fermion)
    assert not sym.is_valid_sector(np.array([3]))
    assert not sym.is_valid_sector(np.array([-1]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion rules')
    assert_array_equal(sym.fusion_outcomes(vac, anyon), anyon[None, :])
    assert_array_equal(sym.fusion_outcomes(vac, fermion), fermion[None, :])
    assert_array_equal(sym.fusion_outcomes(anyon, fermion), anyon[None, :])
    assert_array_equal(sym.fusion_outcomes(anyon, anyon), np.stack([vac, fermion]))

    print('checking equality')
    assert sym == sym
    assert (sym == groups.ising_grading) == (nu == 1)
    assert sym != groups.no_symmetry
    assert sym != groups.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(groups.ising_grading) == (nu == 1)
    assert not sym.is_same_symmetry(groups.no_symmetry)
    assert not sym.is_same_symmetry(groups.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(anyon), anyon)
    assert_array_equal(sym.dual_sector(fermion), fermion)
