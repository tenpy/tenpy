"""A collection of tests for tenpy.linalg.symmetries."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from tenpy.linalg import symmetries


# TODO check n_symbol (+ further topological data, once implemented)


def common_checks(sym: symmetries.Symmetry, example_sector):
    # common consistency checks to be performed on a symmetry instance
    assert sym.trivial_sector.shape == (sym.sector_ind_len,)
    assert sym.is_valid_sector(sym.trivial_sector)
    assert not sym.is_valid_sector(np.zeros(shape=(sym.sector_ind_len + 2), dtype=int))
    for invalid_sector in [0, 1, 42., None, False, 'foo', [0], ['foo'], [None], (), []]:
        assert not sym.is_valid_sector(invalid_sector)
    assert sym.sector_dim(sym.trivial_sector) == 1

    # just check if they run
    _ = sym.sector_str(sym.trivial_sector)
    _ = repr(sym)
    _ = str(sym)

    # defining property of trivial sector
    assert_array_equal(
        sym.fusion_outcomes(example_sector, sym.trivial_sector),
        example_sector[None, :]
    )

    # trivial sector is its own dual
    assert_array_equal(sym.dual_sector(sym.trivial_sector), sym.trivial_sector)

    # defining property of dual sector
    try:
        assert sym.n_symbol(example_sector, sym.dual_sector(example_sector), sym.trivial_sector) == 1
    except NotImplementedError:
        pass  # TODO SU(2) does not implement n_symbol yet


def test_no_symmetry():
    sym = symmetries.NoSymmetry()
    s = np.array([0])
    common_checks(sym, example_sector=s)

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.Group)
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
    assert sym == symmetries.no_symmetry
    assert sym != symmetries.u1_symmetry
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(symmetries.no_symmetry)
    assert not sym.is_same_symmetry(symmetries.u1_symmetry)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry * symmetries.u1_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s), s)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(many_s), many_s)


def test_product_symmetry():
    sym = symmetries.ProductSymmetry([
        symmetries.SU2Symmetry(), symmetries.U1Symmetry(), symmetries.FermionParity()
    ])
    sym_with_name = symmetries.ProductSymmetry([
        symmetries.SU2Symmetry('foo'), symmetries.U1Symmetry('bar'), symmetries.FermionParity()
    ])
    s1 = np.array([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")
    s2 = np.array([3, 2, 0])  # e.g. spin 3/2 , 2 particles , even parity ("bosonic")
    common_checks(sym, example_sector=s1)

    u1_z3 = symmetries.u1_symmetry * symmetries.z3_symmetry
    common_checks(u1_z3, example_sector=np.array([42, 1]))

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.Group)
    assert not sym.is_abelian
    assert isinstance(u1_z3, symmetries.AbelianGroup)
    assert isinstance(u1_z3, symmetries.Group)
    assert u1_z3.is_abelian

    print('checking creation via __mul__')
    sym2 = symmetries.su2_symmetry * symmetries.u1_symmetry * symmetries.fermion_parity
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
        # sym does not have FusionStyle.single, so this should raise
        _ = sym.fusion_outcomes_broadcast(s1[None, :], s2[None, :])
    outcomes = u1_z3.fusion_outcomes_broadcast(np.array([[42, 2], [-2, 0]]), np.array([[1, 1], [2, 1]]))
    assert_array_equal(outcomes, np.array([[43, 0], [0, 1]]))

    print('checking sector dimensions')
    assert sym.sector_dim(s1) == 6
    assert sym.sector_dim(s2) == 4

    print('checking equality')
    assert sym == sym
    assert sym != sym_with_name
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry
    assert sym != symmetries.no_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry * symmetries.u1_symmetry)
    assert not sym.is_same_symmetry(symmetries.no_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s1), np.array([5, -3, 1]))
    assert_array_equal(sym.dual_sector(s2), np.array([3, -2, 0]))

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([s1, s2])), np.array([[5, -3, 1], [3, -2, 0]]))


def test_u1_symmetry():
    sym = symmetries.U1Symmetry()
    sym_with_name = symmetries.U1Symmetry('foo')
    s_0 = np.array([0])
    s_1 = np.array([1])
    s_neg1 = np.array([-1])
    s_2 = np.array([2])
    s_42 = np.array([42])
    common_checks(sym, example_sector=s_42)

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.Group)
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
    assert sym == symmetries.u1_symmetry
    assert sym != symmetries.no_symmetry
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert sym.is_same_symmetry(symmetries.u1_symmetry)
    assert not sym.is_same_symmetry(symmetries.no_symmetry)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry * symmetries.u1_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(s_1), s_neg1)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([s_0, s_1, s_42, s_2])),
                       np.array([0, -1, -42, -2])[:, None])


@pytest.mark.parametrize('N', [2, 3, 4, 42])
def test_ZN_symmetry(N):
    sym = symmetries.ZNSymmetry(N=N)
    sym_with_name = symmetries.ZNSymmetry(N, descriptive_name='foo')
    sectors_a = np.array([0, 1, 2, 10])[:, None] % N
    sectors_b = np.array([0, 1, 3, 11])[:, None] % N
    common_checks(sym, example_sector=np.array([1]))

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.Group)
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
        2: symmetries.z2_symmetry,
        3: symmetries.z3_symmetry,
        4: symmetries.z4_symmetry,
        5: symmetries.z5_symmetry,
        42: symmetries.ZNSymmetry(42),
        43: symmetries.ZNSymmetry(43),
    }
    assert sym == sym
    assert sym != sym_with_name
    assert sym == other[N]
    assert sym != other[N + 1]
    assert sym != symmetries.u1_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(sym_with_name)
    assert sym.is_same_symmetry(other[N])
    assert not sym.is_same_symmetry(other[N + 1])
    assert not sym.is_same_symmetry(symmetries.u1_symmetry)

    print('checking dual_sector')
    for s in sectors_a:
        assert_array_equal(sym.dual_sector(s), (-s) % N)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(sectors_a), (-sectors_a) % N)


def test_su2_symmetry():
    sym = symmetries.SU2Symmetry()
    spin_1 = np.array([2])
    spin_3_half = np.array([3])
    sym_with_name = symmetries.SU2Symmetry('foo')
    common_checks(sym, example_sector=spin_3_half)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.Group)
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
    assert sym == symmetries.su2_symmetry
    assert sym != symmetries.fermion_parity

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(spin_1), spin_1)
    assert_array_equal(sym.dual_sector(spin_3_half), spin_3_half)

    print('checking dual_sectors')
    assert_array_equal(
        sym.dual_sectors(np.stack([spin_1, spin_3_half])),
        np.stack([spin_1, spin_3_half])
    )


def test_fermion_parity():
    sym = symmetries.FermionParity()
    even = np.array([0])
    odd = np.array([1])
    common_checks(sym, example_sector=odd)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.Group)
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
    assert sym == symmetries.fermion_parity
    assert sym != symmetries.no_symmetry
    assert sym != symmetries.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert not sym.is_same_symmetry(symmetries.no_symmetry)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(odd), odd)

    print('checking dual_sectors')
    assert_array_equal(sym.dual_sectors(np.stack([odd, even, odd])),
                       np.stack([odd, even, odd]))


# TODO VectorSpace, ProductSpace
# TODO test VectorSpace.is_trivial
