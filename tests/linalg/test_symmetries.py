# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from tenpy.linalg import symmetries


def sampled_zip(sequence, num_copies: int, num_samples: int, np_random=np.random, accept_fewer=True):
    """Generate a given number of random samples from the zip of multiple sequences"""
    len_sequence = len(sequence)
    num_combinations = len_sequence ** num_copies
    if num_samples > num_combinations:
        if accept_fewer:
            num_samples = num_combinations
        else:
            raise ValueError(f'Can not generate {num_samples} samples.')
    for idx in np_random.choice(num_combinations, size=num_samples, replace=False):
        sample = []
        for _ in range(num_copies):
            sample.append(sequence[idx % len_sequence])
            idx = idx // len_sequence
        assert idx == 0
        yield tuple(sample)


def sample_from(sequence, num_samples: int, accept_fewer: bool = False, np_random=np.random):
    if num_samples > len(sequence):
        if accept_fewer:
            num_samples = len(sequence)
        else:
            raise ValueError(f'Can not generate {num_samples} samples from sequence of len {len(sequence)}')
    for n in np_random.choice(len(sequence), size=num_samples, replace=False):
        yield sequence[n]


def sample_offdiagonal(sequence, num_samples: int, accept_fewer: bool = False, np_random=np.random):
    """Sample pairs ``(sequence[n], sequence[m])`` with ``n != m``."""
    len_sequence = len(sequence)
    num_combinations = len_sequence * (len_sequence - 1)
    if num_samples > num_combinations:
        if accept_fewer:
            num_samples = num_combinations
        else:
            raise ValueError(f'Can not generate {num_samples} samples.')
    for i in np_random.choice(num_combinations, size=num_samples, replace=False):
        idx1, idx2 = divmod(i, len_sequence)
        # now   0 <= idx1 < len-1   and 0 <= idx2 < len
        if idx1 >= idx2:
            idx1 += 1
        yield sequence[idx1], sequence[idx2]


def common_checks(sym: symmetries.Symmetry, example_sectors, np_random):
    """Common consistency checks to be performed on a symmetry instance.

    Assumes example_sectors are duplicate free.
    """
    example_sectors = np.unique(example_sectors, axis=0)
    
    assert sym.trivial_sector.shape == (sym.sector_ind_len,)
    assert sym.is_valid_sector(sym.trivial_sector)
    assert not sym.is_valid_sector(np.zeros(shape=(sym.sector_ind_len + 2), dtype=int))
    # objects which are not 1D arrays should not be valid sectors
    for invalid_sector in [0, 1, 42., None, False, 'foo', [0], ['foo'], [None], (), [],
                           np.zeros((1, 1), dtype=int)]:
        assert not sym.is_valid_sector(invalid_sector)
    assert sym.qdim(sym.trivial_sector) in [1, 1.]
    assert sym.num_sectors == np.inf or (isinstance(sym.num_sectors, int) and sym.num_sectors > 0)

    # check all_sectors
    if sym.num_sectors < np.inf:
        all_sectors = sym.all_sectors()
        assert all_sectors.shape == (sym.num_sectors, sym.sector_ind_len)
        for s in all_sectors:
            assert sym.is_valid_sector(s)

    # string representations : just check if they run
    _ = sym.sector_str(sym.trivial_sector)
    _ = repr(sym)
    _ = str(sym)

    # defining property of trivial sector
    for a in example_sectors:
        fusion = sym.fusion_outcomes(a, sym.trivial_sector)
        assert len(fusion) == 1
        assert_array_equal(fusion[0], a)

    # trivial sector is its own dual
    assert_array_equal(sym.dual_sector(sym.trivial_sector), sym.trivial_sector)

    # check fusion tensors, if available
    if sym.has_fusion_tensor:
        check_fusion_tensor(sym, example_sectors, np_random)
        check_symbols_via_fusion_tensors(sym, example_sectors, np_random)
    else:
        with pytest.raises(NotImplementedError, match='fusion_tensor is not implemented for'):
            _ = sym.fusion_tensor(sym.trivial_sector, sym.trivial_sector, sym.trivial_sector)

    # check N symbol
    for a in example_sectors:
        # duality (diagonal part)
        assert sym.n_symbol(a, sym.dual_sector(a), sym.trivial_sector) == 1
        # left and right unitor (diagonal part)
        assert sym.n_symbol(a, sym.trivial_sector, a) == 1
        assert sym.n_symbol(sym.trivial_sector, a, a) == 1
    for a, b in sample_offdiagonal(example_sectors, num_samples=10, accept_fewer=True, np_random=np_random):
        # duality (off-diagonal part)
        b_dual = sym.dual_sector(b)
        if not np.all(a == b_dual):
            assert sym.n_symbol(a, b_dual, sym.trivial_sector) == 0
        # left and right unitor (off-diagonal part)
        assert sym.n_symbol(a, sym.trivial_sector, b) == 0
        assert sym.n_symbol(sym.trivial_sector, a, b) == 0
    for a, b, c in sampled_zip(example_sectors, num_copies=3, num_samples=10, np_random=np_random):
        #  associativity constraint  :  \sum_e N(a, b, e) N(e, c, d) == \sum_f N(b, c, d) N(a, f, d)
        pass  # TODO probably want a clever way to sample some d such that fusion is allowed.

    # check F symbol
    #   TODO:
    #    - correct shape
    #    - unitary
    #    - triangle equation
    #    - pentagon equation

    # check R symbol
    #   TODO:
    #    - correct shape
    #    - unitary (i.e. just phases)
    #    - hexagon equation
    #    - consistency with twist (when implemented)

    # check C symbol
    #   TODO:
    #    - correct shape
    #    - unitary

    # check B symbol
    #   TODO:
    #    - correct shape
    #    - normalization
    #    - snake equation
    
    # check derived topological data vs the fallback implementations
    methods_to_check = dict(
        frobenius_schur=1,
        qdim=1,
        _b_symbol=3,
        _c_symbol=6,
    )  # values: number of sectors that method takes as arguments
    for method_name, num_sectors in methods_to_check.items():
        # equivalent too e.g. ``sym.qdim(a) == Symmetry.qdim(sym, a)``.
        sym_method = getattr(type(sym), method_name)
        fallback_method = getattr(symmetries.Symmetry, method_name)
        if sym_method is fallback_method:
            continue  # not overridden -> nothing to check
        print(f'checking {method_name} vs fallback implementation')
        for sectors in sampled_zip(example_sectors, num_copies=num_sectors, num_samples=5,
                                   np_random=np_random):
            assert_array_almost_equal(sym_method(sym, *sectors), fallback_method(sym, *sectors))

    # check braiding style
    #   TODO:
    #    - if bosonic, check that twists are +1
    #    - if <= fermionic, check that double braid is identity via R symbols, also twists are +-1

    # check fusion style
    #   TODO:
    #    - if single: check lengths of fusion_outcomes
    #    - if <= multiple_unique: check N symbols to be in [0, 1]


def check_fusion_tensor(sym: symmetries.Symmetry, example_sectors, np_random):
    """Checks if the fusion tensor of a given symmetry behaves as expected.

    Subroutine of `common_checks`.
    The ``example_sectors`` should be duplicate free.

    We check:
    - correct shape
    - orthonormality
    - completeness
    - relationship to left/right unitor

    TODO should check:
    - relationship to cup
    """
    for a, b in sampled_zip(example_sectors, num_copies=2, num_samples=10, np_random=np_random):
        d_a = sym.sector_dim(a)
        d_b = sym.sector_dim(b)
        fusion_outcomes = sym.fusion_outcomes(a, b)

        assert len(fusion_outcomes) < 20  # otherwise we should not loop over all i guess?

        completeness_res = np.zeros((d_a, d_b, d_a, d_b))
        # iterate over coupled sectors and combine several checks
        for c in fusion_outcomes:
            d_c = sym.sector_dim(c)
            X_abc = sym.fusion_tensor(a, b, c)
            N_abc = sym.n_symbol(a, b, c)
            Y_abc = np.conj(X_abc)

            # correct shape?
            assert X_abc.shape == (N_abc, d_a, d_b, d_c)

            # orthonormal?
            res = np.tensordot(Y_abc, X_abc, [[1, 2], [1, 2]])  # [mu', m_c', mu, m_c]
            expect = np.eye(N_abc)[:, None, :, None] * np.eye(sym.sector_dim(c))[None, :, None, :]
            assert_array_almost_equal(res, expect)

            # accumulate projector for checking completeness after the loop
            completeness_res += np.tensordot(X_abc, Y_abc, [[0, 3], [0, 3]])

        # complete?
        eye_a = np.eye(d_a)[:, None, :, None]
        eye_b = np.eye(d_b)[None, :, None, :]
        assert_array_almost_equal(completeness_res, eye_a * eye_b)
            
        # remains: orthonormality for different sectors
        for c, d in sample_offdiagonal(fusion_outcomes, num_samples=5, accept_fewer=True, np_random=np_random):
            d_c = sym.sector_dim(c)
            d_d = sym.sector_dim(d)
            N_abc = sym.n_symbol(a, b, c)
            N_abd = sym.n_symbol(a, b, d)
            Y_abc = np.conj(sym.fusion_tensor(a, b, c))
            X_abd = sym.fusion_tensor(a, b, d)
            res = np.tensordot(Y_abc, X_abd, [[1, 2], [1, 2]])  # [mu', m_c', mu, m_d]
            expect = np.zeros((N_abc, d_c, N_abd, d_d), dtype=res.dtype)
            assert_array_almost_equal(res, expect)

    for a in example_sectors:
        d_a = sym.sector_dim(a)
    
        # reduces to left/right unitor if one input is trivial?
        X_aua = sym.fusion_tensor(a, sym.trivial_sector, a)
        assert_array_almost_equal(X_aua, np.eye(d_a, dtype=X_aua.dtype)[None, :, None, :])
        X_uaa = sym.fusion_tensor(sym.trivial_sector, a, a)
        assert_array_almost_equal(X_uaa, np.eye(d_a, dtype=X_uaa.dtype)[None, None, :, :])

        # relationship to cup
        # TODO unclear how to construct the correct cup. np.eye(d_a) is probably wrong...?
        # cup = np.eye(d_a)
        # assert_array_almost_equal(np.tensordot(cup, cup, 2), d_a)  # check normalization of cup
        # X_a_abar_u = sym.fusion_tensor(a, sym.dual_sector(a), sym.trivial_sector)
        # assert_array_almost_equal(X_a_abar_u, cup[None, :, :, None] / np.sqrt(d_a))


def check_symbols_via_fusion_tensors(sym: symmetries.Symmetry, example_sectors, np_random):
    """Check the defining properties of the symbols with explicit fusion tensors.
    
    Subroutine of `common_checks`.
    The ``example_sectors`` should be duplicate free.

    We check:

    TODO:
    - F symbol
    - R symbol
    - C symbol
    - B symbol
    """
    pass


def test_no_symmetry(np_random):
    sym = symmetries.NoSymmetry()
    s = np.array([0])
    common_checks(sym, example_sectors=s[np.newaxis, :], np_random=np_random)

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.GroupSymmetry)
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


@pytest.mark.xfail('Topological data not implemented.')
def test_product_symmetry(np_random):
    sym = symmetries.ProductSymmetry([
        symmetries.SU2Symmetry(), symmetries.U1Symmetry(), symmetries.FermionParity()
    ])
    sym_with_name = symmetries.ProductSymmetry([
        symmetries.SU2Symmetry('foo'), symmetries.U1Symmetry('bar'), symmetries.FermionParity()
    ])
    s1 = np.array([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")
    s2 = np.array([3, 2, 0])  # e.g. spin 3/2 , 2 particles , even parity ("bosonic")
    common_checks(sym, example_sectors=np.array([s1, s2]), np_random=np_random)

    u1_z3 = symmetries.u1_symmetry * symmetries.z3_symmetry
    common_checks(u1_z3, example_sectors=np.array([[42, 1], [-1, 2], [-2, 0]]), np_random=np_random)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.GroupSymmetry)
    assert not sym.is_abelian
    assert isinstance(u1_z3, symmetries.AbelianGroup)
    assert isinstance(u1_z3, symmetries.GroupSymmetry)
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


def test_u1_symmetry(np_random):
    sym = symmetries.U1Symmetry()
    sym_with_name = symmetries.U1Symmetry('foo')
    s_0 = np.array([0])
    s_1 = np.array([1])
    s_neg1 = np.array([-1])
    s_2 = np.array([2])
    s_42 = np.array([42])
    common_checks(sym, example_sectors=np.array([s_0, s_1, s_neg1, s_2, s_42]), np_random=np_random)

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.GroupSymmetry)
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
def test_ZN_symmetry(N, np_random):
    sym = symmetries.ZNSymmetry(N=N)
    sym_with_name = symmetries.ZNSymmetry(N, descriptive_name='foo')
    sectors_a = np.array([0, 1, 2, 10])[:, None] % N
    sectors_b = np.array([0, 1, 3, 11])[:, None] % N
    common_checks(sym, example_sectors=sectors_a, np_random=np_random)

    print('instancecheck and is_abelian')
    assert isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.GroupSymmetry)
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


@pytest.mark.skip('SU(2) data not yet implemented')
def test_su2_symmetry(np_random):
    sym = symmetries.SU2Symmetry()
    common_checks(sym, example_sectors=np.array([[0], [3], [5], [2], [1], [121]]), np_random=np_random)
    
    spin_1 = np.array([2])
    spin_3_half = np.array([3])
    sym_with_name = symmetries.SU2Symmetry('foo')

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert isinstance(sym, symmetries.GroupSymmetry)
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


def test_fermion_parity(np_random):
    sym = symmetries.FermionParity()
    even = np.array([0])
    odd = np.array([1])
    common_checks(sym, example_sectors=np.array([even, odd]), np_random=np_random)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.GroupSymmetry)
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


@pytest.mark.xfail('C symbol seems to fail.')
@pytest.mark.parametrize('handedness', ['left', 'right'])
def test_fibonacci_grading(handedness, np_random):
    sym = symmetries.FibonacciGrading(handedness)
    vac = np.array([0])
    tau = np.array([1])
    common_checks(sym, example_sectors=sym.all_sectors(), np_random=np_random)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.GroupSymmetry)
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
    assert (sym == symmetries.fibonacci_grading) == (handedness == 'left')
    assert sym != symmetries.no_symmetry
    assert sym != symmetries.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(symmetries.fibonacci_grading) == (handedness == 'left')
    assert not sym.is_same_symmetry(symmetries.no_symmetry)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(tau), tau)


@pytest.mark.xfail('C symbol seems to fail.')
@pytest.mark.parametrize('nu', [*range(1, 16, 2)])
def test_ising_grading(nu, np_random):
    sym = symmetries.IsingGrading(nu)
    vac = np.array([0])
    anyon = np.array([1])
    fermion = np.array([2])
    common_checks(sym, example_sectors=sym.all_sectors(), np_random=np_random)

    print('instancecheck and is_abelian')
    assert not isinstance(sym, symmetries.AbelianGroup)
    assert not isinstance(sym, symmetries.GroupSymmetry)
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
    assert (sym == symmetries.ising_grading) == (nu == 1)
    assert sym != symmetries.no_symmetry
    assert sym != symmetries.su2_symmetry

    print('checking is_same_symmetry')
    assert sym.is_same_symmetry(sym)
    assert sym.is_same_symmetry(symmetries.ising_grading) == (nu == 1)
    assert not sym.is_same_symmetry(symmetries.no_symmetry)
    assert not sym.is_same_symmetry(symmetries.su2_symmetry)

    print('checking dual_sector')
    assert_array_equal(sym.dual_sector(anyon), anyon)
    assert_array_equal(sym.dual_sector(fermion), fermion)
