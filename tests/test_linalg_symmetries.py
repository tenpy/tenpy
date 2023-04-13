"""A collection of tests for tenpy.linalg.symmetries."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np

from tenpy.linalg import symmetries


invalid_sectors = [0, 1, 42., None, False, 'foo', [0], ['foo'], [None], (), []]


def common_checks(sym: symmetries.Symmetry):
    # common consistency checks to be performed on a symmetry instance
    assert sym.is_valid_sector(sym.trivial_sector)
    for s in invalid_sectors:
        assert not sym.is_valid_sector(s)


def test_no_symmetry():
    sym = symmetries.NoSymmetry()
    common_checks(sym)
    
    print('checking valid sectors')
    assert sym.is_valid_sector(np.array([0]))
    assert not sym.is_valid_sector(np.array([0, 0]))

    print('checking fusion')
    pass  # FIXME

    print('checking sector dimensions')
    pass  # FIXME

    print('checking duality')
    pass  # FIXME

    print('checking equality')
    assert sym == symmetries.no_symmetry
    assert sym != symmetries.u1_symmetry
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry


def test_product_symmetry():

    print('checking creation via __mul__')
    sym = symmetries.su2_symmetry * symmetries.u1_symmetry * symmetries.fermion_parity
    common_checks(sym)

    print('checking valid sectors')
    s1 = np.array([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")
    s2 = np.array([3, 2, 0])  # e.g. spin 3/2 , 2 particles , even parity ("bosonic")
    assert sym.is_valid_sector(s1)
    assert sym.is_valid_sector(s2)
    assert not sym.is_valid_sector(np.array([-1, 2, 0]))  # negative spin is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 42]))  # parity not in [0, 1] is invalid
    assert not sym.is_valid_sector(np.array([3, 2, 0, 1]))  # too many entries

    print('checking fusion')
    outcomes = sym.fusion_outcomes(s1, s2)
    # spin 3/2 and 5/2 can fuse to [1, 2, 3, 4]  ;  U(1) charges  3 + 2 = 5  ;  fermion charges 1 + 0 = 1
    expect = np.array([[2, 5, 1], [4, 5, 1], [6, 5, 1], [8, 5, 1]])
    assert np.all(outcomes == expect)

    print('checking sector dimensions')
    pass  # FIXME

    print('checking duality')
    pass  # FIXME

    print('checking equality')
    assert sym == symmetries.ProductSymmetry([symmetries.SU2Symmetry(), symmetries.U1Symmetry(), 
                                             symmetries.FermionParity()])
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry
    assert sym != symmetries.no_symmetry


def test_u1_symmetry():
    sym = symmetries.U1Symmetry()
    common_checks(sym)
    
    print('checking valid sectors')
    for charge in [0, 1, -1, 42]:
        assert sym.is_valid_sector(np.array([charge]))
    assert not sym.is_valid_sector(np.array([0, 0]))
    
    print('checking fusion')
    pass  # FIXME

    print('checking sector dimensions')
    pass  # FIXME

    print('checking duality')
    pass  # FIXME

    print('checking equality')
    assert sym == symmetries.u1_symmetry
    assert sym != symmetries.no_symmetry
    assert sym != symmetries.su2_symmetry * symmetries.u1_symmetry


# TODO ZN, SU2, Fermion

# TODO VectorSpace, ProductSpace
# TODO test VectorSpace.is_trivial
