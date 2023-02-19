"""A collection of tests for tenpy.linalg.symmetries."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from tenpy.linalg import symmetries


def test_no_symmetry():
    sym = symmetries.NoSymmetry()
    print('checking valid sectors')
    assert sym.is_valid_sector(None)
    assert not sym.is_valid_sector(0)
    assert not sym.is_valid_sector(1)
    assert not sym.is_valid_sector(-1)
    assert not sym.is_valid_sector(1.)
    assert not sym.is_valid_sector([])
    assert not sym.is_valid_sector([None])
    assert not sym.is_valid_sector([5, 3, 1])

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

    print('checking valid sectors')
    assert not sym.is_valid_sector(None)
    assert not sym.is_valid_sector(0)
    assert not sym.is_valid_sector(1)
    assert not sym.is_valid_sector(-1)
    assert not sym.is_valid_sector(1.)
    assert not sym.is_valid_sector([])
    assert not sym.is_valid_sector([None])
    assert sym.is_valid_sector([5, 3, 1])  # e.g. spin 5/2 , 3 particles , odd parity ("fermionic")

    print('checking fusion')
    pass  # FIXME

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
    
    print('checking valid sectors')
    assert not sym.is_valid_sector(None)
    assert sym.is_valid_sector(0)
    assert sym.is_valid_sector(1)
    assert not sym.is_valid_sector(1.)
    assert sym.is_valid_sector(-1)
    assert not sym.is_valid_sector([])
    assert not sym.is_valid_sector([5, 3, 1])

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
