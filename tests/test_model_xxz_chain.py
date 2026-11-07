"""test of :class:`tenpy.models.XXZChain`."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import numpy.testing as npt
import pytest
from lasyten.tensors import almost_equal
from test_model import check_general_model

from tenpy.models.xxz_chain import XXZChain, XXZChain2


def _bond_eigenvalues(Hb):
    """Eigenvalues of a two-site bond operator with labels p0, p0*, p1, p1*.

    TODO: replace with tenpy's exact_diag methods once they are ported to lasyten.
    """
    mat = Hb.to_numpy(['p0', 'p1', 'p0*', 'p1*']).reshape(4, 4)
    return np.linalg.eigvalsh(mat)


@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
def test_XXZChain(conserve):
    pars = dict(L=4, Jxx=1.0, Jz=1.0, hz=0.0, bc_MPS='finite', conserve=conserve)
    chain = XXZChain2(pars)
    chain.test_sanity()
    H_bond = chain.get_H_bond()
    for Hb in H_bond[1:]:  # check bond eigenvalues
        W = _bond_eigenvalues(Hb)
        npt.assert_array_almost_equal_nulp(np.sort(W), np.sort([-0.75, 0.25, 0.25, 0.25]), 16**3)
    # now check with non-trivial onsite terms
    pars['hz'] = 0.2
    print('hz =', pars['hz'])

    with pytest.raises(NotImplementedError, match='port models'):
        chain = XXZChain(pars)
    pytest.skip('XXZChain (pedagogical exmaple) not yet ported')
    chain.test_sanity()
    Hb = H_bond[2]  # the only central bonds: boundaries have different hz.
    W = _bond_eigenvalues(Hb)
    print(W)
    npt.assert_array_almost_equal_nulp(
        np.sort(W),
        np.sort([-0.75, 0.25 - 2 * 0.5 * 0.5 * pars['hz'], 0.25, 0.25 + 2.0 * 0.5 * 0.5 * pars['hz']]),
        16**3,
    )

    for L in [2, 3, 4, 5, 6]:
        print('L =', L)
        pars['L'] = L
        chain = XXZChain(pars)
        assert len(H_bond) == L
        # for uniform couplings, all "bulk" bonds (not touching an open boundary) agree
        for Hb in H_bond[2 : L - 1]:
            assert almost_equal(Hb, H_bond[2])
    pars['Jxx'] = 0.0
    chain = XXZChain(pars)
    chain.test_sanity()


@pytest.mark.parametrize('conserve', ['Sz', 'parity', 'None'])
def test_XXZChain_general(conserve, tol=1.0e-14):
    # only the Heisenberg point has Sz symmetry
    Jz_list = [1.0] if conserve == 'Sz' else [0.0, 1.0, 2.0]

    with pytest.raises(NotImplementedError, match='port models'):
        check_general_model(
            XXZChain,
            dict(L=4, Jxx=1.0, hz=0.0, bc_MPS='finite', conserve=conserve),
            dict(Jz=Jz_list, hz=[0.0, 0.2]),
        )
        # once XXZChain is ported, just remove the context manager

    check_general_model(
        XXZChain2,
        dict(L=4, Jxx=1.0, hz=0.0, bc_MPS='finite', conserve=conserve),
        dict(Jz=Jz_list, hz=[0.0, 0.2]),
    )

    model_param = dict(L=3, Jxx=1.0, Jz=1.0 if conserve == 'Sz' else 1.5, hz=0.25, bc_MPS='finite')
    with pytest.raises(NotImplementedError, match='port models'):
        m1 = XXZChain(model_param)
        # once XXZChain is ported, just remove the context manager

        m2 = XXZChain2(model_param)
        for Hb1, Hb2 in zip(m1.H_bond, m2.H_bond):
            if Hb1 is None:
                assert Hb2 is None
                continue
            assert almost_equal(Hb1, Hb2)
