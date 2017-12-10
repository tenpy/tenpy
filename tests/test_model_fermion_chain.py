from __future__ import division

from tenpy.models.fermion_chain import FermionChain
from test_model import check_general_model


def test_FermionChain():
    check_general_model(FermionChain, {'L': 4}, {
        'conserve': [None, 'parity', 'N'],
        'mu': [0., 0.123],
        'bc_MPS': ['finite', 'infinite']
    })
