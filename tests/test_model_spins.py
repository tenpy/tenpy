# Copyright 2018 TeNPy Developers
from tenpy.models import spins
from test_model import check_general_model


def test_SpinModel():
    check_general_model(spins.SpinModel, {'lattice': "Square"}, {})


def test_SpinChain():
    check_general_model(spins.SpinChain, {}, {
        'conserve': [None, 'parity', 'Sz'],
        'S': [0.5, 1, 2]
    })
    check_general_model(spins.SpinChain, {
        'hz': 2.,
        'Jx': -4.,
        'Jz': -0.4,
        'L': 4
    }, {
        'conserve': [None, 'parity'],
        'bc_MPS': ['finite', 'infinite']
    })
