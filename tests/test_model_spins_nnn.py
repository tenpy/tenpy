# Copyright 2018 TeNPy Developers
from tenpy.models import spins_nnn
from test_model import check_general_model


def test_SpinChainNNN():
    check_general_model(spins_nnn.SpinChainNNN, {
        'hz': 0.5,
        'Jx': -2.,
        'Jy': -2.,
        'Jz': 0.4,
        'L': 4
    }, {
        'conserve': [None, 'Sz'],
        'bc_MPS': ['finite', 'infinite']
    })


def test_SpinChainNNN2():
    check_general_model(spins_nnn.SpinChainNNN2, {
        'hz': 0.5,
        'Jx': -2.,
        'Jy': -2.,
        'Jz': 0.4,
        'L': 4
    }, {
        'conserve': [None, 'Sz'],
        'bc_MPS': ['finite', 'infinite']
    })
