# Copyright 2018 TeNPy Developers
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions
from test_model import check_general_model
from nose.plugins.attrib import attr


@attr('slow')
def test_HofstadterBosons():
    check_general_model(HofstadterBosons, {'Lx':9, 'Ly':9, 'phi':(1,9)}, {
        'conserve': [None, 'parity', 'N'],
        'U': [0., 0.123],
        'bc_MPS': ['finite', 'infinite'],
        'gauge': ['landau_x', 'landau_y', 'symmetric'],
        'mu':[0, 0.123],
        'Nmax': [1, 3],
    })

@attr('slow')
def test_HofstadterFermions():
    check_general_model(HofstadterFermions, {'Lx':9, 'Ly':9, 'phi':(1,9)}, {
        'conserve': [None, 'parity', 'N'],
        'bc_MPS': ['finite', 'infinite'],
        'gauge': ['landau_x', 'landau_y', 'symmetric'],
        'mu':[0, 0.123]
    })
