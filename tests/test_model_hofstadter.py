# Copyright 2018 TeNPy Developers
from tenpy.models.hofstadter import HofstadterBosons
from test_model import check_general_model
from nose.plugins.attrib import attr

# TODO (LS) for both tests, add more parameters

@attr('slow')
def test_HofstadterBosons():
    check_general_model(HofstadterBosons, {'Lx': 3, 'U': 0.123}, {
        'conserve': [None, 'N'],
        'bc_MPS': ['finite', 'infinite']
    })

@attr('slow')
def test_HofstadterFermions():
    check_general_model(HofstadterBosons, {'Lx': 3, 'U': 0.123}, {
        'conserve': [None, 'N'],
        'bc_MPS': ['finite', 'infinite']
    })
