# Copyright 2018 TeNPy Developers

from tenpy.models.tf_ising import TFIModel, TFIChain
from test_model import check_general_model
from nose.plugins.attrib import attr


def test_TFIChain_general():
    check_general_model(TFIChain, dict(L=4, J=1., bc_MPS='finite'), {
        'conserve': [None, 'parity'],
        'g': [0., 0.2]
    })


@attr('slow')
def test_TFIModel2D_general():
    check_general_model(TFIModel, dict(Lx=2, J=1., g=0.1), {
        'Ly': [2, 3],
        'bc_MPS': ['finite', 'infinite'],
        'bc_y': ['ladder', 'cylinder'],
        'lattice': ['Square', 'Honeycomb', 'Kagome']
    })
