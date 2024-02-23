# Copyright (C) TeNPy Developers, GNU GPLv3
import numpy as np
from tenpy.models import spins
from test_model import check_general_model


def test_SpinModel():
    check_general_model(spins.SpinModel,
                        {'lattice': "Square", 'Lx': 2, 'Ly': 3, 'sort_charge': True},
                        {})


def test_SpinChain():
    check_general_model(spins.SpinChain, {'sort_charge': True}, {
        'conserve': [None, 'parity', 'Sz'],
        'S': [0.5, 1, 2]
    })
    check_general_model(spins.SpinChain, {
        'hz': 2.,
        'Jx': -4.,
        'Jz': -0.4,
        'L': 4,
        'sort_charge': True,
    }, {
        'conserve': [None, 'parity'],
        'bc_MPS': ['finite', 'infinite']
    })

def test_DipolarSpinChain():
    # check dipolar charges for one specific case
    model = spins.DipolarSpinChain(dict(S=1, J4=1, conserve='dipole'))
    expect_2Sz = np.array([-2, 0, 2])
    for i, s in enumerate(model.lat.mps_sites()):
        expect_dipole = i * expect_2Sz
        expect_charges = np.array([expect_2Sz, expect_dipole]).T
        assert np.all(s.leg.charges == expect_charges)
    # check general properties for many cases
    check_general_model(spins.DipolarSpinChain, {}, {'J4': [0, 1], 'conserve': ['dipole', 'Sz', None]})
    
