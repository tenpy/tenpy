# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
from tenpy.models import spins_nnn
from test_model import check_general_model
from tenpy.models.model import NearestNeighborModel

from tenpy.algorithms.exact_diag import ExactDiag


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


def test_SpinChainNNN_comparison():
    model_pars = {
        'hz': 0.5,
        'Jx': -2.,
        'Jy': -2.,
        'Jz': 0.4,
        'L': 3,
        'conserve': 'Sz',
        'bc_MPS': 'finite'
    }
    M1 = spins_nnn.SpinChainNNN(model_pars.copy())
    model_pars['L'] = 2 * model_pars['L']
    M2 = spins_nnn.SpinChainNNN2(model_pars.copy())
    M2.group_sites(2)
    M2nn = NearestNeighborModel.from_MPOModel(M2)
    ED1 = ExactDiag(M1)
    ED2 = ExactDiag(M2)
    ED2nn = ExactDiag(M2nn)
    ED1.build_full_H_from_mpo()
    ED2.build_full_H_from_mpo()
    ED2nn.build_full_H_from_bonds()
    assert (ED1.full_H - ED2.full_H).norm() < 1.e-13
    assert (ED1.full_H - ED2nn.full_H).norm() < 1.e-13
