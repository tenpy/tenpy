# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions
from test_model import check_general_model
import pytest


@pytest.mark.slow
def test_HofstadterBosons():
    model_pars = {
        'Lx': 3,
        'Ly': 3,
        'phi': (1, 3),
        'conserve': 'N',
        'U': 0.456,
        'mu': 0.123,
        'Nmax': 1
    }
    check_general_model(HofstadterBosons, model_pars, {
        'bc_MPS': ['finite', 'infinite'],
        'gauge': ['landau_x', 'landau_y'],
    })
    model_pars['gauge'] = 'symmetric'
    model_pars['Lx'] = model_pars['Ly'] = 2
    model_pars['mx'] = model_pars['my'] = 2
    model_pars['phi'] = (1, 4)
    check_general_model(HofstadterBosons, model_pars, {'bc_MPS': ['finite', 'infinite']})


@pytest.mark.slow
def test_HofstadterFermions():
    model_pars = {'Lx': 3, 'Ly': 3, 'phi': (1, 3), 'conserve': 'N', 'v': 0.456, 'mu': 0.123}
    check_general_model(HofstadterFermions, model_pars, {
        'bc_MPS': ['finite', 'infinite'],
        'gauge': ['landau_x', 'landau_y'],
    })
    model_pars['gauge'] = 'symmetric'
    model_pars['Lx'] = model_pars['Ly'] = 4
    model_pars['phi'] = (1, 4)
    model_pars['mx'] = model_pars['my'] = 2
    check_general_model(HofstadterFermions, model_pars, {'bc_MPS': ['finite', 'infinite']})
