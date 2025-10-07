# Copyright (C) TeNPy Developers, Apache license
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions
from tenpy.algorithms.exact_diag import ExactDiag
from test_model import check_general_model
import numpy as np
import pytest


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


# comparison values for test_ED_spectrum_HofstadterFermions
# 3x3 system, keys are (bc_x, bc_y)
_HofstadterFermions_spectra = {
    ('periodic', 'periodic'): [-7.19852401, -5.14005494, -5.14005494, -5.14005494, -4.99137522,
                               -4.99137522, -4.99137522, -3.80818865, -3.80818865, -3.80818865],
    ('periodic', 'open'): [-6.50953401, -5.09804883, -5.09804883, -4.60171247, -4.56760104,
                           -4.56760104, -4.30320927, -3.94132612, -3.94132612, -3.61801836],
    ('open', 'periodic'): [-6.50953401, -5.09804883, -5.09804883, -4.60171247, -4.56760104,
                           -4.56760104, -4.30320927, -3.94132612, -3.94132612, -3.61801836],
    ('open', 'open'): [-5.30259582, -5.11380114, -4.49492842, -4.17538168, -4.01502519,
                       -3.93726486, -3.90488366, -3.84296889, -3.74585977, -3.65848589],
}


@pytest.mark.parametrize('bc_x', ['periodic', 'open'])
@pytest.mark.parametrize('bc_y', ['periodic', 'open'])
@pytest.mark.parametrize('gauge', ['landau_x', 'landau_y', 'symmetric'])
def test_ED_spectrum_HofstadterFermions(bc_x, bc_y, gauge):
    model_params = dict(v=1, Lx=3, Ly=3, bc_x=bc_x, bc_y=bc_y, conserve='N', gauge=gauge)
    model = HofstadterFermions(model_params)
    engine = ExactDiag(model)
    engine.build_full_H_from_mpo()
    engine.full_diagonalization()
    low_energy_spectrum = np.sort(engine.E)[:10]
    expect = _HofstadterFermions_spectra[bc_x, bc_y]
    assert np.allclose(low_energy_spectrum, expect)


# comparison values for test_ED_spectrum_HofstadterBosons
# 3x3 system, keys are (bc_x, bc_y)
_HofstadterBosons_spectra = {
    ('periodic', 'periodic'): [-7.10623032, -7.10623032, -7.10623032, -7.10623032, -7.10623032,
                               -7.10623032, -6.87298335, -6.87298335, -6.72092373, -6.72092373],
    ('periodic', 'open'): [-6.68491519, -6.68491519, -6.54147708, -6.54147708, -6.54147708,
                           -6.54147708, -6.3261335 , -6.3261335 , -5.70748017, -5.70748017],
    ('open', 'periodic'): [-6.68491519, -6.68491519, -6.54147708, -6.54147708, -6.54147708,
                           -6.54147708, -6.3261335 , -6.3261335 , -5.70748017, -5.70748017],
    ('open', 'open'): [-5.7278833 , -5.7278833 , -5.53489416, -5.53489416, -5.08982148,
                       -5.08982148, -4.93255091, -4.93255091, -4.92485095, -4.92485095],
}


@pytest.mark.parametrize('bc_x', ['periodic', 'open'])
@pytest.mark.parametrize('bc_y', ['periodic', 'open'])
@pytest.mark.parametrize('gauge', ['landau_x', 'landau_y', 'symmetric'])
def test_ED_spectrum_HofstadterBosons(bc_x, bc_y, gauge):
    model_params = dict(U=1, Lx=3, Ly=3, bc_x=bc_x, bc_y=bc_y, conserve='N', Nmax=1, gauge=gauge)
    model = HofstadterBosons(model_params)
    engine = ExactDiag(model)
    engine.build_full_H_from_mpo()
    engine.full_diagonalization()
    low_energy_spectrum = np.sort(engine.E)[:10]
    expect = _HofstadterBosons_spectra[bc_x, bc_y]
    assert np.allclose(low_energy_spectrum, expect)
