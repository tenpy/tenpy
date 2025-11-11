# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import pytest
from test_model import check_general_model

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions, hopping_phases


def test_HofstadterBosons():
    model_pars = {'Lx': 3, 'Ly': 3, 'phi': (1, 3), 'conserve': 'N', 'U': 0.456, 'mu': 0.123, 'Nmax': 1}
    check_general_model(
        HofstadterBosons,
        model_pars,
        {
            'bc_MPS': ['finite', 'infinite'],
            'gauge': ['landau_x', 'landau_y'],
        },
    )
    model_pars['gauge'] = 'symmetric'
    model_pars['Lx'] = model_pars['Ly'] = 6
    check_general_model(HofstadterBosons, model_pars, {'bc_MPS': ['finite', 'infinite']})


def test_HofstadterFermions():
    model_pars = {'Lx': 3, 'Ly': 3, 'phi': (1, 3), 'conserve': 'N', 'v': 0.456, 'mu': 0.123}
    check_general_model(
        HofstadterFermions,
        model_pars,
        {
            'bc_MPS': ['finite', 'infinite'],
            'gauge': ['landau_x', 'landau_y'],
        },
    )
    model_pars['gauge'] = 'symmetric'
    model_pars['Lx'] = model_pars['Ly'] = 6
    check_general_model(HofstadterFermions, model_pars, {'bc_MPS': ['finite', 'infinite']})


# comparison values for test_ED_spectrum_HofstadterFermions
# 3x3 system, keys are (bc_x, bc_y)
_HofstadterFermions_spectra = {
    ('periodic', 'periodic'): [
        -7.19852401,
        -5.14005494,
        -5.14005494,
        -5.14005494,
        -4.99137522,
        -4.99137522,
        -4.99137522,
        -3.80818865,
        -3.80818865,
        -3.80818865,
    ],
    ('periodic', 'open'): [
        -6.50953401,
        -5.09804883,
        -5.09804883,
        -4.60171247,
        -4.56760104,
        -4.56760104,
        -4.30320927,
        -3.94132612,
        -3.94132612,
        -3.61801836,
    ],
    ('open', 'periodic'): [
        -6.50953401,
        -5.09804883,
        -5.09804883,
        -4.60171247,
        -4.56760104,
        -4.56760104,
        -4.30320927,
        -3.94132612,
        -3.94132612,
        -3.61801836,
    ],
    ('open', 'open'): [
        -5.30259582,
        -5.11380114,
        -4.49492842,
        -4.17538168,
        -4.01502519,
        -3.93726486,
        -3.90488366,
        -3.84296889,
        -3.74585977,
        -3.65848589,
    ],
}


@pytest.mark.parametrize('bc_x', ['periodic', 'open'])
@pytest.mark.parametrize('bc_y', ['periodic', 'open'])
@pytest.mark.parametrize('gauge', ['landau_x', 'landau_y', 'symmetric'])
def test_ED_spectrum_HofstadterFermions(bc_x, bc_y, gauge):
    model_params = dict(v=1, Lx=3, Ly=3, bc_x=bc_x, bc_y=bc_y, conserve='N', gauge=gauge)

    if gauge == 'symmetric' and 'periodic' in [bc_x, bc_y]:
        # smallest non-trivial system where symmetric gauge is commensurate is 4x4.
        # but then ED is too slow for a quick test...
        with pytest.raises(ValueError, match='incommensurate'):
            _ = HofstadterFermions(model_params)
        return

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
    ('periodic', 'periodic'): [
        -7.10623032,
        -7.10623032,
        -7.10623032,
        -7.10623032,
        -7.10623032,
        -7.10623032,
        -6.87298335,
        -6.87298335,
        -6.72092373,
        -6.72092373,
    ],
    ('periodic', 'open'): [
        -6.68491519,
        -6.68491519,
        -6.54147708,
        -6.54147708,
        -6.54147708,
        -6.54147708,
        -6.3261335,
        -6.3261335,
        -5.70748017,
        -5.70748017,
    ],
    ('open', 'periodic'): [
        -6.68491519,
        -6.68491519,
        -6.54147708,
        -6.54147708,
        -6.54147708,
        -6.54147708,
        -6.3261335,
        -6.3261335,
        -5.70748017,
        -5.70748017,
    ],
    ('open', 'open'): [
        -5.7278833,
        -5.7278833,
        -5.53489416,
        -5.53489416,
        -5.08982148,
        -5.08982148,
        -4.93255091,
        -4.93255091,
        -4.92485095,
        -4.92485095,
    ],
}


@pytest.mark.parametrize('bc_x', ['periodic', 'open'])
@pytest.mark.parametrize('bc_y', ['periodic', 'open'])
@pytest.mark.parametrize('gauge', ['landau_x', 'landau_y', 'symmetric'])
def test_ED_spectrum_HofstadterBosons(bc_x, bc_y, gauge):
    model_params = dict(U=1, Lx=3, Ly=3, bc_x=bc_x, bc_y=bc_y, conserve='N', Nmax=1, gauge=gauge)

    if gauge == 'symmetric' and 'periodic' in [bc_x, bc_y]:
        # smallest non-trivial system where symmetric gauge is commensurate is 4x4.
        # but then ED is too slow for a quick test...
        with pytest.raises(ValueError, match='incommensurate'):
            _ = HofstadterBosons(model_params)
        return

    model = HofstadterBosons(model_params)
    engine = ExactDiag(model)
    engine.build_full_H_from_mpo()
    engine.full_diagonalization()
    low_energy_spectrum = np.sort(engine.E)[:10]
    expect = _HofstadterBosons_spectra[bc_x, bc_y]
    assert np.allclose(low_energy_spectrum, expect)


@pytest.mark.parametrize(
    'lx, ly, p, q, commensurate_gauges',
    [
        (6, 6, 1, 3, 'all'),  # A
        (18, 18, 4, 9, 'all'),  # B
        (2, 5, 4, 9, None),  # C
        (2, 9, 4, 9, 'landau_y'),  # D
    ],
    ids='ABCD',
)
@pytest.mark.parametrize('pbc_x', [True, False])
@pytest.mark.parametrize('pbc_y', [True, False])
@pytest.mark.parametrize('gauge', [None, 'landau_x', 'landau_y', 'symmetric'])
def test_hopping_phases(lx, ly, p, q, commensurate_gauges, pbc_x, pbc_y, gauge):
    if commensurate_gauges is None:
        should_fail = True
        if not pbc_x and gauge in ['landau_x', None]:
            should_fail = False
        if not pbc_y and gauge in ['landau_y', None]:
            should_fail = False
        if not pbc_x and not pbc_y:
            should_fail = False
    elif commensurate_gauges == 'all':
        should_fail = False
    elif commensurate_gauges == 'landau_y':
        if gauge in ['landau_y', None]:
            should_fail = False
        elif gauge == 'landau_x':
            should_fail = pbc_x
        elif gauge == 'symmetric':
            should_fail = pbc_x or pbc_y

    if should_fail:
        match = 'None of the supported gauge choices' if gauge is None else 'incommensurate'
        with pytest.raises(ValueError, match=match):
            _ = hopping_phases(p=p, q=q, Lx=lx, Ly=ly, pbc_x=pbc_x, pbc_y=pbc_y, gauge=gauge)
        return

    phases_x, phases_y = hopping_phases(p=p, q=q, Lx=lx, Ly=ly, pbc_x=pbc_x, pbc_y=pbc_y, gauge=gauge)

    # correct shape?
    assert phases_x.shape == (lx if pbc_x else lx - 1, ly)
    assert phases_y.shape == (lx, ly if pbc_y else ly - 1)

    # are phase factors?
    assert np.allclose(np.abs(phases_x), 1)
    assert np.allclose(np.abs(phases_y), 1)

    # check enclosed phase on every plaquette
    expect_plaquette_phase = np.exp(2.0j * np.pi * p / q)
    plaquettes_x_range = range(lx) if pbc_x else range(lx - 1)
    plaquettes_y_range = range(ly) if pbc_y else range(ly - 1)
    for x in plaquettes_x_range:
        for y in plaquettes_y_range:
            phase = phases_x[x, y]
            phase *= phases_y[(x + 1) % lx, y]
            phase *= np.conj(phases_x[x, (y + 1) % ly])
            phase *= np.conj(phases_y[x, y])
            assert np.allclose(phase, expect_plaquette_phase)
