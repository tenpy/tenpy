# Copyright (C) TeNPy Developers, Apache license
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions
from test_model import check_general_model
import pytest
import itertools
import math
from test_model import check_general_model

from tenpy.algorithms import exact_diag
from tenpy.models.hofstadter import HofstadterBosons, HofstadterFermions


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


@pytest.mark.parametrize("bc_x", ["periodic", "open"])
@pytest.mark.parametrize("bc_y", ["periodic", "open"])
@pytest.mark.parametrize("num_particles", range(1, 9))
def test_HofstadterFermions_gauge_equivalence(bc_x: str, bc_y: str, num_particles: int) -> None:
    Lx, Ly = 3, 3
    base_model_params = dict(
        Jx=1.0,
        Jy=1.0,
        mu=0.0,
        v=1.0,
        phi=(1, 3),
        Lx=Lx,
        Ly=Ly,
        filling=(num_particles, Lx * Ly),
        bc_x=bc_x,
        bc_y=bc_y,
        conserve="N",
    )

    legal_gauges = ["landau_x", "landau_y"]
    if bc_x == "open" and bc_y == "open":
        legal_gauges += ["symmetric"]
    gauge_energies = {gauge: None for gauge in legal_gauges}

    for gauge in legal_gauges:
        model_params = {**base_model_params, "gauge": gauge}
        model = HofstadterFermions(model_params)
        diagonalizer = exact_diag.ExactDiag(model)
        diagonalizer.build_full_H_from_mpo()
        diagonalizer.full_diagonalization()
        gauge_energies[gauge], _ = diagonalizer.groundstate(charge_sector=[num_particles])

    for gauge1, gauge2 in itertools.combinations(legal_gauges, 2):
        energy1, energy2 = gauge_energies[gauge1], gauge_energies[gauge2]
        assert math.isclose(
            energy1,
            energy2), f"Different energies {energy1}!={energy2} for gauges {gauge1}, {gauge2}"
