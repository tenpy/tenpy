# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models.haldane import BosonicHaldaneModel, FermionicHaldaneModel


@pytest.mark.slow
def test_BosonicHaldane():
    model_pars = {
        'Lx': 3,
        'Ly': 3,
        'phi_ext': 0.1,
        'conserve': 'N',
    }
    check_general_model(BosonicHaldaneModel, model_pars, {'bc_MPS': ['finite', 'infinite']})


@pytest.mark.slow
def test_FermionicHaldane():
    model_pars = {
        'Lx': 3,
        'Ly': 3,
        'phi_ext': 0.1,
        'conserve': 'N',
    }
    check_general_model(FermionicHaldaneModel, model_pars, {'bc_MPS': ['finite', 'infinite']})
