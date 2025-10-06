# Copyright (C) TeNPy Developers, Apache license
import pytest
from tenpy.models import hubbard
from test_model import check_general_model


def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {'phi_ext': [None, 0.2]})

def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel2, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {'phi_ext': [None, 0.2]})


def test_FermiHubbardChain():
    check_general_model(hubbard.FermiHubbardChain, {}, {})

    # for a chain, adding phi_ext should raise
    with pytest.raises(ValueError) as e_info:
        check_general_model(hubbard.FermiHubbardChain, {'phi_ext': 0.5}, {})
    assert e_info.type is ValueError
    assert e_info.value.args[0] == 'Expected one phase per lattice dimension.'

def test_BoseHubbardModel():
    params = {
        'lattice': "Square",
        'Lx': 2,
        'Ly': 3,
        'V': 0.1,
        'U': 0.3
    }
    check_general_model(hubbard.BoseHubbardModel, params, {'phi_ext': [None, 0.2]})


def test_BoseHubbardChain():
    check_general_model(hubbard.BoseHubbardChain, {}, {})
