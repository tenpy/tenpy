# Copyright 2018-2023 TeNPy Developers, GNU GPLv3
from tenpy.models import hubbard
from test_model import check_general_model


def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})
    check_general_model(hubbard.FermiHubbardModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3, 'phi_ext': 1.}, {})

def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel2, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})
    check_general_model(hubbard.FermiHubbardModel2, {'lattice': "Square", 'Lx': 2, 'Ly': 3, 'phi_ext': 1.}, {})


def test_FermiHubbardChain():
    check_general_model(hubbard.FermiHubbardChain, {}, {})
      # on the chain phi_ext should not do anything and only emit a warning
    check_general_model(hubbard.FermiHubbardChain, {'phi_ext': 1.}, {})


def test_BoseHubbardModel():
    params = {
        'lattice': "Square",
        'Lx': 2,
        'Ly': 3,
        'V': 0.1,
        'U': 0.3
    }
    check_general_model(hubbard.BoseHubbardModel, params, {})
    check_general_model(hubbard.BoseHubbardModel, {**params, 'phi_ext': 1.}, {})


def test_BoseHubbardChain():
    check_general_model(hubbard.BoseHubbardChain, {}, {})
