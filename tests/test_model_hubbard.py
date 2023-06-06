# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
from tenpy.models import hubbard
from test_model import check_general_model


def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})

def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel2, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})


def test_FermiHubbardChain():
    check_general_model(hubbard.FermiHubbardChain, {}, {})


def test_BoseHubbardModel():
    check_general_model(hubbard.BoseHubbardModel, {
        'lattice': "Square",
        'Lx': 2,
        'Ly': 3,
        'V': 0.1,
        'U': 0.3
    }, {})


def test_BoseHubbardChain():
    check_general_model(hubbard.BoseHubbardChain, {}, {})
