# Copyright 2018 TeNPy Developers
from tenpy.models import bose_hubbard
from test_model import check_general_model


def test_BoseHubbardModel():
    check_general_model(bose_hubbard.BoseHubbardModel, {
        'lattice': "Square",
        'Lx': 2,
        'Ly': 3,
        'V': 0.1,
        'U': 0.3
    }, {})


def test_BoseHubbardChain():
    check_general_model(bose_hubbard.BoseHubbardChain, {}, {})
