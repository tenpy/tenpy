# Copyright 2018 TeNPy Developers
from tenpy.models import bose_hubbard
from test_model import check_general_model


def test_BoseHubbardModel():
    check_general_model(bose_hubbard.BoseHubbardModel, {'lattice': "Square"}, {})

def test_BoseHubbardChain():
    check_general_model(bose_hubbard.BoseHubbardChain, {}, {})
