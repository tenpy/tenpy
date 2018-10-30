# Copyright 2018 TeNPy Developers
from tenpy.models import fermions_hubbard
from test_model import check_general_model


def test_FermiHubbardModel():
    check_general_model(fermions_hubbard.FermionicHubbardModel, {'lattice': "Square"}, {})

def test_FermiHubbardChain():
    check_general_model(fermions_hubbard.FermionicHubbardChain, {}, {})
