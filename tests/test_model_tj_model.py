# Copyright (C) TeNPy Developers, GNU GPLv3
from tenpy.models import tj_model
from test_model import check_general_model


def test_tJModel():
    check_general_model(tj_model.tJModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})


def test_tJChain():
    check_general_model(tj_model.tJModel, {}, {})
