
from __future__ import division

from tenpy.models import spins
from test_model import check_general_model


def test_SpinChain():
    check_general_model(spins.SpinChain, {},
                        {'conserve': [None, 'parity', 'Sz', 'best'],
                         'S': [0.5, 1, 2]})

