"""test of :class:`tenpy.models.XXZChain`.
"""

from __future__ import division

from tenpy.models.tf_ising import TFIChain, TFIModel2D
from test_model import check_general_model


def test_TFIChain_general():
    check_general_model(TFIChain, dict(L=4, J=1., bc_MPS='finite'),
                        {'conserve': [None, 'parity'], 'g': [0., 0.2]})


def test_TFIModel2D_general():
    check_general_model(TFIModel2D, dict(Lx=2, J=1., g=0.1),
                        {'Ly': [2, 3], 'bc_MPS': ['finite', 'infinite'],
                         'bc_y': ['ladder', 'cylinder']})
