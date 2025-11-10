# Copyright (C) TeNPy Developers, Apache license

import pytest
from test_model import check_general_model

from tenpy.models.tf_ising import TFIChain, TFIModel


def test_TFIChain_general():
    check_general_model(
        TFIChain, dict(L=4, J=1.0, bc_MPS='finite', sort_charge=True), {'conserve': [None, 'parity'], 'g': [0.0, 0.2]}
    )


@pytest.mark.slow
def test_TFIModel2D_general():
    check_general_model(
        TFIModel,
        dict(Lx=2, J=1.0, g=0.1, sort_charge=True),
        {
            'Ly': [2, 3],
            'bc_MPS': ['finite', 'infinite'],
            'bc_y': ['ladder', 'cylinder'],
            'lattice': ['Square', 'Honeycomb', 'Kagome'],
        },
    )
