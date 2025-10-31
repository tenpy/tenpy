# Copyright (C) TeNPy Developers, Apache license

import pytest
from test_model import check_general_model

from tenpy.models.clock import ClockChain, ClockModel


def test_ClockChain_general():
    check_general_model(
        ClockChain,
        dict(L=4, J=1.0, bc_MPS='finite', sort_charge=True),
        dict(conserve=[None, 'Z'], q=[2, 3, 5], g=[0, 0.2]),
    )


@pytest.mark.slow
def test_ClockModel2D_general():
    check_general_model(
        ClockModel,
        dict(Lx=2, J=1.0, g=0.1, q=3, sort_charge=True),
        dict(
            Ly=[2, 3],
            bc_MPS=['finite', 'infinite'],
            bc_y=['ladder', 'cylinder'],
            lattice=['Square', 'Honeycomb', 'Kagome'],
        ),
    )
