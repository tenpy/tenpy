
# Copyright (C) TeNPy Developers, Apache license
from tenpy.models.pxp import PXPChain
from test_model import check_general_model


def test_XXZChain_general():
    check_general_model(PXPChain, dict(L=10, J=1., bc_MPS='finite'),
                        {'bc_MPS': ['finite', 'infinite']})
