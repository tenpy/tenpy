# Copyright (C) TeNPy Developers, Apache license
from test_model import check_general_model

from tenpy.models.pxp import PXPChain


def test_PXPChain_general():
    check_general_model(PXPChain, dict(L=10, J=1.0, bc_MPS='finite'), {'bc_MPS': ['finite', 'infinite']})
