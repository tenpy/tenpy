
# Copyright (C) TeNPy Developers, Apache license
import pytest
from tenpy.models.pxp import PXPChain
from test_model import check_general_model


@pytest.mark.parametrize('conserve', ['best', 'parity', 'None'])
def test_XXZChain_general(conserve):
    check_general_model(PXPChain, dict(L=10, J=1., bc_MPS='finite', conserve=conserve),
                        {'bc_MPS': ['finite', 'infinite']})
