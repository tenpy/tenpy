# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models.pxp import PXPChain


@pytest.mark.parametrize('conserve', ['best', 'None', None])
def test_PXPChain_general(conserve):
    check_general_model(
        PXPChain, dict(L=10, J=1.0, bc_MPS='finite', conserve=conserve), {'bc_MPS': ['finite', 'infinite']}
    )


@pytest.mark.parametrize('conserve', ['parity', 'Sz'])
def test_PXPChain_rejects_parity(conserve):
    with pytest.raises(ValueError, match='does not conserve'):
        PXPChain(dict(L=6, J=1.0, bc_MPS='finite', conserve=conserve))
