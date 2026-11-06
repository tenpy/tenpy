# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models.pxp import PXPChain


def test_PXPChain_general():
    with pytest.raises(NotImplementedError, match='MultiCouplingTerms add to graph'):
        PXPChain(dict(L=10, J=1.0, bc_MPS='finite')).get_H_MPO()
    pytest.xfail('MultiCouplingTerms add to graph')
    check_general_model(PXPChain, dict(L=10, J=1.0, bc_MPS='finite'), {'bc_MPS': ['finite', 'infinite']})
