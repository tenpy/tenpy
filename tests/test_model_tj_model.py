# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models import tj_model


def test_tJModel():
    with pytest.raises(NotImplementedError):
        tj_model.tJModel({})
    pytest.xfail()
    check_general_model(tj_model.tJModel, {'lattice': 'Square', 'Lx': 2, 'Ly': 3}, {})


def test_tJChain():
    with pytest.raises(NotImplementedError):
        tj_model.tJChain({})
    pytest.xfail()
    check_general_model(tj_model.tJChain, {}, {})
