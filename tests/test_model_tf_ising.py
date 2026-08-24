# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models.tf_ising import TFIChain, TFIModel


def test_TFIChain_general():
    check_general_model(TFIChain, dict(L=4, J=1.0, bc_MPS='finite'), {'conserve': [None, 'parity'], 'g': [0.0, 0.2]})


@pytest.mark.slow
def test_TFIModel2D_general():
    # bc_MPS='infinite' is not yet supported by the cyten-native add_coupling path
    # (CouplingModel.calc_H_MPO() currently requires bc_MPS='finite' when using add_coupling).
    # Lx=1 (rather than 2) since the hermiticity check in check_general_model contracts the
    # whole MPO into a dense operator (MPO.is_hermitian()/overlap() aren't yet ported to cyten
    # tensors), which is only tractable for small system sizes.
    check_general_model(
        TFIModel,
        dict(Lx=1, J=1.0, g=0.1, bc_MPS='finite'),
        {
            'Ly': [2, 3],
            'bc_y': ['ladder', 'cylinder'],
            'lattice': ['Square', 'Honeycomb', 'Kagome'],
        },
    )
