# Copyright (C) TeNPy Developers, Apache license
import pytest
from test_model import check_general_model

from tenpy.models import spins


def test_SpinModel():
    check_general_model(spins.SpinModel, {'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'bc_MPS': 'finite'}, {})


def test_SpinChain():
    check_general_model(spins.SpinChain, {'bc_MPS': 'finite'}, {'conserve': [None, 'parity', 'Sz'], 'S': [0.5, 1, 2]})
    check_general_model(
        spins.SpinChain,
        {
            'hz': 2.0,
            'Jx': -4.0,
            'Jz': -0.4,
            'L': 4,
            'bc_MPS': 'finite',
        },
        {'conserve': [None, 'parity']},
    )


@pytest.mark.skip('dipole conservation is not (yet) supported by the cyten symmetry framework')
def test_DipolarSpinChain():
    # imports kept local (rather than module-level as in the original) since
    # tenpy.algorithms.exact_diag still transitively imports the gutted np_conserved module;
    # a module-level import would break collection of this whole test file even though this
    # test itself is skipped.
    import numpy as np

    from tenpy.algorithms.exact_diag import ExactDiag

    # check dipolar charges on Chain for one specific case
    L = 6  # use size small enough for ED
    model = spins.DipolarSpinChain(dict(S=1, J4=1, conserve='dipole', L=L))
    expect_2Sz = np.array([-2, 0, 2])
    for i, s in enumerate(model.lat.mps_sites()):
        expect_dipole = i * expect_2Sz
        expect_charges = np.array([expect_2Sz, expect_dipole]).T
        assert np.all(s.leg.charges == expect_charges)
    ED = ExactDiag(model)
    ED.build_full_H_from_mpo()
    full_H_qtotal = ED.full_H.qtotal
    assert np.allclose(full_H_qtotal, [0, 0])
    # check general properties for many cases
    check_general_model(spins.DipolarSpinChain, {}, {'J4': [0, 1], 'conserve': ['dipole', 'Sz', None]})
