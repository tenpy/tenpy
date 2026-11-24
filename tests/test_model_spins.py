# Copyright (C) TeNPy Developers, Apache license
import lasyten as lasy
import numpy as np
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


@pytest.mark.parametrize('conserve', ['dipole', 'Sz', 'parity', None])
def test_DipolarSpinChain(conserve):

    if conserve == 'dipole':
        with pytest.raises(ValueError, match='Invalid `conserve`'):
            _ = lasy.sites.SpinSite(conserve='dipole')
        pytest.xfail('Dipole symmetry not ported yet')

    # check dipolar charges on Chain for one specific case
    L = 6  # use size small enough for ED
    S_tot = 1
    model = spins.DipolarSpinChain(dict(S=S_tot, J4=1, conserve=conserve, L=L))
    Sz = np.array([-1, 0, 1])
    for x, s in enumerate(model.lattice.mps_sites()):
        if conserve == 'dipole':
            expect_sectors = np.array([2 * Sz, x * 2 * Sz])
        elif conserve == 'Sz':
            expect_sectors = 2 * Sz[:, None]
        elif conserve == 'parity':
            expect_sectors = (Sz + S_tot)[:, None] % 2
        elif conserve is None:
            expect_sectors = np.zeros((3, 1))
        assert np.array_equal(s.leg.sectors_of_basis, expect_sectors)

    check_general_model(spins.DipolarSpinChain, {'conserve': conserve}, {'J4': [0, 1]})
