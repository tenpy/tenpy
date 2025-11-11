# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import pytest
from test_model import check_general_model

from tenpy.models import hubbard


def test_FermiHubbardModel():
    check_general_model(hubbard.FermiHubbardModel, {'lattice': 'Square', 'Lx': 2, 'Ly': 3}, {'phi_ext': [None, 0.2]})


def test_FermiHubbardModel2():
    check_general_model(hubbard.FermiHubbardModel2, {'lattice': 'Square', 'Lx': 2, 'Ly': 3}, {'phi_ext': [None, 0.2]})


def test_FermiHubbardChain():
    check_general_model(hubbard.FermiHubbardChain, {}, {})

    # for a chain, adding phi_ext should raise
    with pytest.raises(ValueError) as e_info:
        check_general_model(hubbard.FermiHubbardChain, {'phi_ext': 0.5}, {})
    assert e_info.type is ValueError
    assert e_info.value.args[0] == 'Expected one phase per lattice dimension.'


def test_BoseHubbardModel():
    params = {'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'V': 0.1, 'U': 0.3}
    check_general_model(hubbard.BoseHubbardModel, params, {'phi_ext': [None, 0.2]})


def test_BoseHubbardChain():
    check_general_model(hubbard.BoseHubbardChain, {}, {})


def test_DipolarBoseHubbardChain():
    # check dipolar charges for one specific case
    Nmax = 4
    model = hubbard.DipolarBoseHubbardChain(dict(conserve='dipole', Nmax=Nmax))
    expect_N = np.arange(Nmax + 1)
    for i, s in enumerate(model.lat.mps_sites()):
        expect_dipole = i * expect_N
        expect_charges = np.array([expect_N, expect_dipole]).T
        assert np.all(s.leg.charges == expect_charges)
    # check general properties for many cases
    check_general_model(
        hubbard.DipolarBoseHubbardChain,
        {},
        {
            'conserve': ['dipole', 'N', 'parity', None],
        },
    )
