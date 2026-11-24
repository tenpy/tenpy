# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from test_model import check_general_model

from tenpy.models.fermions_spinless import FermionChain, FermionModel
from tenpy.models.spins import SpinChain


def test_FermionModel():
    check_general_model(FermionModel, {'lattice': 'Square', 'Lx': 2, 'Ly': 3}, {'phi_ext': [None, 1.0]})


def test_FermionChain():
    check_general_model(
        FermionChain,
        {'L': 4},
        {'conserve': ['parity', 'N'], 'mu': [0.0, 0.123], 'bc_MPS': ['finite', 'infinite']},
    )


def test_map_Fermions_Spins(L=6, Jxx=1.0, Jz=0.1, hz=0.01):
    # we only check for correctness in the (uniform) bulk, not caring about the boundary terms
    print('Spins')
    spar = dict(L=L, Jx=Jxx, Jy=Jxx, Jz=Jz, hz=hz)
    schain = SpinChain(spar)
    # Sz + 0.5 -> n, thus some constants appearing
    constant = -0.25 * Jz - 0.5 * hz
    Hb_s = schain.get_H_bond()[L // 2].to_numpy(leg_order=['p0', 'p1', 'p0*', 'p1*'], understood_braiding=True)
    Hb_s = Hb_s.reshape(4, 4) + constant * np.eye(4)
    print(Hb_s)

    print('Fermions')
    fpar = dict(L=spar['L'], J=-0.5 * spar['Jx'], V=spar['Jz'], mu=spar['hz'] + spar['Jz'])
    fchain = FermionChain(fpar)
    Hb_f = fchain.get_H_bond()[L // 2].to_numpy(leg_order=['p0', 'p1', 'p0*', 'p1*'], understood_braiding=True)
    Hb_f = Hb_f.reshape(4, 4)
    print(Hb_f)
    for i in range(2, L - 1):
        Hb_s = schain.get_H_bond()[i].to_numpy(leg_order=['p0', 'p1', 'p0*', 'p1*'], understood_braiding=True)
        Hb_f = fchain.get_H_bond()[i].to_numpy(leg_order=['p0', 'p1', 'p0*', 'p1*'], understood_braiding=True)
        assert np.allclose(Hb_s + constant * np.eye(4).reshape(2, 2, 2, 2), Hb_f)
