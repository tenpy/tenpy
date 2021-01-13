# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
import numpy as np
from tenpy.models.fermions_spinless import FermionChain, FermionModel
from test_model import check_general_model
from tenpy.models.spins import SpinChain


def test_FermionModel():
    check_general_model(FermionModel, {'lattice': "Square", 'Lx': 2, 'Ly': 3}, {})


def test_FermionChain():
    check_general_model(FermionChain, {'L': 4}, {
        'conserve': [None, 'parity', 'N'],
        'mu': [0., 0.123],
        'bc_MPS': ['finite', 'infinite']
    })


def test_map_Fermions_Spins(L=6, Jxx=1., Jz=0.1, hz=0.01):
    # we only check for correctness in the (uniform) bulk, not caring about the boundary terms
    print("Spins")
    spar = dict(L=L, Jx=Jxx, Jy=Jxx, Jz=Jz, hz=hz)
    schain = SpinChain(spar)
    # Sz + 0.5 -> n, thus some constants appearing
    constant = -0.25 * Jz - 0.5 * hz
    Hb_s = schain.H_bond[L // 2].transpose(['p0', 'p1', 'p0*', 'p1*']).to_ndarray()
    Hb_s = Hb_s.reshape(4, 4) + constant * np.eye(4)
    print(Hb_s)

    print("Fermions")
    fpar = dict(L=spar['L'], J=-0.5 * spar['Jx'], V=spar['Jz'], mu=spar['hz'] + spar['Jz'])
    fchain = FermionChain(fpar)
    Hb_f = fchain.H_bond[L // 2].transpose(['p0', 'p1', 'p0*', 'p1*']).to_ndarray()
    Hb_f = Hb_f.reshape(4, 4)
    print(Hb_f)
    for i in range(2, L - 1):
        Hb_s = schain.H_bond[i].transpose(['p0', 'p1', 'p0*', 'p1*']).to_ndarray().reshape(4, 4)
        Hb_f = fchain.H_bond[i].transpose(['p0', 'p1', 'p0*', 'p1*']).to_ndarray().reshape(4, 4)
        assert (np.allclose(Hb_s + constant * np.eye(4), Hb_f))
