"""A collection of tests to check the functionality of `tenpy.tebd`"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy.testing as npt
import tenpy.linalg.np_conserved as npc
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinChain
import tenpy.algorithms.tebd as tebd
from tenpy.networks.site import SpinHalfSite
from tenpy.algorithms.exact_diag import ExactDiag
import pytest

from test_dmrg import e0_tranverse_ising


def test_trotter_decomposition():
    # check that the time steps sum up to what we expect
    for order in [1, 2, 4]:
        dt = tebd.TEBDEngine.suzuki_trotter_time_steps(order)
        for N in [1, 2, 5]:
            evolved = [0., 0.]
            for j, k in tebd.TEBDEngine.suzuki_trotter_decomposition(order, N):
                evolved[k] += dt[j]
            npt.assert_array_almost_equal_nulp(evolved, N * np.ones([2]), N * 2)


@pytest.mark.slow
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_tebd(bc_MPS, g=0.5):
    L = 2 if bc_MPS == 'infinite' else 6
    #  xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS=bc_MPS)
    #  M = XXZChain(xxz_pars)
    # factor of 4 (2) for J (h) to change spin-1/2 to Pauli matrices
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1, -1.], [1, -1.]] * L)[:L]  # pointing in (-x)-direction
    psi = MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)

    tebd_param = {
        'dt': 0.01,
        'order': 2,
        'delta_tau_list': [0.1, 1.e-4, 1.e-8],
        'max_error_E': 1.e-9,
        'trunc_params': {
            'chi_max': 50,
            'trunc_cut': 1.e-13
        }
    }
    engine = tebd.TEBDEngine(psi, M, tebd_param)
    engine.run_GS()

    print("norm_test", psi.norm_test())
    if bc_MPS == 'finite':
        psi.canonical_form()
        ED = ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        E_ED, psi_ED = ED.groundstate()
        Etebd = np.sum(M.bond_energies(psi))
        print("E_TEBD={Etebd:.14f} vs E_exact={Eex:.14f}".format(Etebd=Etebd, Eex=E_ED))
        assert (abs((Etebd - E_ED) / E_ED) < 1.e-7)
        ov = npc.inner(psi_ED, ED.mps_to_full(psi), 'range', do_conj=True)
        print("compare with ED: overlap = ", abs(ov)**2)
        assert (abs(abs(ov) - 1.) < 1.e-7)

        # Test real time TEBD: should change on an eigenstate
        Sold = np.average(psi.entanglement_entropy())
        for i in range(3):
            engine.run()
        Enew = np.sum(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(Enew - Etebd) < 1.e-8)
        assert (abs(Sold - Snew) < 1.e-5)  # somehow we need larger tolerance here....

    if bc_MPS == 'infinite':
        Etebd = np.average(M.bond_energies(psi))
        Eexact = e0_tranverse_ising(g)
        print("E_TEBD={Etebd:.14f} vs E_exact={Eex:.14f}".format(Etebd=Etebd, Eex=Eexact))

        Sold = np.average(psi.entanglement_entropy())
        for i in range(2):
            engine.run()
        Enew = np.average(M.bond_energies(psi))
        Snew = np.average(psi.entanglement_entropy())
        assert (abs(Etebd - Enew) < 1.e-7)
        assert (abs(Sold - Snew) < 1.e-5)  # somehow we need larger tolerance here....


def test_RandomUnitaryEvolution():
    L = 8
    spin_half = SpinHalfSite(conserve='Sz')
    psi = MPS.from_product_state([spin_half] * L, [0, 1] * (L // 2), bc='finite')  # Neel state
    assert tuple(psi.chi) == (1, 1, 1, 1, 1, 1, 1)
    TEBD_params = dict(N_steps=2, trunc_params={'chi_max': 10})
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    print(eng.psi.chi)
    assert tuple(eng.psi.chi) == (2, 4, 8, 10, 8, 4, 2)

    # infinite versions
    TEBD_params['trunc_params']['chi_max'] = 20
    psi = MPS.from_product_state([spin_half] * 2, [0, 0], bc='infinite')
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    print(eng.psi.chi)
    assert tuple(eng.psi.chi) == (1, 1)  # all up can not be changed
    eng.psi = MPS.from_product_state([spin_half] * 2, [0, 1], bc='infinite')
    eng.run()
    print(eng.psi.chi)
    assert tuple(eng.psi.chi) == (16, 8)
