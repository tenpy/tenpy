from tenpy.algorithms.mps_compress import *
import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy
from tenpy.models.spins import SpinChain
import pytest


def test_mps_compress(eps=1.e-13):
    # Test compression of a sum of a state with itself
    L = 5
    sites = [tenpy.networks.site.SpinHalfSite(conserve=None) for i in range(L)]
    plus_x = np.array([1., 1.])/np.sqrt(2)
    minus_x = np.array([1., -1.])/np.sqrt(2)
    psi = tenpy.networks.mps.MPS.from_product_state(sites, [plus_x for i in range(L)], bc='finite')
    psiOrth = tenpy.networks.mps.MPS.from_product_state(sites, [minus_x for i in range(L)],
                                                        bc='finite')
    psiSum = psi.add(psi, .5, .5)
    mps_compress(psiSum, {})
    assert (np.abs(psiSum.overlap(psi) - 1) < 1e-13)
    psiSum2 = psi.add(psiOrth, .5, .5)
    mps_compress(psiSum2, {})
    psiSum2.test_sanity()
    assert (np.abs(psiSum2.overlap(psi) - .5) < 1e-13)
    assert (np.abs(psiSum2.overlap(psiOrth) - .5) < 1e-13)


@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_svd_two_theta(bc_MPS):
    L = 4
    g = 0.5
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]] *
             L)[:L]  # pointing in (-x)-direction
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    psi2 = psi.copy()
    for i in range(L if bc_MPS == 'infinite' else L - 1):  # test for every non trivial bond
        svd_two_site(i, psi, {})
    assert (np.abs(psi2.norm - 1) < 1e-5)
    assert (np.abs(psi2.overlap(psi) - 1) < 1e-5)


def test_apply_mpo():
    bc_MPS = "finite"
    # NOTE: overlap doesn't work for calculating the energy (density) in infinite systems!
    # energy is extensive, overlap exponential....
    L = 5
    g = 0.5
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L)  # pointing in (-x)-direction
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    H = M.H_MPO
    Eexp = H.expectation_value(psi)
    psi2 = apply_mpo(H, psi, {})
    Eapply = psi2.overlap(psi)
    assert abs(Eexp - Eapply) < 1e-5


@pytest.mark.parametrize('bc_MPS, method', [('finite', 'I'), ('finite', 'II'),
                                            ('infinite', 'I'), ('infinite', 'II')])
def test_U_I(bc_MPS, method, g=0.5):
    # Test a time evolution against exact diagonalization for finite bc
    L = 10
    dt = 0.01
    if bc_MPS == 'finite':
        L = 6
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L)  # pointing in (-x)-direction
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    psi.test_sanity()

    U = make_U(M.H_MPO, dt * 1j, which=method)

    if bc_MPS == 'finite':
        ED = tenpy.algorithms.exact_diag.ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psiED = ED.mps_to_full(psi)
        psiED /= psiED.norm()

        UED = ED.exp_H(dt)
        for i in range(30):
            psi = apply_mpo(U, psi, {})
            psiED = npc.tensordot(UED, psiED, ('ps*', [0]))
            psi_full = ED.mps_to_full(psi)
            assert (abs(abs(npc.inner(psiED, psi_full, [0, 0], True)) - 1) < 1e-2)

    if bc_MPS == 'infinite':
        psiTEBD = psi.copy()
        TEBD_params = {'dt': dt, 'N_steps': 1}
        EngTEBD = tenpy.algorithms.tebd.Engine(psiTEBD, M, TEBD_params)
        for i in range(30):
            EngTEBD.run()
            psi = apply_mpo(U, psi, {})
            print(np.abs(psi.overlap(psiTEBD) - 1))
            print(psi.norm)
            #This test fails
            assert (abs(abs(psi.overlap(psiTEBD)) - 1) < 1e-2)
