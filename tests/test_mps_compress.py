from tenpy.algorithms import mps_compress
import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy
from tenpy.tools.params import Config
from tenpy.models.spins import SpinChain
import pytest
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


@pytest.mark.parametrize('method', ['SVD', 'variational'])
def test_mps_compress(method, eps=1.e-13):
    # Test compression of a sum of a state with itself
    L = 5
    sites = [tenpy.networks.site.SpinHalfSite(conserve=None) for i in range(L)]
    plus_x = np.array([1., 1.]) / np.sqrt(2)
    minus_x = np.array([1., -1.]) / np.sqrt(2)
    psi = tenpy.networks.mps.MPS.from_product_state(sites, [plus_x for i in range(L)], bc='finite')
    psiOrth = tenpy.networks.mps.MPS.from_product_state(sites, [minus_x for i in range(L)],
                                                        bc='finite')
    options = {'compression_method': method, 'trunc_params': {'chi_max': 30}}
    psiSum = psi.add(psi, .5, .5)
    psiSum.compress(options)

    assert (np.abs(psiSum.overlap(psi) - 1) < 1e-13)
    psiSum2 = psi.add(psiOrth, .5, .5)
    psiSum2.compress(options)
    psiSum2.test_sanity()
    assert (np.abs(psiSum2.overlap(psi) - .5) < 1e-13)
    assert (np.abs(psiSum2.overlap(psiOrth) - .5) < 1e-13)


@pytest.mark.parametrize('method', ['SVD', 'variational'])
def test_apply_mpo(method):
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
    psi2 = psi.copy()
    options = {'compression_method': method, 'trunc_params': {'chi_max': 50}}
    H.apply(psi2, options)
    Eapply = psi2.overlap(psi)
    assert abs(Eexp - Eapply) < 1e-5
    psi3 = psi.copy()
    H.apply(psi3, options)
    Eapply3 = psi3.overlap(psi)
    assert abs(Eexp - Eapply3) < 1e-5


@pytest.mark.parametrize('bc_MPS, approximation, compression', [
    ('finite', 'I', 'SVD'),
    ('finite', 'I', 'variational'),
    ('finite', 'II', 'variational'),
    ('infinite', 'I', 'SVD'),
    ('infinite', 'II', 'SVD'),
    ('infinite', 'II', 'variational'),
])
def test_ExpMPOEvolution(bc_MPS, approximation, compression, g=1.5):
    # Test a time evolution against exact diagonalization for finite bc
    dt = 0.01
    if bc_MPS == 'finite':
        L = 6
    else:
        L = 4
    model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    M = SpinChain(model_pars)
    state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L)  # pointing in (-x)-direction
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)
    psi.test_sanity()

    options = {
        'dt': dt,
        'N_steps': 1,
        'order': 1,
        'approximation': approximation,
        'compression_method': compression,
        'trunc_params': {
            'chi_max': 30
        }
    }
    eng = ExpMPOEvolution(psi, M, options)

    if bc_MPS == 'finite':
        ED = tenpy.algorithms.exact_diag.ExactDiag(M)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        psiED = ED.mps_to_full(psi)
        psiED /= psiED.norm()

        UED = ED.exp_H(dt)
        for i in range(30):
            psi = eng.run()
            psiED = npc.tensordot(UED, psiED, ('ps*', [0]))
            psi_full = ED.mps_to_full(psi)
            assert (abs(abs(npc.inner(psiED, psi_full, [0, 0], True)) - 1) < dt)

    if bc_MPS == 'infinite':
        psiTEBD = psi.copy()
        TEBD_params = {'dt': dt, 'N_steps': 1}
        EngTEBD = tenpy.algorithms.tebd.Engine(psiTEBD, M, TEBD_params)
        for i in range(30):
            EngTEBD.run()
            psi = eng.run()
            print(psi.norm)
            print(np.abs(psi.overlap(psiTEBD) - 1))
            #This test fails
            assert (abs(abs(psi.overlap(psiTEBD)) - 1) < 1e-2)
