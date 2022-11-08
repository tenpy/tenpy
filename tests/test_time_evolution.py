r"""Test several time evolution methods and compare to exact expectation values

Setup is a time evolution with TFI Model starting from a spin polarized state.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import scipy.linalg as LA
import numpy.testing as npt
import pytest

import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinChain
from tenpy.algorithms import tebd, tdvp, mpo_evolution, exact_diag


@pytest.mark.parametrize('bc_MPS, approximation, compression', [
    ('finite', 'I', 'SVD'),
    ('finite', 'I', 'variational'),
    ('finite', 'II', 'variational'),
    ('finite', 'I', 'zip_up'),
    ('finite', 'II', 'zip_up'),
    ('infinite', 'I', 'SVD'),
    ('infinite', 'II', 'SVD'),
    ('infinite', 'II', 'variational'),
])
@pytest.mark.slow
def test_ExpMPOEvolution(bc_MPS, approximation, compression, g=1.5):
    # Test a time evolution against exact diagonalization for finite bc
    dt = 0.01
    if bc_MPS == 'finite':
        L = 6
    else:
        L = 2
    #  model_pars = dict(L=L, Jx=0., Jy=0., Jz=-4., hx=2. * g, bc_MPS=bc_MPS, conserve=None)
    #  state = ([[1 / np.sqrt(2), -1 / np.sqrt(2)]] * L)  # pointing in (-x)-direction
    #  state = ['up'] * L  # pointing in (-z)-direction
    model_pars = dict(L=L, Jx=1., Jy=1., Jz=1., hz=0.2, bc_MPS=bc_MPS, conserve='best')
    state = ['up', 'down'] * (L//2)  # Neel
    M = SpinChain(model_pars)
    psi = MPS.from_product_state(M.lat.mps_sites(), state, bc=bc_MPS)

    options = {
        'dt': dt,
        'N_steps': 1,
        'order': 1,
        'approximation': approximation,
        'compression_method': compression,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-8
        }
    }
    eng = mpo_evolution.ExpMPOEvolution(psi, M, options)

    if bc_MPS == 'finite':
        ED = exact_diag.ExactDiag(M)
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
        TEBD_params = {'dt': dt, 'order': 2, 'N_steps': 1, 'trunc_params': options['trunc_params']}
        EngTEBD = tebd.TEBDEngine(psiTEBD, M, TEBD_params)
        for i in range(30):
            EngTEBD.run()
            psi = eng.run()
            print(psi.norm)
            ov = psi.overlap(psiTEBD, understood_infinite=True)
            print(abs(abs(ov) - 1), abs(ov - 1))
            assert (abs(abs(ov) - 1) < 1e-4)


def fermion_TFI_H(L, g=1.5, J=1.):
    r'''return the quadratic Hamiltonian of the TFI Model after Jordan-Wigner transformation
    This is a 2L*2L matrix of the form:
    H = c^dagger (      A            B   ) c
                   -B^\dagger       -A^T
    where c = (c_1, ..., c_N, c^dagger_1, ..., c^dagger_N)^T
    '''
    A = np.zeros((L, L))
    B = np.zeros((L, L))
    for i in range(L - 1):
        A[i, i] += 2 * g
        A[i, i + 1] += -J
        A[i + 1, i] += -np.conj(J)
        B[i, i + 1] += -J
        B[i + 1, i] += J  # no minus due to anti-commutation
    A[L - 1, L - 1] += 2 * g
    return np.concatenate((np.concatenate(
        (A, B), axis=1), np.concatenate((B.conj().T, -A.T), axis=1)),
                          axis=0)


def exact_expectation(L, g, t=1., dt=0.01):
    """Prepare system in the ground state of H(J=0) and do time evolution with full H.

    Perform a generalized Boguliobov transformation, see e.g.:
    J.-P. Blaizot and G. Ripka, “Quantum Theory of Finite Systems,”
    The MIT Press, Cambridge, Massachusetts(1986)
    """
    gamma = np.kron(np.array([[0, 1], [1, 0]]), np.identity(L))

    H0 = fermion_TFI_H(L, g=g, J=0.)
    vp, U0p = LA.eigh(H0)
    assert np.all(vp.round(10) != 0.)  # so far no handling of zero eigenvalues

    # reshape eigenvalues und -vectors to the form (v1, ..., vN, -v1, ..., -vN)
    U0 = np.zeros(U0p.shape)
    U0[:, :L] = U0p[:, L::][:, ::-1]
    for i in range(L):
        U0[:, L + i] = gamma @ U0[:, i]

    H1 = fermion_TFI_H(L, g=g)
    v2, U2 = LA.eigh(H1)
    assert np.all(vp.round(10) != 0.)
    v = np.zeros(v2.shape)
    U = np.zeros(U2.shape)
    v[:L] = v2[L::][::-1]
    v[L:] = v2[:L]
    U[:, :L] = U2[:, L::][:, ::-1]
    for i in range(L):
        U[:, L + i] = gamma @ U[:, i]

    mag = []  # avarage magnetization
    szsz = []  # correlation functions
    spsm = []  # nearest neighbor <S+S- + S-S+> correlation
    for t in np.arange(0, t, dt):
        Ub = U @ np.diag(np.exp(
            -1j * t * v)) @ U.conj().T @ U0  # the total (unitary) Boguliobov transformation
        X = Ub[L::, L::]
        Y = Ub[L::, :L]
        npt.assert_almost_equal((X @ X.conj().T + Y @ Y.conj().T), np.identity(L), 7)
        npt.assert_almost_equal((X.conj().T @ X + Y.T @ Y.conj()), np.identity(L), 7)
        npt.assert_almost_equal((X @ Y.T + Y @ X.T), 0, 7)
        npt.assert_almost_equal((X.T @ Y.conj() + Y.conj().T @ X), 0, 7)

        X_inv = LA.inv(X)
        M = np.zeros((2 * L, 2 * L), dtype='complex')
        M[:L, :L] = -Y @ (X_inv.conj())
        Nc = np.zeros((2 * L, 2 * L), dtype='complex')
        Nc[L::, L::] = -(X_inv.T) @ (Y.conj().T)
        A = np.kron(np.array([[-1, 1], [0, -1]]), np.identity(L))

        S = np.concatenate((np.concatenate((A, Nc), axis=1), np.concatenate((M, -A.T), axis=1)),
                           axis=0)
        Delta = LA.inv(S)
        npt.assert_almost_equal(np.abs(LA.det(X)) * np.sqrt(LA.det(S)), 1, 7)

        # measure z-magnetization
        mz = []
        for i in range(L):
            ni = Delta[i + L, i]
            assert abs(ni.imag) < 1e-10
            mz.append(1 - 2 * ni.real)
        mag.append(mz)

        # measure sigmaz-sigmaz correlations
        s = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                ninj = -0.25*(-Delta[j+L, i+3*L] + Delta[i+L, j+3*L])*(-Delta[j+2*L, i] + Delta[i+2*L, j]) \
                    + 0.25*(-Delta[j+L, j] + Delta[j+2*L, j+3*L])*(-Delta[i+L, i] + Delta[i+2*L, i+3*L]) \
                    - 0.25*(-Delta[j+L, i] + Delta[i+2*L, j+3*L]) * \
                    (-Delta[i+L, j] + Delta[j+2*L, i+3*L])
                ni = Delta[i + L, i]
                nj = Delta[j + L, j]
                sij = 1 - 2 * ni - 2 * nj + 4 * ninj
                assert abs(sij.imag) < 1e-10
                s[i, j] = sij.real if i != j else 1.
        szsz.append(s)

        # measure sigma+sigma- correlatios
        s = []
        for i in range(L - 1):
            j = i + 1
            ci = 0.5 * (Delta[j + L, i] - Delta[i + 2 * L, j + 3 * L] + Delta[i + L, j] -
                        Delta[j + 2 * L, i + 3 * L])
            assert abs(ci.imag) < 1e-10
            s.append(ci.real)
        spsm.append(s)
    return np.array(mag), np.array(szsz), np.array(spsm)


@pytest.mark.parametrize('algorithm', ['TEBD', 'TDVP', 'ExpMPO'])
@pytest.mark.slow
def test_time_methods(algorithm):
    L = 6
    g = 1.2

    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * L  # prepare system in spin polarized state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dt = 0.01
    N_steps = 2
    t = 0.3  # total time evolution
    params = {
        'order': 2,
        'dt': dt,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }
    if algorithm == 'TEBD':
        eng = tebd.TEBDEngine(psi, M, params)
    elif algorithm == 'TDVP':
        del params['order']
        eng = tdvp.TwoSiteTDVPEngine(psi, M, params)
    elif algorithm == 'ExpMPO':
        params['compression_method'] = 'SVD'
        del params['order']
        eng = mpo_evolution.ExpMPOEvolution(psi, M, params)
    else:
        raise ValueError("test works only for TEDB and TDVP so far")

    mag = [psi.expectation_value("Sigmaz")]
    szsz = [psi.correlation_function("Sigmaz", "Sigmaz")]
    corr = psi.correlation_function("Sp", "Sm")
    spsm = [corr.diagonal(1) + corr.diagonal(-1)]

    for ti in np.arange(0, t, dt * N_steps):
        eng.run()
        mag.append(psi.expectation_value("Sigmaz"))
        szsz.append(psi.correlation_function("Sigmaz", "Sigmaz"))
        corr = psi.correlation_function("Sp", "Sm")
        spsm.append(corr.diagonal(1) + corr.diagonal(-1))

    m_exact, szsz_exact, spsm_exact = exact_expectation(L, g, t, dt * N_steps)
    npt.assert_almost_equal(np.array(mag)[:-1, :], m_exact, 4)
    npt.assert_almost_equal(np.array(szsz)[:-1, :, :], szsz_exact, 4)
    npt.assert_almost_equal(np.array(spsm)[:-1, :], spsm_exact, 4)


class RabiOscillations(CouplingMPOModel,NearestNeighborModel):
    def init_sites(self, model_params):
        site = SpinHalfSite(conserve=None)
        site.add_op('P1', -0.5*(site.Sigmaz - site.Id))
        return site

    def init_terms(self, model_params):
        t = model_params.get('time', 0.)
        omega0 = model_params.get('omega0', 1.)
        omega1 = model_params.get('omega1', 1.)
        omega = model_params.get('omega', 1.)
        self.add_onsite(-0.5*omega0, 0, 'Sigmaz')
        self.add_onsite(-0.5*omega1 * np.cos(omega * t), 0, 'Sigmax')
        self.add_onsite(+0.5*omega1 * np.sin(omega * t), 0, 'Sigmay')

    def exact_solution_P1(self, t):
        # see https://en.wikipedia.org/wiki/Rabi_cycle#In_quantum_computing
        omega = self.options.get('omega', 1.)
        omega0 = self.options.get('omega0', 1.)
        omega1 = self.options.get('omega1', 1.)
        Omega = np.sqrt((omega - omega0)**2 + omega1**2)
        print(omega, omega0, omega1, Omega)
        return (omega1/Omega)**2 * np.sin(0.5*Omega * t)**2


@pytest.mark.slow
def test_time_dependent_evolution(om=0.2*np.pi, om0=np.pi, om1=0.5*np.pi, eps=1.e-4):
    L = 4
    model_params = dict(L=L, omega=om, omega0=om0, omega1=om1, bc_MPS='finite')
    M = RabiOscillations(model_params)
    product_state = ["up"] * L  # prepare system in spin polarized state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dt = 0.025
    N_steps = 2
    params = {
        'order': 1,
        'dt': dt,
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }
    ts = np.arange(0, 1.2, dt * N_steps)
    exact = M.exact_solution_P1(ts)
    # start with TEBD
    P1s = []
    eng = tebd.TimeDependentTEBD(psi, M, params.copy())
    for i, t in enumerate(ts):
        P1 = psi.expectation_value("P1")
        P1s.append(P1)

        assert np.max(np.abs(P1 - exact[i])) < eps

        if abs(t - 0.4) < 1.e-7:
            print('switch to TimeDependentTwoSiteTDVP')
            params['start_time'] = t
            del params['order']
            eng = tdvp.TimeDependentTwoSiteTDVP(psi, M, params.copy())
        if abs(t - 0.8) < 1.e-7:
            print('switch to TimeDependentExpMPOEvolution')
            params['start_time'] = t
            params['compression_method'] = 'SVD'
            #  del params['order']
            eng = mpo_evolution.TimeDependentExpMPOEvolution(psi, M, params.copy())

        eng.run()
    #  import matplotlib.pyplot as plt
    #  plt.plot(ts, exact, '-', label='exact')
    #  plt.plot(ts, P1s, 'o', label='measured')
    #  plt.legend()
    #  plt.show()
