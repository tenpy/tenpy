r"""
Test several time evolution methods and compare to exact expectation values
Setup is a time evolution with TFI Model starting from a spin polarized state
"""
import numpy as np
import scipy.linalg as LA
import numpy.testing as npt
import pytest

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd, tdvp

def fermion_TFI_H(L, g=1.5, J=1.):
    r'''return the quadratic Hamiltonian of the TFI Model after Jordan-Wigner transformation
    This is a 2L*2L matrix of the form:
    H = c^dagger (      A            B   ) c
                   -B^\dagger       -A^T
    where c = (c_1, ..., c_N, c^dagger_1, ..., c^dagger_N)^T
    '''
    A = np.zeros((L,L))
    B = np.zeros((L,L))
    for i in range(L-1):
        A[i,i] += 2*g
        A[i,i+1] += -J
        A[i+1,i] += -np.conj(J)
        B[i,i+1] += -J
        B[i+1,i] += J #no minus due to anti-commutation
    A[L-1,L-1] += 2*g
    return np.concatenate((np.concatenate((A,B),axis=1),np.concatenate((B.conj().T,-A.T),axis=1)),axis=0)

def exact_expectation(L, g, t=1., dt=0.01):
    r'''
    Prepare system in the ground state of H(J=0) and do time evolution with full H
    Perform a generalized Boguliobov transformation, see e.g.:
    J.-P. Blaizot and G. Ripka, “Quantum Theory of Finite Systems,” The MIT Press, Cambridge, Massachusetts(1986)
    '''
    gamma = np.kron(np.array([[0,1],[1,0]]), np.identity(L))

    H0 = fermion_TFI_H(L, g=g, J=0.)
    vp, U0p = LA.eigh(H0)
    assert np.all(vp.round(10) != 0.) #so far no handling of zero eigenvalues

    #reshape eigenvalues und -vectors to the form (v1, ..., vN, -v1, ..., -vN)
    U0 = np.zeros(U0p.shape)
    U0[:, :L] = U0p[:, L::][:, ::-1]
    for i in range(L):
        U0[:, L+i] = gamma@U0[:, i]

    H1 = fermion_TFI_H(L, g=g)
    v2, U2 = LA.eigh(H1)
    assert np.all(vp.round(10) != 0.)
    v = np.zeros(v2.shape)
    U = np.zeros(U2.shape)
    v[:L] = v2[L::][::-1]
    v[L:] = v2[:L]
    U[:, :L] = U2[:, L::][:, ::-1]
    for i in range(L):
        U[:, L+i] = gamma@U[:, i]

    mag = [] #avarage magnetization
    szsz = [] #correlation functions
    spsm = [] # nearest neighbor <S+S- + S-S+> correlation
    for t in np.arange(0,t,dt):
        Ub = U@np.diag(np.exp(-1j*t*v))@U.conj().T@U0 #the total (unitary) Boguliobov transformation
        X = Ub[L::, L::]
        Y = Ub[L::, :L]
        npt.assert_almost_equal((X@X.conj().T+Y@Y.conj().T), np.identity(L), 7)
        npt.assert_almost_equal((X.conj().T@X+Y.T@Y.conj()), np.identity(L), 7)
        npt.assert_almost_equal((X@Y.T+Y@X.T), 0, 7)
        npt.assert_almost_equal((X.T@Y.conj()+Y.conj().T@X), 0,  7)

        X_inv = LA.inv(X)
        M = np.zeros((2*L,2*L), dtype='complex')
        M[:L, :L] = -Y@(X_inv.conj())
        Nc = np.zeros((2*L,2*L), dtype='complex')
        Nc[L::, L::] = -(X_inv.T)@(Y.conj().T)
        A = np.kron(np.array([[-1,1],[0,-1]]), np.identity(L))
        
        S = np.concatenate((np.concatenate((A,Nc),axis=1),np.concatenate((M,-A.T),axis=1)),axis=0)
        Delta = LA.inv(S)
        npt.assert_almost_equal(np.abs(LA.det(X))*np.sqrt(LA.det(S)), 1, 7)

        #measure z-magnetization
        mz = 0
        for i in range(L):
            ni = Delta[i+L, i]
            assert abs(ni.imag) < 1e-10
            mz += 1-2*ni.real
        mag.append(mz/L)

        #measure sigmaz-sigmaz correlations
        i,j = 0, 5
        ninj =  -0.25*(-Delta[j+L, i+3*L] + Delta[i+L, j+3*L])*(-Delta[j+2*L, i] + Delta[i+2*L, j]) \
                +0.25*(-Delta[j+L, j] + Delta[j+2*L, j+3*L])*(-Delta[i+L, i] + Delta[i+2*L, i+3*L]) \
                -0.25*(-Delta[j+L, i] + Delta[i+2*L, j+3*L])*(-Delta[i+L, j] + Delta[j+2*L, i+3*L])
        ni = Delta[i+L, i]
        nj = Delta[j+L, j]
        s = 1-2*ni-2*nj+4*ninj
        assert abs(s.imag) < 1e-10
        szsz.append(s.real)

        #measure sigma+sigma- correlatios
        s = 0
        for i in range(L-1):
            j = i+1
            ci = 0.5*(Delta[j+L, i] - Delta[i+2*L, j+3*L] + Delta[i+L, j] - Delta[j+2*L, i+3*L])
            assert abs(ci.imag) < 1e-10
            s += ci.real
        spsm.append(s/(L-1))
    return np.array(mag), np.array(szsz), np.array(spsm)

@pytest.mark.parametrize('algorithm', [
    'TEBD', 'TDVP'
])
def test_time_methods(algorithm):
    L=6
    g=1.5

    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * L #prepare system in spin polarized state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dt = 0.01
    N_steps = 2
    t = 0.5 #total time evolution
    params = {
        'active_sites': 2,
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
        eng = tebd.Engine(psi, M, params)
    elif algorithm == 'TDVP':
        eng = tdvp.Engine(psi, M, params)
    else:
        raise ValueError("test works only for TEDB and TDVP so far")

    mag = [np.mean(psi.expectation_value("Sigmaz"))]
    i,j = 0, 5
    szsz = [psi.correlation_function("Sigmaz", "Sigmaz", [i], [j])[0,0]]
    spsm = [np.mean(psi.correlation_function("Sp", "Sm").diagonal(1)+psi.correlation_function("Sp", "Sm").diagonal(-1))]
    
    for ti in np.arange(0, t, dt*N_steps):
        eng.run()

        mag.append(np.mean(psi.expectation_value("Sigmaz")))
        i,j = 0, 5
        szsz.append(psi.correlation_function("Sigmaz", "Sigmaz", [i], [j])[0,0])
        spsm.append(np.mean(psi.correlation_function("Sp", "Sm").diagonal(1)+psi.correlation_function("Sp", "Sm").diagonal(-1)))
    
    m_exact, szsz_exact, spsm_exact = exact_expectation(L, g, t, dt*N_steps)
    npt.assert_almost_equal(np.array(mag)[:-1], m_exact, 4)
    npt.assert_almost_equal(np.array(szsz)[:-1], szsz_exact, 4)
    npt.assert_almost_equal(np.array(spsm)[:-1], spsm_exact, 4)

    #TODO add MPO evolution
    #TODO add infinite MPS (?)

