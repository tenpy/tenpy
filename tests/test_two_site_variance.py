"""Tests for the two-site (and n-site) variance error measure of an MPS."""
# Copyright (C) TeNPy Developers, Apache license

import copy

import numpy as np
import pytest
from random_test import random_MPS

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag, get_full_wavefunction, get_numpy_Hamiltonian
from tenpy.models.aklt import AKLTChain
from tenpy.models.fermions_spinless import FermionChain
from tenpy.models.spins_nnn import SpinChainNNN2
from tenpy.models.tf_ising import TFIChain, TFIModel
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.simulations import measurement
from tenpy.simulations.ground_state_search import GroundStateSearch
from tenpy.tools.fit import linear_fit

# The repository turns warnings into errors; an implementation may legitimately warn, for
# example while converging environments of an infinite MPS.
pytestmark = pytest.mark.filterwarnings('ignore')

NNN_PARAMS = dict(Jx=1.0, Jy=0.8, Jz=0.5, Jxp=0.4, Jyp=0.3, Jzp=0.6, hz=0.2, conserve=None, bc_MPS='finite')
TOL = 1.0e-8


def random_state_on_sites(sites, chi, seed, canonical=True):
    """Deterministic random finite MPS with trivial charges on the given sites."""
    np.random.seed(seed)
    L = len(sites)
    triv = random_MPS(L, sites[0].dim, chi, bc='finite', form=None)
    Bflat = [triv.get_B(i, form=None).to_ndarray().transpose(1, 0, 2) for i in range(L)]
    psi = MPS.from_Bflat(sites, Bflat, bc='finite', form=None, unit_cell_width=L)
    if canonical:
        psi.canonical_form()
    return psi


def dense_left_block(psi, i):
    """Isometry of shape (d**i, chi_i) formed by the left-canonical tensors A_0 ... A_{i-1}."""
    blk = np.ones((1, 1))
    for j in range(i):
        A = psi.get_B(j, form='A').to_ndarray()  # vL p vR
        blk = np.einsum('xa,apb->xpb', blk, A).reshape(-1, A.shape[2])
    return blk


def dense_right_block(psi, i):
    """Isometry of shape (d**(L-i), chi_i) formed by the right-canonical tensors B_i ... B_{L-1}."""
    blk = np.ones((1, 1))
    for j in range(psi.L - 1, i - 1, -1):
        B = psi.get_B(j, form='B').to_ndarray()  # vL p vR
        blk = np.einsum('apb,bx->apx', B, blk).reshape(B.shape[0], -1)
    return blk.T


def dense_n_site_variance(H, psi, n):
    """Independent dense evaluation: ||P H psi||^2 - E^2 with P the projector onto the span of
    all states that differ from psi on at most n neighboring sites."""
    L = psi.L
    d = psi.sites[0].dim
    vec = get_full_wavefunction(psi, undo_sort_charge=False)
    vec = vec / np.linalg.norm(vec)
    Hvec = H @ vec
    E = np.vdot(vec, Hvec).real
    cols = []
    for i in range(L - n + 1):
        left = dense_left_block(psi, i)  # (d^i, chi_i)
        right = dense_right_block(psi, i + n)  # (d^(L-i-n), chi_{i+n})
        block = np.einsum('xa,yb->xyab', left, right)
        basis = np.zeros((d**i, d**n, d ** (L - i - n), left.shape[1], d**n, right.shape[1]), dtype=complex)
        for s in range(d**n):
            basis[:, s, :, :, s, :] = block
        cols.append(basis.reshape(d**L, -1))
    Mall = np.concatenate(cols, axis=1)
    U, S, _ = np.linalg.svd(Mall, full_matrices=False)
    Q = U[:, S > 1.0e-9]
    P_Hvec = Q @ (Q.conj().T @ Hvec)
    return np.linalg.norm(P_Hvec) ** 2 - E**2


def dense_full_variance(H, psi):
    vec = get_full_wavefunction(psi, undo_sort_charge=False)
    vec = vec / np.linalg.norm(vec)
    Hvec = H @ vec
    return np.vdot(Hvec, Hvec).real - np.vdot(vec, Hvec).real ** 2


def dense_block_terms(H, psi, n):
    """Independent dense evaluation of the individual n-site block contributions: the squared
    norm of H psi projected onto the n-site variations of the block starting at site i that are
    orthogonal to psi and to all variations of fewer sites."""
    L = psi.L
    d = psi.sites[0].dim
    vec = get_full_wavefunction(psi, undo_sort_charge=False)
    vec = vec / np.linalg.norm(vec)
    Hvec = H @ vec
    terms = []
    for i in range(L - n + 1):
        left = dense_left_block(psi, i)  # (d^i, chi_i)
        right = dense_right_block(psi, i + n)  # (d^(L-i-n), chi_{i+n})
        C = Hvec.reshape(d**i, d**n, d ** (L - i - n))
        C = np.einsum('xa,xsy,yb->asb', left.conj(), C, right.conj())  # (chi_i, d^n, chi_{i+n})
        A = psi.get_B(i, form='A').to_ndarray()  # vL p vR
        A = A.transpose(0, 1, 2).reshape(-1, A.shape[2])  # (chi_i d, chi_{i+1})
        C = C.reshape(A.shape[0], -1)
        C = C - A @ (A.conj().T @ C)  # orthogonal to A_i on the left legs
        if n > 1:
            B = psi.get_B(i + n - 1, form='B').to_ndarray()  # vL p vR
            B = B.reshape(B.shape[0], -1)  # (chi_{i+n-1}, d chi_{i+n})
            C = C.reshape(-1, B.shape[1])
            C = C - (C @ B.conj().T) @ B  # orthogonal to B_{i+n-1} on the right legs
        terms.append(np.linalg.norm(C) ** 2)
    return np.array(terms)


@pytest.mark.parametrize('chi', [1, 3, 6])
def test_nearest_neighbor_equals_full_variance(chi):
    L = 7
    M = TFIChain(dict(L=L, J=1.0, g=0.7, conserve=None, bc_MPS='finite'))
    psi = random_state_on_sites(M.lat.mps_sites(), chi, seed=100 + chi)
    H = get_numpy_Hamiltonian(M, undo_sort_charge=False)
    var_dense = dense_full_variance(H, psi)
    var_naive = M.H_MPO.variance(psi)
    assert abs(var_naive - var_dense) < TOL * max(1.0, abs(var_dense))
    two_site = float(M.H_MPO.two_site_variance(psi))
    assert abs(two_site - var_dense) < TOL * max(1.0, abs(var_dense))
    assert two_site >= -TOL


@pytest.mark.parametrize('chi', [1, 2, 4])
def test_longer_range_against_dense_projection(chi):
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), chi, seed=200 + chi)
    H = get_numpy_Hamiltonian(M, undo_sort_charge=False)
    var_full = dense_full_variance(H, psi)
    expected = [dense_n_site_variance(H, psi, n) for n in [1, 2, 3]]
    two_site = float(M.H_MPO.two_site_variance(psi))
    assert abs(two_site - expected[1]) < TOL * max(1.0, abs(expected[1]))
    assert -TOL <= two_site <= var_full + TOL
    results = [float(M.H_MPO.n_site_variance(psi, n)) for n in [1, 2, 3]]
    for res, exp in zip(results, expected):
        assert abs(res - exp) < TOL * max(1.0, abs(exp))
    assert abs(results[1] - two_site) < TOL
    assert results[0] <= results[1] + TOL <= results[2] + 2 * TOL
    # next-nearest neighbor terms: the three-site variance is the full variance
    assert abs(results[2] - var_full) < TOL * max(1.0, abs(var_full))
    # n_sites defaults to two
    assert abs(float(M.H_MPO.n_site_variance(psi)) - two_site) < TOL
    total_default, terms_default = M.H_MPO.n_site_variance(psi, return_terms=True)
    total_2, terms_2 = M.H_MPO.n_site_variance(psi, 2, return_terms=True)
    assert abs(float(total_default) - float(total_2)) < TOL
    assert len(terms_default) == len(terms_2) == 2
    # the individual block contributions, against the independent dense evaluation
    total_3, terms_3 = M.H_MPO.n_site_variance(psi, 3, return_terms=True)
    for n, terms_n in enumerate(terms_3, start=1):
        expected_terms = dense_block_terms(H, psi, n)
        assert np.allclose(np.asarray(terms_n, dtype=float), expected_terms, atol=TOL, rtol=TOL)
    for n, terms_n in enumerate(terms_default, start=1):
        assert np.allclose(np.asarray(terms_n, dtype=float), np.asarray(terms_3[n - 1], dtype=float), atol=TOL)


def test_two_sites_equals_full_variance():
    M = SpinChainNNN2(dict(L=2, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), 2, seed=3)
    H = get_numpy_Hamiltonian(M, undo_sort_charge=False)
    var_full = dense_full_variance(H, psi)
    assert abs(float(M.H_MPO.two_site_variance(psi)) - var_full) < TOL
    # a charge-conserving and a fermionic two-site system
    M = XXZChain(dict(L=2, Jxx=1.0, Jz=0.7, hz=0.3, bc_MPS='finite'))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'], unit_cell_width=2)
    var_full = M.H_MPO.variance(psi)
    assert abs(float(M.H_MPO.two_site_variance(psi)) - var_full) < TOL
    M = FermionChain(dict(L=2, J=1.0, V=0.7, mu=0.3, bc_MPS='finite'))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['empty', 'full'], unit_cell_width=2)
    var_full = M.H_MPO.variance(psi)
    assert abs(float(M.H_MPO.two_site_variance(psi)) - var_full) < TOL


def test_individual_terms():
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), 3, seed=17)
    total, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    for array in [one_site, two_site]:
        assert isinstance(array, np.ndarray) and array.ndim == 1
    one_site = np.asarray(one_site, dtype=float)
    two_site = np.asarray(two_site, dtype=float)
    assert one_site.shape == (L,)
    assert two_site.shape == (L - 1,)
    assert np.all(one_site >= -TOL)
    assert np.all(two_site >= -TOL)
    assert abs(float(total) - one_site.sum() - two_site.sum()) < TOL
    assert abs(float(total) - float(M.H_MPO.two_site_variance(psi))) < TOL
    total3, terms = M.H_MPO.n_site_variance(psi, 3, return_terms=True)
    assert len(terms) == 3
    for array in terms:
        assert isinstance(array, np.ndarray) and array.ndim == 1
    assert [len(np.asarray(t)) for t in terms] == [L, L - 1, L - 2]
    assert np.allclose(np.asarray(terms[0], dtype=float), one_site, atol=TOL)
    assert np.allclose(np.asarray(terms[1], dtype=float), two_site, atol=TOL)
    assert abs(float(total3) - sum(np.sum(np.asarray(t, dtype=float)) for t in terms)) < TOL
    total1, terms1 = M.H_MPO.n_site_variance(psi, 1, return_terms=True)
    assert len(terms1) == 1
    assert np.allclose(np.asarray(terms1[0], dtype=float), one_site, atol=TOL)
    # at the largest allowed block size the variations span everything
    totalL, termsL = M.H_MPO.n_site_variance(psi, L, return_terms=True)
    assert len(termsL) == L
    assert [len(np.asarray(t)) for t in termsL] == [L - k for k in range(L)]
    assert abs(float(totalL) - sum(np.sum(np.asarray(t, dtype=float)) for t in termsL)) < TOL
    assert abs(float(totalL) - M.H_MPO.variance(psi)) < TOL


def test_eigenstates_vanish():
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    ED = ExactDiag(M)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization()
    for k in [0, 1, 5]:
        vec = ED.V.take_slice(k, axes='ps*')
        psi = ED.full_to_mps(vec)
        total, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
        assert abs(float(total)) < TOL
        assert np.all(np.abs(np.asarray(one_site, dtype=float)) < TOL)
        assert np.all(np.abs(np.asarray(two_site, dtype=float)) < TOL)
        assert abs(float(M.H_MPO.n_site_variance(psi, 3))) < TOL
    # product states (polarized along x) are eigenstates of the classical Ising model
    M = TFIChain(dict(L=L, J=1.0, g=0.0, conserve=None, bc_MPS='finite'))
    plus, minus = np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)
    psi = MPS.from_product_state(M.lat.mps_sites(), [plus, minus, minus, plus, plus, plus], unit_cell_width=L)
    assert abs(float(M.H_MPO.two_site_variance(psi))) < TOL
    assert abs(float(M.H_MPO.variance(psi))) < TOL  # sanity check: really an eigenstate


class ShiftedTFIChain(TFIChain):
    def init_terms(self, model_params):
        super().init_terms(model_params)
        shift = model_params.get('shift', 0.0, 'real')
        self.add_onsite(shift, 0, 'Id')


@pytest.mark.parametrize('shift', [37.0, -2.5])
def test_constant_shift_invariance(shift):
    L = 6
    params = dict(L=L, J=1.0, g=0.9, conserve=None, bc_MPS='finite')
    M = ShiftedTFIChain(dict(params))
    M_shifted = ShiftedTFIChain(dict(shift=shift, **params))
    psi = random_state_on_sites(M.lat.mps_sites(), 4, seed=5)
    E = M.H_MPO.expectation_value(psi)
    assert abs(M_shifted.H_MPO.expectation_value(psi) - E - shift * L) < 1.0e-6  # the shift is really there
    two_site = float(M.H_MPO.two_site_variance(psi))
    assert abs(float(M_shifted.H_MPO.two_site_variance(psi)) - two_site) < TOL
    assert abs(float(M_shifted.H_MPO.n_site_variance(psi, 3)) - float(M.H_MPO.n_site_variance(psi, 3))) < TOL


def run_dmrg(M, psi, chi):
    options = {
        'trunc_params': {'chi_max': chi, 'svd_min': 1.0e-12},
        'mixer': False,
        'N_sweeps_check': 1,
        'min_sweeps': 3,
        'max_sweeps': 3,
        'max_E_err': 1.0e-14,
        'max_trunc_err': 1.0,
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, options)
    eng.run()
    return psi


def test_charge_conservation():
    L = 8
    M = XXZChain(dict(L=L, Jxx=1.0, Jz=0.7, hz=0.3, bc_MPS='finite'))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    psi = run_dmrg(M, psi, chi=5)
    var_naive = M.H_MPO.variance(psi)
    assert abs(float(M.H_MPO.two_site_variance(psi)) - var_naive) < TOL
    # fermions with Jordan-Wigner strings
    M = FermionChain(dict(L=L, J=1.0, V=0.7, mu=0.3, bc_MPS='finite'))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['empty', 'full'] * (L // 2), unit_cell_width=L)
    psi = run_dmrg(M, psi, chi=5)
    var_naive = M.H_MPO.variance(psi)
    assert abs(float(M.H_MPO.two_site_variance(psi)) - var_naive) < TOL


def test_explicit_plus_hc():
    L = 8
    params = dict(L=L, J=1.0, V=0.7, mu=0.3, bc_MPS='finite')
    M = FermionChain(dict(params))
    M_hc = FermionChain(dict(explicit_plus_hc=True, **params))
    assert M_hc.H_MPO.explicit_plus_hc
    psi = MPS.from_product_state(M.lat.mps_sites(), ['empty', 'full'] * (L // 2), unit_cell_width=L)
    psi = run_dmrg(M, psi, chi=4)
    var_naive = M.H_MPO.variance(psi)
    two_site = float(M_hc.H_MPO.two_site_variance(psi))
    assert abs(two_site - var_naive) < TOL
    assert abs(M_hc.H_MPO.variance(psi) - var_naive) < TOL  # the full variance accepts it too
    assert abs(float(M_hc.H_MPO.n_site_variance(psi, 3)) - var_naive) < TOL
    assert abs(float(M_hc.H_MPO.n_site_variance(psi, 1)) - float(M.H_MPO.n_site_variance(psi, 1))) < TOL


def test_norm_and_gauge_independence():
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), 3, seed=42)
    expected = float(M.H_MPO.two_site_variance(psi))
    psi_norm = psi.copy()
    psi_norm.norm = 2.5
    assert abs(float(M.H_MPO.two_site_variance(psi_norm)) - expected) < TOL
    assert psi_norm.norm == 2.5
    # a non-canonical representation of the same state: insert random gauge transformations
    psi_raw = psi.copy()
    np.random.seed(43)
    for i in range(1, L):
        B_prev = psi_raw.get_B(i - 1, form=None)
        B_next = psi_raw.get_B(i, form=None)
        leg = B_prev.get_leg('vR')
        x = np.random.standard_normal((leg.ind_len, leg.ind_len)) + 2.0 * np.eye(leg.ind_len)
        X = npc.Array.from_ndarray(x, [leg.conj(), leg], labels=['vL', 'vR'])
        X_inv = npc.Array.from_ndarray(np.linalg.inv(x), [leg.conj(), leg], labels=['vL', 'vR'])
        psi_raw.set_B(i - 1, npc.tensordot(B_prev, X, axes=('vR', 'vL')), form=None)
        psi_raw.set_B(i, npc.tensordot(X_inv, B_next, axes=('vR', 'vL')), form=None)
    psi_raw.norm = 1.7
    psi_check = psi_raw.copy()
    psi_check.canonical_form()
    psi_check.norm = 1.0
    assert abs(abs(psi_check.overlap(psi)) - 1.0) < 1.0e-10  # the same state
    Bs_before = [psi_raw.get_B(i, form=None).copy() for i in range(L)]
    form_before = list(psi_raw.form)
    norm_before = psi_raw.norm
    assert abs(float(M.H_MPO.two_site_variance(psi_raw)) - expected) < TOL
    assert abs(float(M.H_MPO.n_site_variance(psi_raw, 3)) - float(M.H_MPO.n_site_variance(psi, 3))) < TOL
    # the input must not have been modified
    for i in range(L):
        assert npc.norm(psi_raw.get_B(i, form=None) - Bs_before[i]) < 1.0e-14
    assert list(psi_raw.form) == form_before
    assert psi_raw.norm == norm_before


def test_single_site():
    # a single site carries an onsite Hamiltonian only, and n_sites=1 is the largest block
    M = TFIModel(dict(L=1, J=1.0, g=0.7, conserve=None, bc_MPS='finite'))
    B = np.array([[[0.6]], [[0.8]]])  # p, vL, vR
    psi_canonical = MPS.from_Bflat(M.lat.mps_sites(), [B], bc='finite', form='B', unit_cell_width=1)
    var_full = M.H_MPO.variance(psi_canonical)
    # the same normalized state up to a phase, with no canonical form set
    psi = MPS.from_Bflat(M.lat.mps_sites(), [np.exp(0.7j) * B], bc='finite', form=None, unit_cell_width=1)
    total, terms = M.H_MPO.n_site_variance(psi, 1, return_terms=True)
    assert abs(float(total) - var_full) < TOL
    assert len(terms) == 1
    array = terms[0]
    assert isinstance(array, np.ndarray) and array.ndim == 1 and array.shape == (1,)
    assert float(array[0]) >= -TOL
    assert abs(float(array[0]) - float(total)) < TOL
    # the canonical representation gives the same value, and the input is not modified
    assert abs(float(M.H_MPO.n_site_variance(psi_canonical, 1)) - var_full) < TOL
    assert list(psi.form) == [None]
    with pytest.raises(ValueError):
        M.H_MPO.two_site_variance(psi)


def test_approximately_canonical_state():
    """A state can claim a canonical form that its tensors only approximately have, as after an
    unconverged algorithm run; the result must follow the tensors."""
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), 3, seed=77)
    expected = float(M.H_MPO.two_site_variance(psi))
    # a gauge on one bond leaves the state alone but breaks the canonical form it still claims
    psi_stale = psi.copy()
    np.random.seed(78)
    leg = psi_stale.get_B(2, form='B').get_leg('vR')
    x = np.diag(np.linspace(1.5, 3.0, leg.ind_len))
    X = npc.Array.from_ndarray(x, [leg.conj(), leg], labels=['vL', 'vR'])
    X_inv = npc.Array.from_ndarray(np.linalg.inv(x), [leg.conj(), leg], labels=['vL', 'vR'])
    psi_stale.set_B(2, npc.tensordot(psi_stale.get_B(2, form='B'), X, axes=('vR', 'vL')), form='B')
    psi_stale.set_B(3, npc.tensordot(X_inv, psi_stale.get_B(3, form='B'), axes=('vR', 'vL')), form='B')
    assert all(f is not None for f in psi_stale.form)  # it still claims to be canonical
    assert np.linalg.norm(psi_stale.norm_test()) > 1.0e-3  # but it is not
    assert abs(float(M.H_MPO.two_site_variance(psi_stale)) - expected) < TOL
    assert abs(float(M.H_MPO.n_site_variance(psi_stale, 3)) - float(M.H_MPO.n_site_variance(psi, 3))) < TOL


def run_idmrg(M, psi, chi):
    options = {
        'trunc_params': {'chi_max': chi, 'svd_min': 1.0e-12},
        'mixer': False,
        'N_sweeps_check': 2,
        'max_sweeps': 30,
        'max_E_err': 1.0e-13,
        'max_trunc_err': 1.0,
    }
    dmrg.TwoSiteDMRGEngine(psi, M, options).run()
    return psi


def test_infinite_mps():
    params = dict(J=1.0, g=1.2, conserve=None, bc_MPS='infinite')
    M = TFIChain(dict(L=2, **params))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'up'], bc='infinite', unit_cell_width=2)
    psi = run_idmrg(M, psi, chi=4)
    density, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    density = float(density)
    one_site = np.asarray(one_site, dtype=float)
    two_site = np.asarray(two_site, dtype=float)
    assert one_site.shape == (2,) and two_site.shape == (2,)  # one per site and per bond
    assert np.all(one_site >= -TOL) and np.all(two_site >= -TOL)
    assert abs(density - (one_site.sum() + two_site.sum()) / 2) < TOL  # a density per site
    assert abs(density - float(M.H_MPO.n_site_variance(psi, 2))) < TOL
    assert float(M.H_MPO.n_site_variance(psi, 1)) <= density + TOL
    # blocks may be longer than the unit cell
    total_3, terms_3 = M.H_MPO.n_site_variance(psi, 3, return_terms=True)
    assert len(terms_3) == 3
    assert all(np.asarray(t).shape == (2,) for t in terms_3)
    assert np.allclose(np.asarray(terms_3[0], dtype=float), one_site, atol=TOL)
    assert np.allclose(np.asarray(terms_3[1], dtype=float), two_site, atol=TOL)
    assert abs(float(total_3) - sum(np.sum(np.asarray(t, dtype=float)) for t in terms_3) / 2) < TOL
    # the density does not depend on the choice of unit cell
    psi4 = psi.copy()
    psi4.enlarge_mps_unit_cell(2)
    M4 = TFIChain(dict(L=4, **params))
    density4, one4, two4 = M4.H_MPO.two_site_variance(psi4, return_terms=True)
    assert abs(float(density4) - density) < TOL
    assert np.allclose(np.asarray(one4, dtype=float), np.tile(one_site, 2), atol=TOL)
    assert np.allclose(np.asarray(two4, dtype=float), np.tile(two_site, 2), atol=TOL)
    assert abs(float(M4.H_MPO.n_site_variance(psi4, 3)) - float(total_3)) < TOL
    # the norm attribute is ignored here as well
    psi_norm = psi.copy()
    psi_norm.norm = 0.3
    assert abs(float(M.H_MPO.two_site_variance(psi_norm)) - density) < TOL


def test_infinite_unequal_unit_cells():
    params = dict(J=1.0, g=0.8, conserve=None, bc_MPS='infinite')
    M2 = TFIChain(dict(L=2, **params))
    psi = MPS.from_product_state(M2.lat.mps_sites(), ['up', 'down'], bc='infinite', unit_cell_width=2)
    density, one_site, two_site = M2.H_MPO.two_site_variance(psi, return_terms=True)
    one_site = np.asarray(one_site, dtype=float)
    two_site = np.asarray(two_site, dtype=float)
    # an MPO whose unit cell is three sites long: the common period of the two cells is six
    M3 = TFIChain(dict(L=3, **params))
    density6, one6, two6 = M3.H_MPO.two_site_variance(psi, return_terms=True)
    assert np.asarray(one6).shape == (6,) and np.asarray(two6).shape == (6,)
    assert abs(float(density6) - float(density)) < TOL
    assert np.allclose(np.asarray(one6, dtype=float), np.tile(one_site, 3), atol=TOL)
    assert np.allclose(np.asarray(two6, dtype=float), np.tile(two_site, 3), atol=TOL)
    # and the other way round
    psi6 = psi.copy()
    psi6.enlarge_mps_unit_cell(3)
    density_mixed, one_mixed, _ = M2.H_MPO.two_site_variance(psi6, return_terms=True)
    assert abs(float(density_mixed) - float(density)) < TOL
    assert np.asarray(one_mixed).shape == (6,)


def test_infinite_translation():
    L = 4
    M = TFIChain(dict(L=L, J=1.0, g=0.8, conserve=None, bc_MPS='infinite'))
    sites = M.lat.mps_sites()
    p_state = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([np.cos(0.3), np.sin(0.3)]),
        np.array([0.0, 1.0]),
    ]
    psi = MPS.from_product_state(sites, p_state, bc='infinite', unit_cell_width=L)
    density, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    one_site = np.asarray(one_site, dtype=float)
    two_site = np.asarray(two_site, dtype=float)
    assert one_site.shape == (L,) and two_site.shape == (L,)
    # this state is inhomogeneous, so rolling the contributions is distinguishable
    assert not np.allclose(one_site, np.roll(one_site, -1), atol=TOL)
    assert not np.allclose(two_site, np.roll(two_site, -1), atol=TOL)
    psi_rolled = MPS(
        sites,
        [psi.get_B(i + 1) for i in range(L)],
        [psi.get_SL(i + 1) for i in range(L + 1)],
        bc='infinite',
        form='B',
        unit_cell_width=L,
    )
    density_rolled, one_rolled, two_rolled = M.H_MPO.two_site_variance(psi_rolled, return_terms=True)
    assert abs(float(density_rolled) - float(density)) < TOL
    assert np.allclose(np.asarray(one_rolled, dtype=float), np.roll(one_site, -1), atol=TOL)
    assert np.allclose(np.asarray(two_rolled, dtype=float), np.roll(two_site, -1), atol=TOL)


def test_infinite_matches_dense_bulk():
    """A product state has strictly zero correlations, so for a nearest neighbour Hamiltonian the
    contributions in the bulk of a long chain are exactly the ones of the infinite state."""
    L = 4
    params = dict(J=1.0, g=0.8, conserve=None)
    p_state = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([np.cos(0.3), np.sin(0.3)]),
        np.array([0.0, 1.0]),
    ]
    M = TFIChain(dict(L=L, bc_MPS='infinite', **params))
    psi = MPS.from_product_state(M.lat.mps_sites(), p_state, bc='infinite', unit_cell_width=L)
    density, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    one_site = np.asarray(one_site, dtype=float)
    two_site = np.asarray(two_site, dtype=float)
    # the same state on three unit cells of a finite chain, with an independent dense reference
    M_fin = TFIChain(dict(L=3 * L, bc_MPS='finite', **params))
    psi_fin = MPS.from_product_state(M_fin.lat.mps_sites(), p_state * 3, unit_cell_width=3 * L)
    H = get_numpy_Hamiltonian(M_fin, undo_sort_charge=False)
    middle = slice(L, 2 * L)
    assert np.allclose(one_site, dense_block_terms(H, psi_fin, 1)[middle], atol=TOL)
    assert np.allclose(two_site, dense_block_terms(H, psi_fin, 2)[middle], atol=TOL)
    assert abs(float(density) - (one_site.sum() + two_site.sum()) / L) < TOL
    assert one_site.max() > 0.5  # the reference values are not all zero


def test_infinite_eigenstate_and_explicit_plus_hc():
    M = AKLTChain({'L': 2, 'bc_MPS': 'infinite', 'sort_charge': True})
    psi = M.psi_AKLT()
    density, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    assert abs(float(density)) < TOL  # an exact eigenstate
    assert np.all(np.abs(np.asarray(one_site, dtype=float)) < TOL)
    assert np.all(np.abs(np.asarray(two_site, dtype=float)) < TOL)
    params = dict(L=2, J=1.0, V=0.5, mu=0.2, bc_MPS='infinite')
    M = FermionChain(dict(params))
    M_hc = FermionChain(dict(explicit_plus_hc=True, **params))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['empty', 'full'], bc='infinite', unit_cell_width=2)
    psi = run_idmrg(M, psi, chi=6)
    density = float(M.H_MPO.two_site_variance(psi))
    assert abs(float(M_hc.H_MPO.two_site_variance(psi)) - density) < TOL
    assert abs(float(M_hc.H_MPO.n_site_variance(psi, 3)) - float(M.H_MPO.n_site_variance(psi, 3))) < TOL


def test_invalid_arguments():
    L = 6
    M = TFIChain(dict(L=L, J=1.0, g=0.7, conserve=None, bc_MPS='finite'))
    psi = random_state_on_sites(M.lat.mps_sites(), 2, seed=1)
    M_other = TFIChain(dict(L=L + 1, J=1.0, g=0.7, conserve=None, bc_MPS='finite'))
    with pytest.raises(ValueError):
        M_other.H_MPO.two_site_variance(psi)
    M_inf = TFIChain(dict(L=2, J=1.0, g=0.7, conserve=None, bc_MPS='infinite'))
    with pytest.raises(ValueError):
        M_inf.H_MPO.two_site_variance(psi)
    with pytest.raises(ValueError):
        M_inf.H_MPO.n_site_variance(psi, 1)
    psi_inf = MPS.from_product_state(M_inf.lat.mps_sites(), ['up', 'up'], bc='infinite', unit_cell_width=2)
    with pytest.raises(ValueError):
        M.H_MPO.two_site_variance(psi_inf)
    with pytest.raises(ValueError):
        M.H_MPO.n_site_variance(psi, 0)
    with pytest.raises(ValueError):
        M.H_MPO.n_site_variance(psi, L + 1)
    # still fine, and equal to the full variance since the variations span everything:
    assert abs(float(M.H_MPO.n_site_variance(psi, L)) - M.H_MPO.variance(psi)) < TOL
    # segment boundary conditions are not supported either, on neither side
    psi_seg = MPS.from_product_state(M.lat.mps_sites(), ['up'] * L, bc='segment', unit_cell_width=L)
    with pytest.raises(ValueError):
        M.H_MPO.two_site_variance(psi_seg)
    H_seg = M_inf.H_MPO.extract_segment(0, L - 1)
    assert H_seg.bc == 'segment'
    with pytest.raises(ValueError):
        H_seg.two_site_variance(psi)
    with pytest.raises(ValueError):
        H_seg.n_site_variance(psi, 1)


MAX_SWEEPS = 8


def dmrg_options(**kwargs):
    options = {
        'trunc_params': {'chi_max': 4},
        'mixer': True,
        'N_sweeps_check': 1,
        'min_sweeps': 2,
        'max_sweeps': MAX_SWEEPS,
    }
    options.update(kwargs)
    return options


def test_dmrg_option():
    L = 10
    M = TFIChain(dict(L=L, J=1.0, g=1.5, conserve=None, bc_MPS='finite'))

    def initial_state():
        return MPS.from_product_state(M.lat.mps_sites(), ['up'] * L, unit_cell_width=L)

    eng = dmrg.SingleSiteDMRGEngine(initial_state(), M, dmrg_options())
    eng.run()
    assert not eng.sweep_stats.get('two_site_variance')  # not computed by default
    N_default = len(eng.sweep_stats['E'])

    eng = dmrg.SingleSiteDMRGEngine(initial_state(), M, dmrg_options(compute_two_site_variance=True))
    E, psi = eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    assert len(recorded) == len(eng.sweep_stats['E']) == N_default
    assert all(v >= -TOL for v in recorded)
    assert abs(recorded[-1] - M.H_MPO.variance(psi)) < TOL  # nearest neighbor: exact
    assert abs(recorded[-1] - float(M.H_MPO.two_site_variance(psi))) < TOL

    # a threshold the recorded values never reach can never be met, so no iteration counts as
    # converged and the run uses every allowed check
    eng = dmrg.SingleSiteDMRGEngine(initial_state(), M, dmrg_options(max_two_site_variance=1.0e-30))
    eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    assert len(recorded) == len(eng.sweep_stats['E'])
    assert recorded[-1] < 1.0e-30 or len(eng.sweep_stats['E']) >= MAX_SWEEPS

    # a loose threshold changes nothing
    eng = dmrg.SingleSiteDMRGEngine(initial_state(), M, dmrg_options(max_two_site_variance=1.0))
    eng.run()
    assert len(eng.sweep_stats['E']) == N_default
    assert len(eng.sweep_stats['two_site_variance']) == N_default

    # an explicitly disabled diagnostic does not prevent the threshold from being evaluated
    eng = dmrg.SingleSiteDMRGEngine(
        initial_state(), M, dmrg_options(max_two_site_variance=1.0e-30, compute_two_site_variance=False)
    )
    eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    assert len(recorded) == len(eng.sweep_stats['E'])
    assert recorded[-1] < 1.0e-30 or len(eng.sweep_stats['E']) >= MAX_SWEEPS

    # a model with next-nearest neighbor terms, where the two-site variance is strictly below the
    # full variance, so the recorded quantity is identified unambiguously
    M_nnn = SpinChainNNN2(dict(L=8, **NNN_PARAMS))

    def nnn_state():
        return MPS.from_product_state(M_nnn.lat.mps_sites(), ['up', 'down'] * 4, unit_cell_width=8)

    nnn_options = dict(
        trunc_params={'chi_max': 2},
        mixer=False,
        N_sweeps_check=1,
        min_sweeps=2,
        max_sweeps=20,
        max_trunc_err=1.0,
    )
    eng = dmrg.TwoSiteDMRGEngine(nnn_state(), M_nnn, dict(nnn_options, compute_two_site_variance=True))
    E, psi = eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    n_nnn = len(eng.sweep_stats['E'])
    two_site = float(M_nnn.H_MPO.two_site_variance(psi))
    var_full = M_nnn.H_MPO.variance(psi)
    assert abs(recorded[-1] - two_site) < TOL

    # a threshold between the two-site and the full variance must not delay convergence
    eng = dmrg.TwoSiteDMRGEngine(
        nnn_state(), M_nnn, dict(nnn_options, max_two_site_variance=0.5 * (two_site + var_full))
    )
    eng.run()
    assert len(eng.sweep_stats['E']) == n_nnn

    # one that the recorded values never reach keeps every iteration from counting as converged
    eng = dmrg.TwoSiteDMRGEngine(nnn_state(), M_nnn, dict(nnn_options, max_two_site_variance=1.0e-30))
    eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    assert recorded[-1] < 1.0e-30 or len(eng.sweep_stats['E']) >= nnn_options['max_sweeps']

    # two-site engine with several sweeps per convergence check: one entry per recorded energy
    eng = dmrg.TwoSiteDMRGEngine(
        initial_state(), M, dmrg_options(mixer=False, N_sweeps_check=2, compute_two_site_variance=True)
    )
    E, psi = eng.run()
    recorded = [float(v) for v in eng.sweep_stats['two_site_variance']]
    assert len(recorded) == len(eng.sweep_stats['E']) >= 1
    assert eng.sweep_stats['sweep'][-1] >= 2 * len(recorded)
    assert abs(recorded[-1] - M.H_MPO.variance(psi)) < TOL


def test_extrapolate_in_variance():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    options = {'mixer': False, 'max_sweeps': 6, 'max_trunc_err': 1.0}
    options_before = copy.deepcopy(options)
    chi_list = [2, 4, 8]
    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list)
    assert set(results) >= {'chi', 'E', 'two_site_variance', 'E_extrapolated', 'fit_residuum'}
    chi = np.asarray(results['chi'])
    energies = np.asarray(results['E'], dtype=float)
    variances = np.asarray(results['two_site_variance'], dtype=float)
    assert chi.ndim == energies.ndim == variances.ndim == 1
    assert list(chi) == chi_list
    assert energies.shape == variances.shape == (len(chi_list),)
    assert np.all(variances >= -TOL)
    # the state is optimized in place, so the last record belongs to the state we are left with
    assert abs(variances[-1] - float(M.H_MPO.two_site_variance(psi))) < TOL
    # the extrapolation is the intercept of a linear fit of the energies against the variances
    slope, intercept, residuum = linear_fit(variances, energies)
    assert abs(float(results['E_extrapolated']) - intercept) < TOL
    assert abs(float(results['fit_residuum']) - residuum) < TOL
    assert options == options_before  # the given options are not modified
    with pytest.raises(ValueError):
        dmrg.extrapolate_in_variance(psi, M, options, [4])
    with pytest.raises(ValueError):
        dmrg.extrapolate_in_variance(psi, M, options, [])


def test_extrapolate_two_bond_dimensions():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    options = {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}
    observables = {'mean_Sz': lambda state, model: float(np.mean(state.expectation_value('Sz')))}
    results = dmrg.extrapolate_in_variance(psi, M, options, [2, 4], observables)
    variances = np.asarray(results['two_site_variance'], dtype=float)
    energies = np.asarray(results['E'], dtype=float)
    assert variances.shape == energies.shape == (2,)
    # the extrapolation is the value at zero variance of a least squares fit, here over two points
    matrix = np.vstack([variances, np.ones(2)]).T
    for key, values in [('E', energies), ('mean_Sz', np.asarray(results['mean_Sz'], dtype=float))]:
        expected = np.linalg.lstsq(matrix, values, rcond=None)[0][1]
        extrapolated = results['E_extrapolated'] if key == 'E' else results[key + '_extrapolated']
        assert abs(float(extrapolated) - expected) < TOL
    assert abs(float(results['fit_residuum'])) < TOL


def test_extrapolate_observables():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    calls = []

    def mean_Sz(state, model):
        calls.append((state.L, model.lat.N_sites))
        return float(np.mean(state.expectation_value('Sz')))

    observables = {
        'mean_Sz': mean_Sz,
        'middle_entropy': lambda state, model: float(state.entanglement_entropy()[L // 2 - 1]),
    }
    options = {'mixer': False, 'max_sweeps': 6, 'max_trunc_err': 1.0}
    chi_list = [2, 4, 8]
    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list, observables)
    assert len(calls) == len(chi_list)  # called once after every run
    assert calls == [(L, L)] * len(chi_list)  # called with the state and the model
    variances = np.asarray(results['two_site_variance'], dtype=float)
    for name in observables:
        values = np.asarray(results[name], dtype=float)
        assert values.ndim == 1 and values.shape == (len(chi_list),)
        assert abs(float(results[name + '_extrapolated']) - linear_fit(variances, values)[1]) < TOL
    assert abs(float(results['mean_Sz'][-1]) - float(np.mean(psi.expectation_value('Sz')))) < TOL
    # a name clashing with one of the fixed keys is rejected
    for clashing in ['chi', 'E', 'two_site_variance', 'E_extrapolated', 'fit_residuum']:
        with pytest.raises(ValueError):
            dmrg.extrapolate_in_variance(psi, M, options, chi_list, {clashing: lambda state, model: 0.0})
    # so is a name that another observable would produce
    with pytest.raises(ValueError):
        dmrg.extrapolate_in_variance(
            psi,
            M,
            options,
            chi_list,
            {'x': lambda state, model: 0.0, 'x_extrapolated': lambda state, model: 1.0},
        )


def test_extrapolate_uses_the_two_site_engine(monkeypatch):
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    states = []
    returned = []

    class Recording(dmrg.TwoSiteDMRGEngine):
        def __init__(self, state, model, options, **kwargs):
            states.append(state)
            super().__init__(state, model, options, **kwargs)

        def run(self):
            E, state = super().run()
            E = float(E) + 1000.0 + len(returned)  # a value the caller cannot recompute
            returned.append(E)
            return E, state

    monkeypatch.setattr(dmrg, 'TwoSiteDMRGEngine', Recording)
    chi_list = [2, 4]
    results = dmrg.extrapolate_in_variance(psi, M, {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}, chi_list)
    assert len(states) == len(chi_list)  # one two-site engine per bond dimension
    assert all(state is psi for state in states)  # each continues from the state in place
    assert [float(value) for value in np.asarray(results['E'], dtype=float)] == returned


def test_extrapolate_callback():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    options = {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}
    chi_list = [2, 4]
    seen = []
    observed = []

    def observable(state, model):
        observed.append(len(seen))  # the observables run before the callback
        return float(max(state.chi))

    def callback(state, model):
        seen.append((state, model))
        return 'ignored'

    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list, {'max_chi': observable}, callback)
    assert len(seen) == len(chi_list)  # called once after every run
    assert all(state is psi and model is M for state, model in seen)
    assert observed == list(range(len(chi_list)))  # and after the observables
    assert 'max_chi' in results  # the return value of the callback is ignored
    assert set(results) >= {'chi', 'E', 'two_site_variance', 'E_extrapolated', 'fit_residuum'}


def test_extrapolate_decreasing_bond_dimensions():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    options = {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}
    observables = {'max_chi': lambda state, model: float(max(state.chi))}
    chi_list = [8, 2]
    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list, observables)
    # every run stays within the bond dimension it was asked for
    observed = np.asarray(results['max_chi'], dtype=float)
    assert observed.shape == (len(chi_list),)
    assert all(observed[k] <= chi_list[k] for k in range(len(chi_list)))
    assert max(psi.chi) <= chi_list[-1]
    assert list(np.asarray(results['chi'])) == chi_list


def test_extrapolate_overrides_chi_schedule():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    # a bond dimension schedule in the options must not override the requested bond dimensions
    options = {
        'mixer': False,
        'max_sweeps': 4,
        'max_trunc_err': 1.0,
        'trunc_params': {'chi_max': 2},
        'chi_list': {0: 16},
    }
    options_before = copy.deepcopy(options)
    dmrg.extrapolate_in_variance(psi, M, options, [2, 4])
    assert max(psi.chi) <= 4
    assert options == options_before


SIM_MODEL_PARAMS = dict(L=6, **NNN_PARAMS)

simulation_params = {
    'model_class': 'SpinChainNNN2',
    'model_params': SIM_MODEL_PARAMS,
    'algorithm_class': 'TwoSiteDMRGEngine',
    'algorithm_params': {
        'trunc_params': {'chi_max': 2},
        'mixer': False,
        'N_sweeps_check': 1,
        'min_sweeps': 2,
        'max_sweeps': 3,
        'max_trunc_err': 1.0,
    },
    'initial_state_params': {'method': 'lat_product_state', 'product_state': [['up'], ['down']]},
    'connect_measurements': [
        ('tenpy.simulations.measurement', 'm_two_site_variance'),
        ('tenpy.simulations.measurement', 'm_two_site_variance', {'results_key': 'my_variance'}),
        ('tenpy.simulations.measurement', 'm_n_site_variance'),
        ('tenpy.simulations.measurement', 'm_n_site_variance', {'n_sites': 3}),
    ],
}


def test_variance_extrapolation_simulation():
    from tenpy.simulations.ground_state_search import VarianceExtrapolation

    chi_list = [2, 4, 8]
    sim_params = copy.deepcopy(simulation_params)
    sim_params['extrapolation_chi_list'] = chi_list
    sim = VarianceExtrapolation(sim_params)
    results = sim.run()
    extrapolation = results['variance_extrapolation']
    assert list(np.asarray(extrapolation['chi'])) == chi_list
    energies = np.asarray(extrapolation['E'], dtype=float)
    variances = np.asarray(extrapolation['two_site_variance'], dtype=float)
    assert energies.shape == variances.shape == (len(chi_list),)
    assert np.all(variances >= -TOL)
    assert abs(float(results['energy']) - energies[-1]) < TOL
    matrix = np.vstack([variances, np.ones(len(chi_list))]).T
    assert abs(float(extrapolation['E_extrapolated']) - np.linalg.lstsq(matrix, energies, rcond=None)[0][1]) < TOL
    assert abs(variances[-1] - float(sim.model.H_MPO.two_site_variance(sim.psi))) < TOL
    # the usual measurements are taken after each run; a simulation may also measure around them,
    # so look for the run of values that belongs to the runs
    measured = np.asarray(results['measurements']['two_site_variance'], dtype=float)
    assert measured.size >= len(chi_list)
    starts = [
        i
        for i in range(measured.size - len(chi_list) + 1)
        if np.allclose(measured[i : i + len(chi_list)], variances, atol=TOL)
    ]
    assert starts, (measured, variances)


def test_measurement_function():
    sim_params = copy.deepcopy(simulation_params)
    sim = GroundStateSearch(sim_params)
    results = sim.run()
    meas = results['measurements']
    assert 'two_site_variance' in meas
    assert 'my_variance' in meas
    values = np.asarray(meas['two_site_variance'], dtype=float)
    # m_n_site_variance defaults its key to the block size followed by _site_variance
    assert '2_site_variance' in meas and '3_site_variance' in meas
    assert np.allclose(np.asarray(meas['2_site_variance'], dtype=float), values, atol=TOL)
    three = np.asarray(meas['3_site_variance'], dtype=float)
    assert three.shape == values.shape
    assert abs(three[-1] - float(sim.model.H_MPO.n_site_variance(sim.psi, 3))) < TOL
    assert values.shape == (2,)  # once before and once after the DMRG run
    assert np.allclose(values, np.asarray(meas['my_variance'], dtype=float), atol=TOL)
    # the same initial state outside the simulation; this model has next-nearest neighbor terms,
    # so the two-site variance is strictly below the full variance
    M = SpinChainNNN2(dict(SIM_MODEL_PARAMS))
    psi_initial = MPS.from_lat_product_state(M.lat, [['up'], ['down']])
    initial_two_site = float(M.H_MPO.two_site_variance(psi_initial))
    assert abs(values[0] - initial_two_site) < TOL
    final_two_site = float(sim.model.H_MPO.two_site_variance(sim.psi))
    assert abs(values[-1] - final_two_site) < TOL


def test_infinite_next_nearest_neighbor_blocks():
    """Blocks of three sites only contribute for a Hamiltonian that reaches beyond neighbors, so
    check an infinite next-nearest neighbor chain against the bulk of a long finite chain."""
    L = 3
    params = dict(NNN_PARAMS, bc_MPS='infinite')
    p_state = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([np.cos(0.3), np.sin(0.3)]),
    ]
    M = SpinChainNNN2(dict(L=L, **params))
    psi = MPS.from_product_state(M.lat.mps_sites(), p_state, bc='infinite', unit_cell_width=L)
    density, blocks = M.H_MPO.n_site_variance(psi, 3, return_terms=True)
    terms = [np.asarray(block, dtype=float) for block in blocks]
    assert len(terms) == 3 and all(term.shape == (L,) for term in terms)
    # the same product state on three unit cells of a finite chain, with a dense reference
    M_fin = SpinChainNNN2(dict(L=3 * L, **dict(NNN_PARAMS, bc_MPS='finite')))
    psi_fin = MPS.from_product_state(M_fin.lat.mps_sites(), p_state * 3, unit_cell_width=3 * L)
    H = get_numpy_Hamiltonian(M_fin, undo_sort_charge=False)
    middle = slice(L, 2 * L)
    for n in range(1, 4):
        assert np.allclose(terms[n - 1], dense_block_terms(H, psi_fin, n)[middle], atol=TOL)
    assert abs(float(density) - sum(term.sum() for term in terms) / L) < TOL
    assert terms[2].max() > 1.0e-3  # the three-site reference values are not all zero


def test_extrapolate_compresses_before_the_run(monkeypatch):
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    at_start = []

    class Recording(dmrg.TwoSiteDMRGEngine):
        def run(self):
            at_start.append(max(self.psi.chi))
            return super().run()

    monkeypatch.setattr(dmrg, 'TwoSiteDMRGEngine', Recording)
    chi_list = [8, 2]
    observables = {'max_chi': lambda state, model: float(max(state.chi))}
    options = {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}
    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list, observables)
    after_run = np.asarray(results['max_chi'], dtype=float)
    assert len(at_start) == len(chi_list)
    # the first run grows the state beyond the second bond dimension, which is only visible if the
    # state is brought down to it before that run rather than during it
    assert after_run[0] > chi_list[1]
    assert all(at_start[k] <= chi_list[k] for k in range(len(chi_list)))


def test_measurement_functions_use_the_simulation_state():
    """Under grouped sites the arguments of a measurement function and the state and model held by
    the simulation differ, and only the latter pair belongs together."""
    L = 12
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    options = {'mixer': False, 'min_sweeps': 3, 'max_sweeps': 3, 'max_trunc_err': 1.0, 'trunc_params': {'chi_max': 4}}
    dmrg.TwoSiteDMRGEngine(psi, M, options).run()
    grouped_psi = psi.copy()
    grouped_psi.group_sites(2)
    grouped_model = copy.deepcopy(M)
    grouped_model.group_sites(2)

    class Simulation:  # only the two attributes that the measurement functions need
        pass

    simulation = Simulation()
    simulation.psi = grouped_psi
    simulation.model = grouped_model
    results = {}
    measurement.m_two_site_variance(results, psi, M, simulation)
    measurement.m_n_site_variance(results, psi, M, simulation, n_sites=1)
    grouped_two_site = float(grouped_model.H_MPO.two_site_variance(grouped_psi))
    grouped_one_site = float(grouped_model.H_MPO.n_site_variance(grouped_psi, 1))
    assert abs(float(results['two_site_variance']) - grouped_two_site) < TOL
    assert abs(float(results['1_site_variance']) - grouped_one_site) < TOL
    # the values of the arguments are clearly different ones
    assert abs(float(M.H_MPO.two_site_variance(psi)) - grouped_two_site) > 1.0e-5
    assert abs(float(M.H_MPO.n_site_variance(psi, 1)) - grouped_one_site) > 1.0e-5


def test_measurement_function_with_grouped_sites():
    sim_params = copy.deepcopy(simulation_params)
    sim_params['model_params'] = dict(SIM_MODEL_PARAMS, L=8)
    sim_params['group_sites'] = 2
    sim = GroundStateSearch(sim_params)
    results = sim.run()
    meas = results['measurements']
    values = np.asarray(meas['two_site_variance'], dtype=float)
    assert values.shape == (2,)
    # the first measurement is taken while the simulation holds the grouped state and model
    M = SpinChainNNN2(dict(SIM_MODEL_PARAMS, L=8))
    psi_initial = MPS.from_lat_product_state(M.lat, [['up'], ['down']])
    grouped_psi = psi_initial.copy()
    grouped_psi.group_sites(2)
    grouped_model = copy.deepcopy(M)
    grouped_model.group_sites(2)
    grouped_two_site = float(grouped_model.H_MPO.two_site_variance(grouped_psi))
    assert abs(values[0] - grouped_two_site) < TOL
    assert abs(float(M.H_MPO.two_site_variance(psi_initial)) - grouped_two_site) > 1.0e-4
    assert np.allclose(values, np.asarray(meas['my_variance'], dtype=float), atol=TOL)
    assert np.allclose(np.asarray(meas['2_site_variance'], dtype=float), values, atol=TOL)


def test_extrapolate_records_the_bond_dimension_reached():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    # the second bond dimension is larger than this state can ever have, so the recorded value is
    # the one that the state reached and not the one that was asked for
    chi_list = [2, 64]
    options = {'mixer': False, 'max_sweeps': 4, 'max_trunc_err': 1.0}
    results = dmrg.extrapolate_in_variance(psi, M, options, chi_list)
    reached = np.asarray(results['chi_reached'], dtype=int)
    assert reached.shape == (len(chi_list),)
    assert all(reached[k] <= chi_list[k] for k in range(len(chi_list)))
    assert reached[-1] == max(psi.chi)
    # and the name is taken, so an observable may not use it
    with pytest.raises(ValueError):
        dmrg.extrapolate_in_variance(psi, M, options, chi_list, {'chi_reached': lambda state, model: 0.0})


def test_measurement_terms():
    L = 6
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = random_state_on_sites(M.lat.mps_sites(), 4, seed=11)

    class Simulation:
        pass

    simulation = Simulation()
    simulation.psi = psi
    simulation.model = M
    results = {}
    measurement.m_two_site_variance(results, psi, M, simulation, return_terms=True)
    measurement.m_n_site_variance(results, psi, M, simulation, n_sites=3, return_terms=True)
    total, one_site, two_site = M.H_MPO.two_site_variance(psi, return_terms=True)
    assert abs(float(results['two_site_variance']) - float(total)) < TOL
    assert np.allclose(
        np.asarray(results['two_site_variance_one_site_terms'], dtype=float),
        np.asarray(one_site, dtype=float),
        atol=TOL,
    )
    assert np.allclose(
        np.asarray(results['two_site_variance_two_site_terms'], dtype=float),
        np.asarray(two_site, dtype=float),
        atol=TOL,
    )
    total, terms = M.H_MPO.n_site_variance(psi, 3, return_terms=True)
    assert abs(float(results['3_site_variance']) - float(total)) < TOL
    for n in range(1, 4):
        key = f'3_site_variance_{n:d}_site_terms'
        assert np.allclose(np.asarray(results[key], dtype=float), np.asarray(terms[n - 1], dtype=float), atol=TOL)
    # without the option only the total is stored
    plain = {}
    measurement.m_two_site_variance(plain, psi, M, simulation)
    measurement.m_n_site_variance(plain, psi, M, simulation, n_sites=3)
    assert set(plain) == {'two_site_variance', '3_site_variance'}


def test_variance_extrapolation_simulation_overrides_chi_schedule():
    from tenpy.simulations.ground_state_search import VarianceExtrapolation

    chi_list = [2, 4]
    sim_params = copy.deepcopy(simulation_params)
    sim_params['model_params'] = dict(SIM_MODEL_PARAMS, L=8)
    sim_params['extrapolation_chi_list'] = chi_list
    # a bond dimension schedule in the algorithm options must not override the requested ones
    sim_params['algorithm_params']['chi_list'] = {0: 16}
    sim = VarianceExtrapolation(sim_params)
    results = sim.run()
    extrapolation = results['variance_extrapolation']
    assert list(np.asarray(extrapolation['chi'])) == chi_list
    reached = np.asarray(extrapolation['chi_reached'], dtype=int)
    assert reached.shape == (len(chi_list),)
    assert all(reached[k] <= chi_list[k] for k in range(len(chi_list)))
    assert max(sim.psi.chi) <= chi_list[-1]


def test_extrapolated_values_are_numbers():
    L = 8
    M = SpinChainNNN2(dict(L=L, **NNN_PARAMS))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (L // 2), unit_cell_width=L)
    observables = {'max_chi': lambda state, model: int(max(state.chi))}
    options = {'mixer': False, 'max_sweeps': 3, 'max_trunc_err': 1.0}
    results = dmrg.extrapolate_in_variance(psi, M, options, [2, 4], observables)
    for key in ['E_extrapolated', 'fit_residuum', 'max_chi_extrapolated']:
        assert np.ndim(results[key]) == 0, (key, results[key])
        assert np.isfinite(float(results[key]))
    # the records themselves are one value per run
    for key in ['E', 'two_site_variance', 'chi', 'chi_reached', 'max_chi']:
        assert np.shape(results[key]) == (2,), (key, results[key])
