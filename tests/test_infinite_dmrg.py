"""Tests for infinite two-site DMRG with uniform spin chains.

All tests use plain two-site DMRG without mixer.
For the iDMRG with SU(2) symmetry, we only test cases where the bond stays small (AKLT), since the
fusion tree backend is currently very slow once several spin sectors are present on the bond.

The Sz iDMRG tests are marked slow, since they take many sweeps (with the Debug build of lasyten used in the CI).
"""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import numpy.testing as npt
import pytest

from tenpy.algorithms import dmrg
from tenpy.models.aklt import AKLTChain
from tenpy.models.spins import SpinChain
from tenpy.networks import mps

E_HEISENBERG_S12 = 0.25 - np.log(2)  # exact energy per site of the spin-1/2 Heisenberg chain
E_HEISENBERG_S1 = -1.401484038971  # energy per site of the spin-1 Heisenberg chain, White & Huse (1993)


def _run_idmrg(M, psi, chi_max, max_sweeps=30, N_sweeps_check=4):
    dmrg_pars = {
        'combine': False,
        'mixer': False,
        'trunc_params': {'svd_min': 1.0e-10, 'chi_max': chi_max},
        'max_E_err': 1.0e-10,
        'max_S_err': 1.0e-6,
        'N_sweeps_check': N_sweeps_check,
        'max_sweeps': max_sweeps,
        'active_sites': 2,
        'start_env': 1,
    }
    return dmrg.run(psi, M, dmrg_pars, resume_data={'init_env_data': dict(start_env_sites=0)})


# slow: many iDMRG sweeps (especially with a Debug build of lasyten). Independent of the SU(N) symbol caching.
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore:Re-using environment with `chi_list` set! Do you want this:UserWarning')
@pytest.mark.parametrize(
    'S, state, E_exact, tol, chi_max',
    [
        (0.5, [0, 1], E_HEISENBERG_S12, 5e-4, 16),
        (1, [0, 2], E_HEISENBERG_S1, 5e-4, 12),
    ],
)
def test_heisenberg_Sz(S, state, E_exact, tol, chi_max):
    M = SpinChain(dict(L=2, S=S, conserve='Sz', bc_MPS='infinite'))
    psi = mps.MPS.from_product_state(M.lattice.mps_sites(), state, bc='infinite', unit_cell_width=2)
    res = _run_idmrg(M, psi, chi_max, max_sweeps=12)
    E = res['E']
    assert E_exact - 1e-8 <= E <= E_exact + tol  # variational (up to the truncation error), and close
    assert max(psi.chi) <= chi_max
    assert np.max(np.abs(psi.norm_test())) < 1e-10
    npt.assert_allclose(M.get_H_MPO().expectation_value(psi), E, atol=1e-6)
    # correlation length: the (triplet) sector dominates; finite chi gives a finite value
    xi = psi.correlation_length2(charge_sector=None)
    assert 1.0 < xi < 100.0
    if S == 1:
        # the Haldane chain has a correlation length of ~6.03 in the limit of large chi, we get a smaller value
        assert 3.0 < xi < 6.5
        # entanglement entropy of the Haldane chain is ~ 1 for large chi
        assert 0.6 < np.mean(psi.entanglement_entropy()) < 1.1


# slow: many iDMRG sweeps (especially with a Debug build of lasyten). Independent of the SU(N) symbol caching.
@pytest.mark.slow
def test_heisenberg_Sz_matches_tenpy_v1():
    # Reference values were obtained with tenpy v1 (numpy_conserved, ``SpinChain`` with ``bc_MPS='infinite'``)
    # for the identical algorithm and parameters: Neel start state, two-site DMRG without mixer, chi_max=8.
    M = SpinChain(dict(L=2, S=1, conserve='Sz', bc_MPS='infinite'))
    psi = mps.MPS.from_product_state(M.lattice.mps_sites(), [0, 2], bc='infinite', unit_cell_width=2)
    res = _run_idmrg(M, psi, chi_max=8, max_sweeps=60)
    npt.assert_allclose(res['E'], -1.4010933106673065, atol=1e-9)
    npt.assert_allclose(psi.entanglement_entropy(), 0.8416008768252741, atol=1e-8)
    npt.assert_allclose(psi.correlation_length2(charge_sector=0), 3.33846832350632, atol=1e-7)
    assert [int(S.get_leg('vL').dim) for S in psi._S] == [8, 8]


def test_aklt_idmrg_su2():
    # the AKLT state is the exact ground state, with a single spin-1/2 multiplet on each bond
    M = AKLTChain(dict(L=2, bc_MPS='infinite', conserve='SU2'))
    psi = M.psi_AKLT()
    E_exact = -2.0 / 3.0
    npt.assert_allclose(M.get_H_MPO().expectation_value(psi), E_exact, atol=1e-12)  # already the exact state
    # the state is exact: a few sweeps suffice
    res = _run_idmrg(M, psi, chi_max=8, max_sweeps=4, N_sweeps_check=2)
    npt.assert_allclose(res['E'], E_exact, atol=1e-10)
    assert [int(S.get_leg('vL').dim) for S in psi._S] == [2, 2]  # truncation removed everything but the spin-1/2
    npt.assert_allclose(psi.entanglement_entropy(), np.log(2), atol=1e-8)
    npt.assert_allclose(psi.correlation_length2(charge_sector=None), 1 / np.log(3), atol=1e-8)


def test_heisenberg_on_aklt_state_su2():
    # <S_i . S_{i+1}> = -4/3 in the AKLT state: tests the SU(2) MPO of the Heisenberg chain and the canonical form
    M = SpinChain(dict(L=2, S=1, conserve='SU2', bc_MPS='infinite'))
    psi = AKLTChain(dict(L=2, bc_MPS='infinite', conserve='SU2')).psi_AKLT()
    npt.assert_allclose(M.get_H_MPO().expectation_value(psi), -4.0 / 3.0, atol=1e-12)
