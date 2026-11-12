# Copyright (C) TeNPy Developers, Apache license
import lasyten as lasy
import numpy as np
import pytest
from test_model import check_general_model

from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.models import aklt
from tenpy.networks.mps import MPS


@pytest.mark.parametrize('conserve', ['best', 'Sz'])
def test_AKLT_finite(conserve):
    L = 4
    check_general_model(aklt.AKLTChain, {'L': L, 'conserve': conserve}, {})
    M = aklt.AKLTChain({'L': L, 'bc_MPS': 'finite', 'conserve': conserve})
    if conserve == 'best':
        pytest.skip('Need to build an SU(2) symmetric initial state...')
    psi = MPS.from_lat_product_state(M.lattice, [['up'], ['down']])
    eng = TwoSiteDMRGEngine(psi, M, {'trunc_params': {'svd_min': 1.0e-10, 'chi_max': 10}})
    E0, psi0 = eng.run()
    assert abs(E0 - (-2 / 3.0) * (L - 1)) < 1.0e-10
    # note: if we view the system as spin-1/2 projected to spin-1, the first and last spin
    # are arbitrary.
    # if they are in a superposition, this will contribute at most another factor of 2 to chi.
    assert all([chi <= 4 for chi in psi0.chi])


@pytest.mark.slow
@pytest.mark.parametrize('conserve', ['best', 'Sz'])
def test_AKLT_infinite(conserve):
    # switch to infinite
    L = 4
    M = aklt.AKLTChain({'L': L, 'bc_MPS': 'infinite', 'conserve': conserve})
    if conserve == 'best':
        psi = M.psi_AKLT()
    else:
        psi = MPS.from_lat_product_state(M.lattice, [['up'], ['down']])
    eng = TwoSiteDMRGEngine(
        psi,
        M,
        {
            'trunc_params': {'svd_min': 1.0e-10, 'chi_max': 10},
            'N_sweeps_check': 1,
            'mixer': False,
            'min_sweeps': 50,  # TODO workaround
        },
        resume_data={'init_env_data': dict(start_env_sites=0)},
    )
    E0, psi0 = eng.run()
    assert abs(E0 - (-2 / 3.0)) < 1.0e-10
    if conserve == 'best':
        assert abs(1.0 - abs(psi0.overlap(M.psi_AKLT(), understood_infinite=True))) < 1.0e-10
    # TODO: compare to M.psi_AKLT() for conserve='Sz' once it is ported


@pytest.mark.parametrize(
    'bc_MPS, conserve',
    [
        ('infinite', 'SU2'),
        *[
            pytest.param(
                bc_MPS,
                conserve,
                marks=pytest.mark.xfail(
                    reason='psi_AKLT is only ported for conserve="SU2" and bc_MPS="infinite" (#764)',
                    raises=NotImplementedError,
                    strict=True,
                ),
            )
            for conserve in ['None', 'parity', 'Sz', 'SU2']
            for bc_MPS in ['finite', 'infinite']
            if (bc_MPS, conserve) != ('infinite', 'SU2')
        ],
    ],
)
def test_psi_AKLT(bc_MPS, conserve):
    L = 4
    M = aklt.AKLTChain(dict(L=L, bc_MPS=bc_MPS, conserve=conserve))
    psi_aklt = M.psi_AKLT()
    assert M.get_H_MPO().expectation_value(psi_aklt) == pytest.approx(-2 / 3.0, abs=1e-10)  # energy per site
    assert psi_aklt.entanglement_entropy() == pytest.approx(np.log(2), abs=1e-10)


@pytest.mark.parametrize('L', [4, 6])
def test_psi_AKLT_SU2_vs_exact_diagonalization(L):
    # The AKLT state on a ring of L spin-1 sites is the unique ground state with energy -2L/3. We obtain it
    # by contracting the (dense) tensors of the infinite MPS around the ring and compare to exact diagonalization.
    # The Hamiltonian is built by hand from `aklt_coupling`, since `get_numpy_Hamiltonian` is not ported yet.
    M = aklt.AKLTChain(dict(L=2, bc_MPS='infinite', conserve='SU2'))
    site = M.lattice.unit_cell[0]
    psi = M.psi_AKLT()
    Bs = [
        lasy.planar_permute_legs(psi.get_B(i, 'B'), codomain=['vL', 'p'], domain=['vR']).to_numpy(
            understood_braiding=True
        )
        for i in range(2)
    ]  # legs vL, p, vR
    ring = np.eye(Bs[0].shape[0]).reshape(Bs[0].shape[0], 1, -1)  # legs vL, p, vR; trace over vL and vR at the end
    for i in range(L):
        ring = np.einsum('apb,bqc->apqc', ring, Bs[i % 2]).reshape(ring.shape[0], -1, Bs[0].shape[2])
    psi_ring = np.einsum('apa->p', ring)
    psi_ring /= np.linalg.norm(psi_ring)

    h_two_site = lasy.models.aklt_coupling([site, site], name='AKLT').to_tensor()
    h_two_site = lasy.planar_permute_legs(h_two_site, codomain=['p0', 'p1'], domain=['p0*', 'p1*'])
    # numpy leg order is codomain legs, then domain legs in reversed order: p0, p1, p1*, p0*
    h_two_site = h_two_site.to_numpy(understood_braiding=True).transpose([0, 1, 3, 2]).reshape([9, 9])
    H = np.zeros([3**L, 3**L], dtype=complex)
    for k, basis_vector in enumerate(np.eye(3**L)):
        state = basis_vector.reshape([3] * L)
        for i in range(L):
            bond = [i, (i + 1) % L]
            two_site = np.moveaxis(state, bond, [0, 1]).reshape(9, -1)
            H[:, k] += np.moveaxis((h_two_site @ two_site).reshape([3, 3] + [3] * (L - 2)), [0, 1], bond).reshape(-1)
    energies, eigenvectors = np.linalg.eigh(H)

    assert energies[0] == pytest.approx(-2 * L / 3)
    assert energies[1] > energies[0] + 0.1  # unique ground state
    assert abs(np.vdot(eigenvectors[:, 0], psi_ring)) == pytest.approx(1.0)
