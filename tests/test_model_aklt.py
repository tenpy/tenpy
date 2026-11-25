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
        psi = M.psi_AKLT()
    else:
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
    assert abs(1.0 - abs(psi0.overlap(M.psi_AKLT(), understood_infinite=True))) < 1.0e-10


@pytest.mark.parametrize('conserve', ['None', 'parity', 'Sz', 'SU2'])
def test_psi_AKLT_finite_needs_two_sites(conserve):
    # This is the intended behaviour: the finite state needs two edge spins, which is not possible for a single site.
    M = aklt.AKLTChain(dict(L=1, bc_MPS='finite', conserve=conserve))
    with pytest.raises(ValueError, match='at least two sites'):
        M.psi_AKLT()


@pytest.mark.parametrize('L', [2, 3, 4, 6])
@pytest.mark.parametrize('conserve', ['None', 'parity', 'Sz', 'SU2'])
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_psi_AKLT(bc_MPS, conserve, L, request):
    if bc_MPS == 'infinite' and conserve == 'parity' and L % 2 == 1:
        # the Sz=0 state has the parity charge 1, so the unit cell has a non-trivial total charge
        request.applymarker(
            pytest.mark.xfail(
                reason='canonical_form_infinite1 does not support a non-trivial total charge yet (#765)',
                raises=NotImplementedError,
                strict=True,
            )
        )
    M = aklt.AKLTChain(dict(L=L, bc_MPS=bc_MPS, conserve=conserve))
    psi_aklt = M.psi_AKLT()
    # the ground state energy is -2/3 per bond, see the AKLT docstring (and Affleck, Kennedy, Lieb, Tasaki 1987).
    # expectation_value of the MPO gives the energy per site for infinite and the total energy for finite MPS
    if bc_MPS == 'infinite':
        assert M.get_H_MPO().expectation_value(psi_aklt) == pytest.approx(-2 / 3.0, abs=1e-10)
        # all bonds carry a singlet of two spin-1/2, i.e. a maximally entangled pair of two-level systems
        assert psi_aklt.entanglement_entropy() == pytest.approx(np.log(2), abs=1e-10)
    else:
        assert M.get_H_MPO().expectation_value(psi_aklt) == pytest.approx(-(L - 1) * 2 / 3.0, abs=1e-10)
        assert psi_aklt.overlap(psi_aklt) == pytest.approx(1.0, abs=1e-10)
        bond_dims = [psi_aklt.get_SL(i).get_leg('vL').dim for i in range(L)]
        if conserve == 'SU2':
            # Edge singlet: the Schmidt rank of a cut is the dimension of (spin-1/2 of the AKLT bond) x (spin-1/2 of the
            # edge singlet) = 4, except if one side of the cut is a single site, which has the dimension 3.
            assert bond_dims == [1] + [3 if min(i, L - i) == 1 else 4 for i in range(1, L)]
            # The state is a total singlet, so the reduced density matrix of a single site is proportional to the
            # identity on the 3-dimensional spin-1 space.
            assert psi_aklt.entanglement_entropy()[0] == pytest.approx(np.log(3), abs=1e-10)
            assert psi_aklt.entanglement_entropy()[-1] == pytest.approx(np.log(3), abs=1e-10)
        else:
            # one of the two virtual states is projected out on each edge, the bonds have the dimension 2 of a spin-1/2
            assert bond_dims == [1] + [2] * (L - 1)


@pytest.mark.parametrize('L', [4, 6])
@pytest.mark.parametrize('conserve', ['None', 'parity', 'Sz', 'SU2'])
def test_psi_AKLT_infinite_vs_exact_diagonalization(conserve, L):
    # The AKLT state on a ring of L spin-1 sites is the unique ground state with energy -2L/3. We obtain it
    # by contracting the (dense) tensors of the infinite MPS around the ring and compare to exact diagonalization.
    # The Hamiltonian is built by hand from `aklt_coupling`, since `get_numpy_Hamiltonian` is not ported yet.
    M = aklt.AKLTChain(dict(L=2, bc_MPS='infinite', conserve=conserve))
    site = M.lattice.unit_cell[0]
    psi = M.psi_AKLT()
    psi_ring = np.einsum('apa->p', _contract_mps_tensors(psi, L))
    psi_ring /= np.linalg.norm(psi_ring)

    H = _dense_aklt_hamiltonian(site, L, periodic=True)
    energies, eigenvectors = np.linalg.eigh(H)

    assert energies[0] == pytest.approx(-2 * L / 3)
    assert energies[1] > energies[0] + 0.1  # unique ground state
    assert abs(np.vdot(eigenvectors[:, 0], psi_ring)) == pytest.approx(1.0)


@pytest.mark.parametrize('L', [4, 6])
@pytest.mark.parametrize('conserve', ['None', 'parity', 'Sz', 'SU2'])
def test_psi_AKLT_finite_vs_exact_diagonalization(conserve, L):
    # On an open chain, the ground space is 4-fold degenerate with energy -(L-1)*2/3: the spin-1/2 at each end is free.
    # We contract the MPS to a dense vector and check that it is in the ground space and that it is the state
    # described in the docstring of `psi_AKLT`, which we obtain from an explicit numpy construction of the tensor
    #     B = sqrt(4/3) * (singlet on the bond) x (projector spin-1/2 x spin-1/2 -> spin-1)
    # in the basis Sz = -1, 0, 1 and for the virtual states Sz = -1/2, +1/2.
    M = aklt.AKLTChain(dict(L=L, bc_MPS='finite', conserve=conserve))
    site = M.lattice.unit_cell[0]
    psi = M.psi_AKLT()
    psi_dense = np.einsum('apb->p', _contract_mps_tensors(psi, L))  # the outer bonds are trivial
    assert np.linalg.norm(psi_dense) == pytest.approx(1.0)

    H = _dense_aklt_hamiltonian(site, L, periodic=False)
    energies, eigenvectors = np.linalg.eigh(H)
    assert energies[0] == pytest.approx(-(L - 1) * 2 / 3)
    assert energies[3] == pytest.approx(energies[0])  # 4-fold degenerate
    assert energies[4] > energies[3] + 0.1  # gap
    assert np.linalg.norm(eigenvectors[:, :4].T @ psi_dense) == pytest.approx(1.0)  # in the ground space

    ref = _reference_mps_vector(L)  # legs vL, p, vR
    if conserve == 'SU2':
        # The singlet on the bond between two sites is `sL @ sR`, which is part of the tensors. Thus, closing the
        # chain with the identity gives the singlet of the two edge spins: the state on a ring, without the bond term.
        ref = np.einsum('apa->p', ref)
    else:
        # left edge spin up, right edge spin down: project the first vL and the last vR to the first virtual state
        ref = ref[0, :, 0]
    ref = ref.reshape(-1) / np.linalg.norm(ref)
    assert abs(np.vdot(ref, psi_dense)) == pytest.approx(1.0)


@pytest.mark.parametrize('L', [2, 3, 4, 5, 6])
@pytest.mark.parametrize('bc_MPS', ['finite', 'infinite'])
def test_psi_AKLT_SU2_is_ring_state(bc_MPS, L):
    # The edge spins of the finite SU(2) state are in the singlet of the bond which closes the chain, and the open
    # chain Hamiltonian has no term on this bond. Hence, the finite state is the AKLT state on a periodic ring of L
    # sites, which is also the infinite state with L sites per unit cell, wrapped around the ring.
    psi = aklt.AKLTChain(dict(L=L, bc_MPS=bc_MPS, conserve='SU2')).psi_AKLT()
    state = np.einsum('apa->p', _contract_mps_tensors(psi, L))
    # reference: the numpy tensor with the singlets on the bonds, traced around the ring
    ref = np.einsum('apa->p', _reference_mps_vector(L))
    assert abs(np.vdot(ref / np.linalg.norm(ref), state / np.linalg.norm(state))) == pytest.approx(1.0)


@pytest.mark.parametrize('L', [2, 3, 4, 5, 6])
@pytest.mark.parametrize('component', ['x', 'y', 'z'])
def test_psi_AKLT_SU2_finite_is_singlet(component, L):
    # A singlet is annihilated by each component of the total spin, i.e. <S_c^2> = 0 for c = x, y, z.
    psi = aklt.AKLTChain(dict(L=L, bc_MPS='finite', conserve='SU2')).psi_AKLT()
    psi_dense = np.einsum('apb->p', _contract_mps_tensors(psi, L))  # the outer bonds are trivial
    # spin-1 matrices in the basis Sz = -1, 0, 1; S+ raises Sz, which is the row index
    Sp = np.sqrt(2) * np.diag([1.0, 1.0], k=-1)
    op = {'x': (Sp + Sp.T) / 2, 'y': (Sp - Sp.T) / 2j, 'z': np.diag([-1.0, 0.0, 1.0])}[component]
    S_total = sum(np.kron(np.kron(np.eye(3**i), op), np.eye(3 ** (L - i - 1))) for i in range(L))
    assert np.linalg.norm(S_total @ psi_dense) == pytest.approx(0.0, abs=1e-10)


def _reference_B_tensor():
    # The right canonical tensor of the AKLT state with legs vL, p, vR for the basis Sz = -1, 0, 1 of the spin-1 and
    # the virtual states Sz = -1/2, +1/2: sqrt(4/3) * (singlet on the bond) x (projector spin-1/2 x spin-1/2 -> spin-1)
    sL = np.sqrt(0.5) * np.array([[1.0, 0.0], [0.0, -1.0]])
    sR = np.array([[0.0, 1.0], [1.0, 0.0]])
    proj = np.zeros([3, 2, 2])
    proj[0, 0, 0] = 1.0  # |m=-1> = |down down>
    proj[1, 0, 1] = proj[1, 1, 0] = np.sqrt(0.5)  # |m=0> = (|up down> + |down up>) / sqrt(2)
    proj[2, 1, 1] = 1.0  # |m=+1> = |up up>
    return np.einsum('ab,pbc,cd->apd', sR, proj, sL) * np.sqrt(4 / 3)


def _reference_mps_vector(L):
    # Contract L copies of the reference tensor to an array with legs vL, p (all sites combined), vR, where the left
    # leg is the identity, such that tracing vL with vR closes a periodic ring.
    B_ref = _reference_B_tensor()
    ref = np.eye(2).reshape(2, 1, 2)
    for _ in range(L):
        ref = np.einsum('apb,bqc->apqc', ref, B_ref).reshape(2, -1, 2)
    return ref


def _contract_mps_tensors(psi, L):
    # Contract the first L tensors of the MPS to a dense array with legs vL, p (all sites combined), vR. The left leg is
    # the identity, such that tracing vL with vR closes a periodic ring.
    tensors = [
        lasy.planar_permute_legs(psi.get_B(i, 'B'), codomain=['vL', 'p'], domain=['vR']).to_numpy(
            understood_braiding=True
        )
        for i in range(L)
    ]  # legs vL, p, vR
    result = np.eye(tensors[0].shape[0]).reshape(tensors[0].shape[0], 1, -1)
    for B in tensors:
        result = np.einsum('apb,bqc->apqc', result, B).reshape(result.shape[0], -1, B.shape[2])
    return result


def _dense_aklt_hamiltonian(site, L, periodic):
    # The AKLT Hamiltonian on L sites as a dense matrix in the basis Sz = -1, 0, 1 of each site, built from the
    # two-site coupling `aklt_coupling`, with or without the term on the bond between the last and the first site.
    h_two_site = lasy.models.aklt_coupling([site, site], name='AKLT').to_tensor()
    h_two_site = lasy.planar_permute_legs(h_two_site, codomain=['p0', 'p1'], domain=['p0*', 'p1*'])
    # numpy leg order is codomain legs, then domain legs in reversed order: p0, p1, p1*, p0*
    h_two_site = h_two_site.to_numpy(understood_braiding=True).transpose([0, 1, 3, 2]).reshape([9, 9])
    H = np.zeros([3**L, 3**L], dtype=complex)
    for k, basis_vector in enumerate(np.eye(3**L)):
        state = basis_vector.reshape([3] * L)
        for i in range(L if periodic else L - 1):
            bond = [i, (i + 1) % L]
            two_site = np.moveaxis(state, bond, [0, 1]).reshape(9, -1)
            H[:, k] += np.moveaxis((h_two_site @ two_site).reshape([3, 3] + [3] * (L - 2)), [0, 1], bond).reshape(-1)
    return H
