"""A collection of tests for (classes in) :mod:`tenpy.algorithms.mps_common`."""

# Copyright (C) TeNPy Developers, Apache license

import lasyten as lasy
import numpy as np
import numpy.testing as npt
import pytest
from lasyten.models.sites import SpinSite

from tenpy.algorithms import dmrg
from tenpy.algorithms.mps_common import OneSiteH, TwoSiteH, VariationalCompression, ZeroSiteH
from tenpy.models import tf_ising
from tenpy.models.fermions_spinless import FermionChain
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.mps import MPS, MPSEnvironment
from tenpy.tools import TruncationError


def _compress_random_mps(L, chi_small, seed):
    """Build a random dense state, embed it as an MPS at full bond dimension, and compress it.

    Returns `(block, psi_large, psi_small, p_labels, d)`, where `block` is the original dense
    array, `psi_large` the uncompressed MPS, and `psi_small` the :class:`VariationalCompression`
    result (bond dimension capped at `chi_small`).
    """
    rng = np.random.default_rng(seed)
    site = SpinSite(S=0.5, conserve=None)
    sites = [site] * L
    d = site.dim
    backend = lasy.get_backend('no_symmetry', 'numpy')

    block = rng.normal(size=(d,) * L) + 1.0j * rng.normal(size=(d,) * L)
    block /= np.linalg.norm(block)

    p_labels = [f'p{i}' for i in range(L)]
    psi_tensor = lasy.SymmetricTensor.from_dense_block(block, codomain=[site.leg] * L, backend=backend, labels=p_labels)
    psi_large = MPS.from_full(sites, psi_tensor, form=None, unit_cell_width=L)

    options = {
        'trunc_params': {'chi_max': chi_small},
        'min_sweeps': 5,
        'max_sweeps': 5,
        'tol_theta_diff': None,
        'max_trunc_err': 1.0,  # we *want* a large, known truncation error here
    }
    compressor = VariationalCompression(psi_large.copy(), options)
    trunc_err = compressor.run()
    psi_small = compressor.psi
    return block, psi_large, psi_small, p_labels, d, trunc_err


def test_variational_compression():
    """Compress a random finite MPS to a smaller bond dimension.

    With no Hamiltonian involved, the exact optimum for compressing a state to a fixed bond
    dimension is given by an SVD truncation of the corresponding bipartition of the dense state.
    We pick a system size such that only a single bond needs truncating, so that this SVD
    truncation (done here by hand, directly on the dense array) is *the* global optimum that
    :class:`VariationalCompression` should reproduce.
    """
    L = 4
    chi_small = 2
    block, psi_large, psi_small, p_labels, d, trunc_err = _compress_random_mps(L, chi_small, seed=0)
    assert max(psi_large.chi) == d**2  # bond (1, 2) is at its maximal, untruncated bond dimension
    assert max(psi_small.chi) == chi_small

    # hard-coded reference: numpy SVD of the dense array at the only bond that needs truncating
    mat = block.reshape(d**2, d**2)
    S_ideal = np.linalg.svd(mat, compute_uv=False)
    err_ideal = TruncationError.from_S(S_ideal[chi_small:], norm_old=np.linalg.norm(S_ideal))

    npt.assert_allclose(trunc_err.eps, err_ideal.eps, atol=1e-10)

    # contract the compressed MPS back to a dense array and check its fidelity loss directly
    theta = psi_small.get_theta(0, n=L)
    got = theta.to_numpy(['vL'] + p_labels + ['vR']).reshape((d,) * L)
    overlap = np.vdot(got.ravel(), block.ravel())
    fidelity_loss = 1.0 - abs(overlap) ** 2 / (np.linalg.norm(got) ** 2 * np.linalg.norm(block) ** 2)
    npt.assert_allclose(fidelity_loss, err_ideal.eps, atol=1e-10)


def test_variational_compression_overlap():
    """Check ``MPSEnvironment.full_contraction`` against a plain numpy overlap.

    ``test_variational_compression`` only ever compares dense arrays (via a manual ``to_numpy``
    contraction), which never exercises the environment's own overlap machinery
    (``BaseEnvironment._full_contraction_LP_RP``/``MPSEnvironment._contract_LP``/``_contract_RP``).
    This test drives that code path directly, on the same large/compressed-small MPS pair.
    """
    L = 4
    chi_small = 2
    block, psi_large, psi_small, p_labels, d, _ = _compress_random_mps(L, chi_small, seed=1)

    got = MPSEnvironment(psi_small, psi_large).full_contraction(0).as_complex128()
    # full_contraction includes the factor bra.norm * ket.norm. `psi_large.norm` is 1, but the
    # compression leaves `psi_small.norm = sqrt(1 - eps) < 1`, the norm of the kept (truncated)
    # part of the state, where eps is the discarded weight. `get_theta` is normalized to 1, so
    # divide `psi_small.norm` out to compare with the normalized dense state below.
    got = got / psi_small.norm

    theta = psi_small.get_theta(0, n=L)
    got_dense = theta.to_numpy(['vL'] + p_labels + ['vR']).reshape((d,) * L)
    # `get_theta` and `block` both have unit norm (as dense arrays), so this is the overlap of the normalized states
    expected = np.vdot(got_dense.ravel(), block.ravel())
    npt.assert_allclose(got, expected, atol=1e-10)


def test_two_site_h_matvec(J=1.5, g=0.9):
    """Check `TwoSiteH.matvec` against a hand-coded dense Hamiltonian, on a 2-site Ising model.

    Nothing has ever exercised the ``PlanarDiagram``/``matvec`` machinery `OneSiteH`/`TwoSiteH`
    got ported to (``adapt_effective_h``) -- `VariationalCompression` uses `DummyTwoSiteH`, whose
    `combine_theta` is a no-op and never calls `matvec`. This builds a real (tiny) Hamiltonian MPO
    and checks the actual LP-W0-W1-RP contraction.
    """
    M = tf_ising.TFIChain(dict(L=2, bc_MPS='finite', conserve=None, J=J, g=g))
    H_MPO = M.get_H_MPO()
    site = M.lattice.mps_sites()[0]
    rng = np.random.default_rng(2)
    d = 2
    block = rng.normal(size=(d, d)) + 1.0j * rng.normal(size=(d, d))
    block /= np.linalg.norm(block)
    backend = lasy.get_backend('no_symmetry', 'numpy')
    psi_tensor = lasy.SymmetricTensor.from_dense_block(
        block, codomain=[site.leg] * 2, backend=backend, labels=['p0', 'p1']
    )
    psi = MPS.from_full([site, site], psi_tensor, form=None, unit_cell_width=2)

    env = MPOEnvironment(psi, H_MPO, psi)
    eff_H = TwoSiteH(env, i0=0)
    theta = psi.get_theta(0, n=2)
    got = eff_H.matvec(theta)
    got_dense = got.to_numpy(['vL', 'p0', 'p1', 'vR']).reshape(d, d)

    # hard-coded reference Hamiltonian, built purely with numpy (same spin-1/2 basis convention
    # as SpinSite(S=0.5, conserve=None): state 0 = down, state 1 = up)
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[-1, 0], [0, 1]])
    Id = np.eye(2)
    H_dense = -J * np.kron(Sx, Sx) - g * (np.kron(Sz, Id) + np.kron(Id, Sz))
    want_dense = (H_dense @ block.reshape(d * d)).reshape(d, d)

    npt.assert_allclose(got_dense, want_dense, atol=1e-10)

    # bonus: to_matrix() (also unexercised by any test so far) should be Hermitian, since H is
    to_mat = eff_H.to_matrix()
    dense_op = to_mat.to_numpy(to_mat.codomain_labels + to_mat.domain_labels)
    n = len(to_mat.codomain_labels)
    dense_op = dense_op.reshape(np.prod(dense_op.shape[:n]), -1)
    npt.assert_allclose(dense_op, dense_op.conj().T, atol=1e-10)


@pytest.mark.parametrize(
    'conserve, state',
    [
        ('parity', ['empty', 'full', 'empty', 'full', 'empty', 'empty']),
        ('parity', ['empty', 'full'] * 3),
        ('N', ['empty', 'full', 'empty', 'full', 'empty', 'empty']),
        ('N', ['empty', 'full'] * 3),
    ],
    ids=['parity_even', 'parity_odd', 'N_even', 'N_odd'],
)
def test_effective_h_fermionic_matvec(conserve, state):
    """Verify TwoSiteH, OneSiteH, and ZeroSiteH planar matvec match to_tensor on fermions."""
    L = len(state)
    M = FermionChain(dict(L=L, J=1.0, V=0.7, mu=0.2, bc_MPS='finite', bc_x='open', conserve=conserve))
    psi = MPS.from_product_state(M.lattice.mps_sites(), state, bc='finite', unit_cell_width=L)
    dmrg.run(
        psi,
        M,
        dict(combine=False, mixer=False, diag_method='lanczos', max_sweeps=3, trunc_params=dict(chi_max=8)),
    )
    env = MPOEnvironment(psi, M.get_H_MPO(), psi)

    for i0 in [0, 2, L - 2]:
        H2 = TwoSiteH(env, i0, combine=False)
        th2 = H2.combine_theta(psi.get_theta(i0, n=2))
        ref2 = lasy.compose(H2.to_tensor(), th2)
        assert lasy.almost_equal(H2.matvec(th2), ref2, rtol=1e-10, atol=1e-10)

    for i0 in [0, 2, L - 1]:
        H1 = OneSiteH(env, i0, combine=False)
        H1_from = OneSiteH.from_LP_W0_RP(env.get_LP(i0), env.H.get_W(i0), env.get_RP(i0), i0=i0)
        th1 = H1.combine_theta(psi.get_theta(i0, n=1))
        ref1 = lasy.compose(H1.to_tensor(), th1)
        assert lasy.almost_equal(H1.matvec(th1), ref1, rtol=1e-10, atol=1e-10)
        assert lasy.almost_equal(H1_from.matvec(th1), ref1, rtol=1e-10, atol=1e-10)

    for i0 in [1, 3, L - 1]:
        H0 = ZeroSiteH(env, i0)
        H0_from = ZeroSiteH.from_LP_RP(env.get_LP(i0), env.get_RP(i0 - 1), i0=i0)
        th0 = lasy.planar_permute_legs(psi.get_SL(i0).as_SymmetricTensor(), codomain=H0.acts_on, domain=[])
        ref0 = lasy.compose(H0.to_tensor(), th0)
        assert lasy.almost_equal(H0.matvec(th0), ref0, rtol=1e-10, atol=1e-10)
        assert lasy.almost_equal(H0_from.matvec(th0), ref0, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    'conserve, state, tol_theta_diff',
    [
        ('parity', ['empty', 'full', 'empty', 'full', 'empty', 'empty'], None),
        ('parity', ['empty', 'full'] * 3, 1e-8),
        ('N', ['empty', 'full', 'empty', 'full', 'empty', 'empty'], 1e-8),
        ('N', ['empty', 'full'] * 3, None),
    ],
    ids=['parity_even', 'parity_odd_tol', 'N_even_tol', 'N_odd'],
)
def test_variational_compression_fermionic(conserve, state, tol_theta_diff):
    """Check VariationalCompression on entangled fermionic MPS (including HiddenLegTensor)."""
    L = len(state)
    M = FermionChain(dict(L=L, J=1.0, V=0.5, mu=0.0, bc_MPS='finite', bc_x='open', conserve=conserve))
    psi_full = MPS.from_product_state(M.lattice.mps_sites(), state, bc='finite', unit_cell_width=L)
    dmrg.run(
        psi_full,
        M,
        dict(combine=False, mixer=False, diag_method='lanczos', max_sweeps=4, trunc_params=dict(chi_max=8)),
    )
    options = {
        'trunc_params': {'chi_max': 4},
        'min_sweeps': 3,
        'max_sweeps': 6,
        'tol_theta_diff': tol_theta_diff,
        'max_trunc_err': 1.0,
    }
    compressor = VariationalCompression(psi_full.copy(), options)
    compressor.run()
    psi_comp = compressor.psi
    psi_comp.test_sanity()
    assert max(psi_comp.chi) <= 4
    ov = abs(MPSEnvironment(psi_comp, psi_full).full_contraction(0).as_complex128()) / psi_comp.norm
    assert ov > 0.98
    E_full = float(np.real(M.get_H_MPO().expectation_value(psi_full)))
    E_comp = float(np.real(M.get_H_MPO().expectation_value(psi_comp)))
    assert E_comp >= E_full - 1e-10


@pytest.mark.parametrize('width', [0, 1, 2], ids=['ZeroSiteH', 'OneSiteH', 'TwoSiteH'])
def test_effective_h_interleaved_hidden_legs(width):
    """Verify EffectiveH.matvec preserves sign and layout for interleaved HiddenLegTensor legs."""
    # Only site 1 has -n_1, occupied in the left environment -> local action at i0=2 is exactly -I.
    M = FermionChain(
        dict(L=6, J=0.0, V=0.0, mu=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), conserve='parity', bc_MPS='finite')
    )
    psi = MPS.from_product_state(M.lattice.mps_sites(), ['empty', 'full'] * 3, bc='finite', unit_cell_width=6)
    env = MPOEnvironment(psi, M.get_H_MPO(), psi)
    if width == 0:
        H = ZeroSiteH(env, 2)
    elif width == 1:
        H = OneSiteH(env, 2, combine=False)
    else:
        H = TwoSiteH(env, 2, combine=False)

    op = H.to_tensor()
    pub = list(H.acts_on)
    spaces = dict(zip(pub, op.domain.factors))
    odd = lasy.ElementarySpace.from_defining_sectors(op.symmetry, [[1]])
    spaces.update(a=odd, b=odd.dual, c=odd)

    layouts = {
        'domain': (pub, ['a', 'b']),
        'edge-mixed': (['a', *pub], ['b']),
        'separated': ([pub[0], 'a', *pub[1:], 'b'], []),
        'interior-mixed': ([pub[0], 'a', *pub[1:]], ['b']),
        'interior-pair': ([pub[0], 'a', 'b', *pub[1:]], []),
    }
    if width > 0:
        layouts.update(
            {
                'single-domain': (pub, ['a']),
                'single-interior': ([pub[0], 'a', *pub[1:]], []),
                'triple-interior-mixed': ([pub[0], 'a', 'b', *pub[1:]], ['c']),
            }
        )

    for co, dom in layouts.values():
        rng = np.random.default_rng(196)
        raw = lasy.SymmetricTensor.from_block_func(
            lambda shape: rng.normal(size=shape),
            [spaces[l] for l in co],
            [spaces[l] for l in dom],
            backend=op.backend,
            labels=[co, dom],
            dtype=lasy.float64,
        )
        theta = lasy.HiddenLegTensor(raw, [l for l in co + dom if l in ('a', 'b', 'c')])
        theta.test_sanity()
        out = H.matvec(theta)
        ref = -1.0 * theta
        assert out.codomain_labels == theta.codomain_labels
        assert out.domain_labels == theta.domain_labels
        assert float(lasy.norm(out - ref)) < 1e-12
        ov = float(np.real(lasy.inner(ref, out).as_complex128())) / (float(lasy.norm(ref)) ** 2)
        assert abs(ov - 1.0) < 1e-12
        if not any(l.startswith('!') for l in theta.codomain_labels):
            assert lasy.almost_equal(out, lasy.compose(op, theta), rtol=1e-12, atol=1e-12)
