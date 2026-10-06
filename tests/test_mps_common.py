"""A collection of tests for (classes in) :mod:`tenpy.algorithms.mps_common`."""

# Copyright (C) TeNPy Developers, Apache license
import cyten as ct
import numpy as np
import numpy.testing as npt
from cyten.models.couplings import spin_field_coupling, spin_spin_coupling
from cyten.models.sites import SpinSite

from tenpy.algorithms.mps_common import TwoSiteH, VariationalCompression
from tenpy.models import lattice, new_model
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
    backend = ct.get_backend('no_symmetry', 'numpy')

    block = rng.normal(size=(d,) * L) + 1.0j * rng.normal(size=(d,) * L)
    block /= np.linalg.norm(block)

    p_labels = [f'p{i}' for i in range(L)]
    psi_tensor = ct.SymmetricTensor.from_dense_block(block, codomain=[site.leg] * L, backend=backend, labels=p_labels)
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
    got = got / psi_small.norm  # TODO why is this needed??

    theta = psi_small.get_theta(0, n=L)
    got_dense = theta.to_numpy(['vL'] + p_labels + ['vR']).reshape((d,) * L)
    # both states have norm 1 here (VariationalCompression/MPS.from_full keep them normalized),
    # so the raw dense overlap already matches full_contraction's bra.norm * ket.norm convention
    expected = np.vdot(got_dense.ravel(), block.ravel())
    npt.assert_allclose(got, expected, atol=1e-10)


def _boundary_LP_RP(H_MPO, psi):
    """Hand-build the LP/RP environment tensors for a 2-site system, sidestepping the still
    np_conserved-only ``MPOEnvironment.init_LP``/``init_RP`` (``tenpy/networks/mpo.py``).

    For an ``L=2`` system, the environment `TwoSiteH` needs at ``i0=0`` (``LP`` left of site 0,
    ``RP`` right of site 1) is just the two trivial chain boundaries, decorated with a "one-hot"
    MPO bond leg at the ``IdL``/``IdR`` index -- exactly what ``init_LP``/``init_RP`` are supposed
    to build, just done here directly with cyten calls since that code path isn't ported yet.
    """
    backend = ct.get_backend('no_symmetry', 'numpy')
    env = MPSEnvironment(psi, psi)

    W0 = H_MPO.get_W(0)
    leg_L = W0.get_leg('wL').dual
    onehot_L = np.zeros(leg_L.dim, dtype=complex)
    IdL = 0  # TODO dummy, how to do it cleanly?
    onehot_L[IdL] = 1.0
    onehot_L = ct.SymmetricTensor.from_dense_block(onehot_L, codomain=[leg_L], backend=backend, labels=['wR'])
    LP = ct.outer(env.init_LP(0, 0).as_SymmetricTensor(), onehot_L)

    W1 = H_MPO.get_W(psi.L - 1)
    leg_R = W1.get_leg('wR').dual
    onehot_R = np.zeros(leg_R.dim, dtype=complex)
    IdR = -1  # TODO dummy, how to do it cleanly?
    onehot_R[IdR] = 1.0
    onehot_R = ct.SymmetricTensor.from_dense_block(onehot_R, codomain=[leg_R], backend=backend, labels=['wL'])
    RP = ct.outer(env.init_RP(psi.L - 1, 0).as_SymmetricTensor(), onehot_R)
    RP = ct.permute_legs(RP, codomain=['vL*', 'vL'], domain=['wL'])
    return LP, RP


class _BoundaryMPOEnv:
    """Minimal stand-in for :class:`~tenpy.networks.mpo.MPOEnvironment`, exposing only what
    :class:`~tenpy.algorithms.mps_common.TwoSiteH` needs (``get_LP``, ``get_RP``, ``H``), backed
    by the hand-built boundary tensors from :func:`_boundary_LP_RP`. Only valid at the outer
    boundary of the chain -- see :func:`_boundary_LP_RP`.
    """

    class _H:
        def __init__(self, H_MPO):
            self._H_MPO = H_MPO
            self.dtype = H_MPO.dtype

        def get_W(self, i):
            return self._H_MPO.get_W(i)

    def __init__(self, H_MPO, psi):
        self.LP, self.RP = _boundary_LP_RP(H_MPO, psi)
        self.H = self._H(H_MPO)

    def get_LP(self, i):
        return self.LP

    def get_RP(self, i):
        return self.RP


def test_two_site_h_matvec():
    """Check `TwoSiteH.matvec` against a hand-coded dense Hamiltonian, on a 2-site Ising model.

    Nothing has ever exercised the ``PlanarDiagram``/``matvec`` machinery `OneSiteH`/`TwoSiteH`
    got ported to (``adapt_effective_h``) -- `VariationalCompression` uses `DummyTwoSiteH`, whose
    `combine_theta` is a no-op and never calls `matvec`. This builds a real (tiny) Hamiltonian MPO
    and checks the actual LP-W0-W1-RP contraction.
    """
    J, g = 1.5, 0.9
    site = SpinSite(S=0.5, conserve=None)
    lat = lattice.Chain(2, site, bc='open', bc_MPS='finite')
    M = new_model.CouplingModel(lat)
    M.add_coupling(spin_spin_coupling([site, site], Jx=1.0), [0, 1], strength=-J)
    for i in range(2):
        M.add_coupling(spin_field_coupling([site], hz=1.0), [i], strength=-g)
    H_MPO = M.calc_H_MPO()

    rng = np.random.default_rng(2)
    d = site.dim
    block = rng.normal(size=(d, d)) + 1.0j * rng.normal(size=(d, d))
    block /= np.linalg.norm(block)
    backend = ct.get_backend('no_symmetry', 'numpy')
    psi_tensor = ct.SymmetricTensor.from_dense_block(
        block, codomain=[site.leg] * 2, backend=backend, labels=['p0', 'p1']
    )
    psi = MPS.from_full([site, site], psi_tensor, form=None, unit_cell_width=2)

    env = _BoundaryMPOEnv(H_MPO, psi)
    eff_H = TwoSiteH(env, i0=0)
    theta = psi.get_theta(0, n=2)
    got = eff_H.matvec(theta)
    got_dense = got.to_numpy(['vL', 'p0', 'p1', 'vR']).reshape(d, d)

    # hard-coded reference Hamiltonian, built purely with numpy (same spin-1/2 basis convention
    # as SpinSite(S=0.5, conserve=None): state 0 = down, state 1 = up)
    Sx = np.array([[0, 0.5], [0.5, 0]])
    Sz = np.array([[-0.5, 0], [0, 0.5]])
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
