"""Lanczos algorithm for np_conserved arrays."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from .sparse import FlatHermitianOperator, OrthogonalNpcLinearOperator, ShiftNpcLinearOperator
import logging
logger = logging.getLogger(__name__)

from . import np_conserved as npc
from ..tools.params import asConfig
from ..tools.misc import argsort

__all__ = [
    'KrylovBased', 'Arnoldi', 'LanczosGroundState', 'LanczosEvolution', 'GMRES', 'lanczos_arpack',
    'gram_schmidt', 'iadd_prefactor_other', 'iscale_prefactor', 'plot_stats'
]


class KrylovBased:
    r"""Base class for iterative algorithms building a Krylov basis with np_conserved arrays.

    Algorithms like :class:`LanczosGroundState` and `:class:`ArnoldiDiagonalize`
    are based on iteratively building an orthonormal basis of the Krylov space spanned by
    ``|psi0>, H|psi0>, H^2|psi0>, ... H^N |psi0>``, where `N` is the number of iterations
    performed so far, and ``|psi0>`` is an initial guess and starting vector.
    During that iteration, the projection of `H` into the Krylov space is built, where it can
    be solved effectively (with `H` being just a N by N matrix), yielding the "Ritz" eigenvalues/
    eigenvectors. Finally, the solution can be translated back into the original space using the
    basis.

    An important strategy is also to (implicitly) restart the algorithm after some number of steps.
    This is **not** done here: when we use these classes, we usually have an explicit outer loop
    performed until convergence, e.g., the "sweeps" in DMRG.

    Parameters
    ----------
    H : :class:`~tenpy.linalg.sparse.NpcLinearOperator`-like
        A hermitian linear operator. Must implement the method `matvec` acting on a
        :class:`~tenpy.linalg.np_conserved.Array`; nothing else required.
        The result has to have the same legs as the argument.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        The starting vector defining the Krylov basis.
        For finding the ground state, this should be the best guess available.
        Note that it does not have to be a 1D "vector"; we are fine with viewing
        higher-rank tensors as vectors.
    options : dict
        Further optional parameters as described in :cfg:config:`Lanczos`.
        The algorithm stops if *both* criteria for `e_tol` and `p_tol` are met
        or if the maximum number of steps was reached.

    Options
    -------
    .. cfg:config :: KrylovBased

        N_min : int
            Minimum number of steps to perform.
        N_max : int
            Maximum number of steps to perform.
        P_tol : float
            Tolerance for the error estimate from the Ritz Residual,
            stop if ``(RitzRes/gap)**2 < P_tol``
        min_gap : float
            Lower cutoff for the gap estimate used in the P_tol criterion.
        cutoff : float
            Cutoff to abort if the norm of the new krylov vector is too small.
            This is necessary if the rank of `H` is smaller than `N_max`, but it's *not* the error
            tolerance for final values!
        E_shift : float
            Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
            :class:`~tenpy.linalg.sparse.ShiftNpcLinearOperator`.
            The ground state energy `E0` returned by :meth:`run` is made independent of the shift.
            This option is useful if the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`
            is used: the orthogonal vectors are *exact* eigenvectors with eigenvalue 0 independent
            of the shift, so you can use it to ensure that the energy is smaller than zero
            to avoid getting those.
        reortho : bool
            For poorly conditioned matrices, one can quickly loose orthogonality of the
            generated Krylov basis.
            If `reortho` is True, we re-orthogonalize against all the
            vectors kept in cache to avoid that problem.

    Attributes
    ----------
    options : :class:`~tenpy.tools.params.Config`
        Optional parameters.
    H : :class:`~tenpy.linalg.sparse.NpcLinearOperator`-like
        The linear operator used for building the Krylov space.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        The starting vector; normalized copy.
    N_min, N_max, P_tol, min_gap, _cutoff, E_shift:
        Parameters as described in the options.
    Es : ndarray, shape(N_max, N_max)
        ``Es[n, :]`` contains the energies of ``_h_krylov[:n+1, :n+1]`` in step `n`.
    _h_krylov : ndarray, shape (N_max + 1, N_max +1)
        The matrix representing `H` projected onto the orthonormalized Krylov basis.
    _psi0_norm : float
        Initial norm of the `psi0` parameter. Note that ``self.psi0`` gets normalized.
    _cache : list of psi0-like vectors
        The ONB of the Krylov space generated during the iteration.
        FIFO (first in first out) cache of at most `N_cache` vectors.
    _result_krylov : ndarray
        Result in the ONB of the Krylov space, e.g. the ground state of `_h_krylov`.
        What exactly this is depends on the subclass.

    Notes
    -----
    The Ritz residual `RitzRes` is computed according to
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
    Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
    ``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.
    """

    _dtype_h_krylov = np.complex128
    _dtype_E = np.complex128

    def __init__(self, H, psi0, options):
        self.H = H
        self.psi0 = psi0.copy()
        self._psi0_norm = None
        self.options = options = asConfig(options, self.__class__.__name__)
        self.N_min = options.get('N_min', 2, int)
        self.N_max = options.get('N_max', 20, int)
        self.N_cache = self.N_max
        self.P_tol = options.get('P_tol', 1.e-14, 'real')
        self.min_gap = options.get('min_gap', 1.e-12, 'real')
        self.reortho = options.get('reortho', False, bool)
        self.E_shift = options.get('E_shift', None, 'real')
        if self.N_min < 2:
            raise ValueError("Should perform at least 2 steps.")
        self._cutoff = options.get(
            'cutoff',
            np.finfo(psi0.dtype if not isinstance(psi0, list) else psi0[0].dtype).eps * 100,
            'real'
        )
        if self.E_shift is not None:
            if isinstance(self.H, OrthogonalNpcLinearOperator):
                self.H.orig_operator = ShiftNpcLinearOperator(self.H.orig_operator, self.E_shift)
            else:
                self.H = ShiftNpcLinearOperator(self.H, self.E_shift)
        self._cache = []
        self.Es = np.zeros([self.N_max, self.N_max], dtype=self._dtype_E)
        self._h_krylov = np.zeros([self.N_max + 1, self.N_max + 1], dtype=self._dtype_h_krylov)

    def run(self):
        raise NotImplementedError("subclasses should implement this")

    def _build_krylov(self):
        raise NotImplementedError("subclasses should implement this")

    def _calc_result_full(self, N):
        """Transform self._result_krylov from the Krylov ONB to the original (npc) basis.

        Construct the result ``psi_f = sum_k  _result_krylov[k] psi[k]``, where ``psi[k]``
        is the k-th vector of the ONB of the Krylov space generated during the iteration.
        """
        vf = self._result_krylov
        assert N == len(vf) > 1
        if isinstance(self.psi0, npc.Array):
            psif = self.psi0 * vf[0]  # the start vector is still known and got normalized
        else:
            assert isinstance(self.psi0, list)
            psif = [p * vf[0] for p in self.psi0]
        len_cache = len(self._cache)
        # and the last len_cache vectors have been cached
        for k in range(1, min(len_cache + 1, N)):
            self.iadd_prefactor_other(psif, vf[N - k], self._cache[-k])
        # other vectors are not cached, so we need to restart the Lanczos iteration.
        self._cache = []  # free memory: we need at least two more vectors

        self._rebuild_krylov_for_result_full(psif, N - len_cache - 1)

        psif_norm = npc.norm(psif)
        if abs(1. - psif_norm) > 1.e-5:
            # One reason can be that `H` is not Hermitian
            # Otherwise, the matrix (even if small) might be ill conditioned.
            # If you get this warning, you can try to set the parameters
            # `reortho`=True and `N_cache` >= `N_max`
            logger.warning("poorly conditioned H matrix in KrylovBased! |psi_0| = %f", psif_norm)
        self.iscale_prefactor(psif, 1. / psif_norm)
        return psif

    def _to_cache(self, psi):
        """add psi to cache, keep at most self.N_cache."""
        cache = self._cache
        cache.append(psi)
        if len(cache) > self.N_cache:
            cache.pop(0)  # remove *first* entry

    def _calc_result_krylov(self, k):
        raise NotImplementedError("subclasses should implement this")

    def iscale_prefactor(self, w, scale):
        iscale_prefactor(w, scale)

    def iadd_prefactor_other(self, w, alpha, v):
        iadd_prefactor_other(w, alpha, v)

class GMRES():
    def __init__(self, A, x, b, options):
        self.options = options = asConfig(options, self.__class__.__name__)
        self.N_min = options.get('N_min', 5, int)
        self.N_max = options.get('N_max', 20, int)
        self.restart = options.get('restart', 10, int)
        self.res = self.options.get('res', 1.e-8, 'real')
        self.A = A
        self.b = b.copy()
        self.x = x.copy() # Initial guess
        self.rs = [self.b.copy()]
        self.rs[0] = self.rs[0].iadd_prefactor_other(-1, self.A.matvec(self.x)) # residuals
        self.b_norm = npc.norm(self.b)
        self.total_error = [[npc.norm(self.rs[0]) / self.b_norm]]
        self.total_iters = []
        self.r_norm = npc.norm(self.rs[0])
        self.qs = [self.rs[0].copy()]
        self.qs[0].iscale_prefactor(1./self.r_norm)

        self.sine = np.zeros(self.N_max)*1.j
        self.cosine = np.zeros(self.N_max)*1.j
        self.e1 = npc.Array.from_ndarray_trivial(np.zeros(self.N_max+1)*1.j)
        self.e1[0] = 1
        self.e1.iscale_prefactor(self.r_norm)

        self.H = npc.Array.from_ndarray_trivial(np.zeros((self.N_max+1, self.N_max))*1.j)

    def run(self):
        for _ in range(self.restart):
            converged=False
            for k in range(0,self.N_max):
                self.arnoldi(k)
                self.apply_givens_rotation(k)
                self.e1[k+1] = -self.sine[k] * self.e1[k]
                self.e1[k] = self.cosine[k] * self.e1[k]
                error = np.abs(self.e1[k+1]) / self.b_norm # The residual is just the last element of $\beta$ vector (see Wikipedia) since $y$ is found exactly.
                self.total_error[-1].append(error)
                if error < self.res and k >= self.N_min:
                    converged=True
                    break
            self.total_iters.append(k+1)
            self.backsolve(k+1)
            Q_mat = npc.concatenate(self.qs[:k+1], axis=1)
            for i in range(k+1):
                self.x.iadd_prefactor_other(self.y[i], self.qs[i])
            if not converged:
                self.reset()
            else:
                break

        return self.x, npc.norm(self.A.matvec(self.x) - self.b) / self.b_norm, self.total_error, self.total_iters

    def arnoldi(self, k):
        # Iterative build orthogonal Krylov subspace and $H$ matrix.
        q = self.A.matvec(self.qs[-1])
        for i in range(k+1):
            self.H[i,k] = npc.inner(q, self.qs[i], axes='range', do_conj=True)
            q.iadd_prefactor_other(-self.H[i,k], self.qs[i])
        self.H[k+1,k] = npc.norm(q)
        q.iscale_prefactor(1./self.H[k+1,k])
        self.qs.append(q)

    def apply_givens_rotation(self, k):
        # Apply rotation to $H$ so that it becomes upper triangular.
        for i in range(k):
            temp = self.cosine[i] * self.H[i,k] + self.sine[i] * self.H[i+1,k]
            self.H[i+1,k] = -self.sine[i] * self.H[i,k] + self.cosine[i] * self.H[i+1,k]
            self.H[i,k] = temp

        self.givens_rotation(k)
        self.H[k,k] = self.cosine[k] * self.H[k,k] + self.sine[k] * self.H[k+1,k]
        self.H[k+1,k] = 0

    def givens_rotation(self, k):
        # Find cosine and sine such that the element below the diagonal of kth column of $H$ is removed.
        v1, v2 = self.H[k,k], self.H[k+1,k]
        t = np.sqrt(v1**2 + v2**2)
        self.cosine[k] = v1 / t
        self.sine[k] = v2 / t

    def backsolve(self, k):
        # $H$ is now a diagonal matrix; backsolve to find $y$ exactly.
        H = self.H[:k,:k]
        e2 = self.e1[:k]
        #e2[np.abs(e2.to_ndarray()) < 1.e-14] = 0 # N_max should be less than the size of A.
        self.y = npc.Array.from_ndarray_trivial(np.ones(k))*1.j
        for i in range(k-1,-1,-1):
            self.y[i] = e2[i]
            for j in range(i+1,k):
                self.y[i] -= H[i,j]*self.y[j]
            self.y[i] /= H[i,i]

    def reset(self):
        # Restart GMRES algorithm using current $x$ as initial guess.
        self.rs.append(self.b.copy())
        self.rs[-1] = self.rs[-1].iadd_prefactor_other(-1, self.A.matvec(self.x)) # residuals
        self.total_error.append([npc.norm(self.rs[-1]) / self.b_norm])
        self.r_norm = npc.norm(self.rs[-1])
        self.qs = [self.rs[-1].copy()]
        self.qs[-1].iscale_prefactor(1./self.r_norm)

        self.sine = np.zeros(self.N_max)*1.j
        self.cosine = np.zeros(self.N_max)*1.j
        self.e1 = npc.Array.from_ndarray_trivial(np.zeros(self.N_max+1)*1.j)
        self.e1[0] = 1
        self.e1.iscale_prefactor(self.r_norm)

        self.H = npc.Array.from_ndarray_trivial(np.zeros((self.N_max+1, self.N_max))*1.j)

class Arnoldi(KrylovBased):
    """Arnoldi method for diagonalizing square, non-hermitian/symmetric matrices.

    Generalization of :class:`LanczosGroundState`, allowing general, square matrices.

    Options
    -------
    .. cfg:config :: Arnoldi
        :include: KrylovBased

        which : ``'LM' | 'LR' | 'SR'``
            Determines which (extremal) eigenvalues to look for, name
            largest magnitude (in absolute value, ``'LM'``), or
            largest or smallest real part (``'LR'`` and ``'SR'``, respectively).
        num_ev : int
            Number of eigenvectors to look for/return in `run`.

    """
    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self.E_tol = self.options.get('E_tol', np.inf, 'real')
        self.which = self.options.get('which', 'LM', str)
        self.num_ev = self.options.get('num_ev', 1, int)  # number of desired eigenvectors

    def run(self):
        """Find the ground state of self.H.

        Returns
        -------
        E0s : numpy array
            Best eigenvalue estimates, :cfg:option:`Arnoldi.num_ev` entries,
            sorted according to :cfg:option:`Arnoldi.which`.
        psis : list of :class:`~tenpy.linalg.np_conserved.Array`
            Corresponding best eigenvectors (estimates).
        N : int
            Used dimension of the Krylov space, i.e., how many iterations where performed.
        """
        assert self.N_cache >= self.N_max
        N = self._build_krylov()
        E0 = self.Es[N - 1, :self.num_ev]
        if self.E_shift is not None:
            E0 = E0 - self.E_shift
        if N == 1:
            return E0, [self.psi0.copy()], N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _build_krylov(self):
        """Build the Krylov space and the projection of H into it.

        Returns the number of steps performed.
        """
        h = self._h_krylov
        w = self.psi0  # initialize
        norm = npc.norm(w)
        for k in range(self.N_max):
            self.iscale_prefactor(w, 1. / norm)
            self._to_cache(w)
            w = self.H.matvec(w)
            for i, v_i in enumerate(self._cache):
                h[i, k] = ov = npc.inner(v_i, w, axes='range', do_conj=True)
                self.iadd_prefactor_other(w, -ov, v_i)
            h[k + 1, k] = norm = npc.norm(w)
            self._calc_result_krylov(k)
            if norm < self._cutoff or (k + 1 >= self.N_min and self._converged(k)):
                break
        return k + 1

    def _calc_result_krylov(self, k):
        """calculate ground state of _h_krylov[:k+1, :k+1]"""
        h = self._h_krylov
        if k == 0:
            self.Es[0, 0] = h[0, 0]
            self._result_krylov = np.ones([1, 1], self._dtype_h_krylov)
        else:
            # Diagonalize h
            E_kr, v_kr = np.linalg.eig(h[:k + 1, :k + 1]) # not hermitian!
            sort = argsort(E_kr, self.which)
            self.Es[k, :k + 1] = E_kr[sort]
            self._result_krylov = v_kr[:, sort]  # ground state of _h_krylov

    def _calc_result_full(self, N):
        """Transform self._result_krylov from the Krylov ONB to the original (npc) basis.

        Construct the result ``psi_f = sum_k  _result_krylov[k] psi[k]``, where ``psi[k]``
        is the k-th vector of the ONB of the Krylov space generated during the iteration.
        """
        psis = []
        for i in range(min(N, self.num_ev)):
            vf = self._result_krylov[:, i]
            vf = np.real_if_close(vf)  # try to convert to real:
            # e.g. the dominant eigenvectors of the MPS transfermatrix should be equivalent to
            # the power method, which will be purely real for H.dtype=float, even if there might
            # be other eigenvectors which are complex
            assert N == len(vf) > 1
            krylov_basis = self._cache
            assert len(krylov_basis) >= N

            if isinstance(self.psi0, npc.Array):
                psi = vf[0] * krylov_basis[0]  # copy!
            else:
                assert isinstance(self.psi0, list)
                psi = [p * vf[0] for p in krylov_basis[0]]

            # and the last len_cache vectors have been cached
            for k in range(1, N):
                self.iadd_prefactor_other(psi, vf[k], krylov_basis[k])

            psi_norm = npc.norm(psi)
            if abs(1. - psi_norm) > 1.e-5:
                # One reason can be that `H` is not Hermitian
                # Otherwise, the matrix (even if small) might be ill conditioned.
                # If you get this warning, you can try to set the parameters
                # `reortho`=True and `N_cache` >= `N_max`
                logger.warning("poorly conditioned H matrix in Arnoldi! |psi| = %f", psi_norm)
            self.iscale_prefactor(psi, 1. / psi_norm)
            psis.append(psi)
        return psis

    def _to_cache(self, psi):
        """add psi to cache, keep at most self.N_cache."""
        cache = self._cache
        cache.append(psi)
        assert len(cache) <= self.N_cache

    def _converged(self, k):
        v0 = self._result_krylov[:, 0]
        E = self.Es[k, :]  # current energies
        RitzRes = abs(v0[k]) * self._h_krylov[k + 1, k]
        gap = max(min([np.min(np.abs(E[i+1:] - E[i])) for i in range(self.num_ev)]), self.min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol


class LanczosGroundState(KrylovBased):
    """Lanczos algorithm to find the ground state.

    **Assumes** that `H` is hermitian.
    
    Options
    -------
    .. cfg:config :: LanczosGroundState
        :include: KrylovBased

        E_tol : float
            Stop if energy difference per step < `E_tol`
        N_cache : int
            The maximum number of `psi` to keep in memory during the first iteration.
            By default, we keep all states (up to N_max).
            Set this to a number >= 2 if you are short on memory.
            The penalty is that one needs another Lanczos iteration to
            determine the ground state in the end, i.e., runtime is large.
    """

    _dtype_h_krylov = np.float64
    _dtype_E = np.float64

    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self.E_tol = self.options.get('E_tol', np.inf, 'real')
        self.N_cache = self.options.get('N_cache', self.N_max, int)
        if self.N_cache < 2:
            raise ValueError("Need to cache at least two vectors.")

    def run(self):
        """Find the ground state of H.

        Returns
        -------
        E0 : float
            Ground state energy (estimate).
        psi0 : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state vector (estimate).
        N : int
            Used dimension of the Krylov space, i.e., how many iterations where performed.
        """
        N = self._build_krylov()
        E0 = self.Es[N - 1, 0]
        if N > 1:
            logger.debug("Lanczos N=%d, gap=%.3e, DeltaE0=%.3e, _result_krylov[-1]=%.3e", N,
                         self.Es[N - 1, 1] - E0, self.Es[N - 2, 0] - E0, self._result_krylov[-1])
        else:
            logger.debug("Lanczos N=%d, first alpha=%.3e, beta=%.3e", N, self._h_krylov[0, 0],
                         self._h_krylov[0, 1])
        if self.E_shift is not None:
            E0 -= self.E_shift
        if N == 1:
            return E0, self.psi0.copy(), N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _build_krylov(self):
        """Build the Krylov space and the projection of H into it.

        Returns the number of steps performed.
        """
        h = self._h_krylov
        w = self.psi0  # initialize
        beta = npc.norm(w)
        if beta < self._cutoff:
            raise ValueError(f'Norm of self.psi0 too small: {beta}')
        if self._psi0_norm is None:
            # this is only needed for normalization in LanczosEvolution
            self._psi0_norm = beta
        for k in range(self.N_max):
            self.iscale_prefactor(w, 1. / beta)
            self._to_cache(w)
            w = self.H.matvec(w)
            alpha = np.real(npc.inner(w, self._cache[-1], axes='range', do_conj=True)).item()
            h[k, k] = alpha
            self._calc_result_krylov(k)
            self.iadd_prefactor_other(w, -alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    self.iadd_prefactor_other(w, -npc.inner(c, w, axes='range', do_conj=True), c)
            elif k > 0:
                self.iadd_prefactor_other(w, -beta, self._cache[-2])
            beta = npc.norm(w)
            h[k, k + 1] = h[k + 1, k] = beta  # needed for the next step and convergence criteria
            if abs(beta) < self._cutoff or (k + 1 >= self.N_min and self._converged(k)):
                break
        return k + 1

    def _converged(self, k):
        v0 = self._result_krylov
        E = self.Es[k, :]  # current energies
        RitzRes = abs(v0[k]) * self._h_krylov[k, k + 1]
        gap = max(E[1] - E[0], self.min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol

    def _rebuild_krylov_for_result_full(self, psif, N_max):
        vf = self._result_krylov
        h = self._h_krylov
        w = self.psi0  # initialize
        for k in range(0, N_max):
            self._to_cache(w)
            w = self.H.matvec(w)
            alpha = h[k, k]
            self.iadd_prefactor_other(w, -alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    self.iadd_prefactor_other(w, -npc.inner(c, w, axes='range', do_conj=True), c)
            elif k > 0:
                self.iadd_prefactor_other(w, -beta, self._cache[-2])  # noqa: F821
            beta = h[k, k + 1]  # = norm(w)
            self.iscale_prefactor(w, 1. / beta)
            self.iadd_prefactor_other(psif, vf[k + 1], w)
        # continue in _calc_result_full

    def _calc_result_krylov(self, k):
        """calculate ground state of _h_krylov[:k+1, :k+1]"""
        h = self._h_krylov
        if k == 0:
            self.Es[0, 0] = h[0, 0]
            self._result_krylov = np.ones(1, np.float64)
        else:
            # Diagonalize h
            E_kr, v_kr = np.linalg.eigh(h[:k + 1, :k + 1])
            self.Es[k, :k + 1] = E_kr
            self._result_krylov = v_kr[:, 0]  # ground state of _h_krylov


class LanczosEvolution(LanczosGroundState):
    """Calculate :math:`exp(delta H) |psi0>` using Lanczos.

    It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
    applied to the starting vector. Instead of diagonalizing the tri-diagonal `h` and taking the
    ground state, we now calculate ``exp(delta h) e_0`` in the Krylov ONB, where
    ``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.

    Parameters
    ----------
    H, psi0, options :
        Hamiltonian, starting vector and parameters as defined in :class:`LanczosGroundState`.
        The option :cfg:option`LanczosEvolution.P_tol` defines when convergence is reached,
        see :meth:`_converged` for details.

    Options
    -------
    .. cfg:config :: LanczosEvolution
        :include: LanczosGroundState

        E_tol :
            Ignored.
        min_gap :
            Ignored.

    Attributes
    ----------
    delta : float/complex
        Prefactor of H in the exponential.
    _result_norm : float
        Norm of the resulting vector.
    """
    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self._result_norm = 1.
        self.delta = None  # set in run()

    def run(self, delta, normalize=None):
        """Calculate ``expm(delta H).dot(psi0)`` using Lanczos.

        Parameters
        ----------
        delta : float/complex
            Time step by which we should evolve psi0: prefactor of H in the exponential.
            Note that the complex `i` is *not* included!
        normalize : bool
            Whether to normalize the resulting state.
            Defaults to ``np.real(delta) == 0``.

        Returns
        -------
        psi_f : :class:`~tenpy.linalg.np_conserved.Array`
            Best approximation for ``expm(delta H).dot(psi0)``.
            If :cfg:option:`Lanczos.E_shift` is used, it's an approximation for
            ``expm(delta (H + E_shift)).dot(psi)``.
        N : int
            Krylov space dimension used.
        """
        self.delta = delta
        N = self._build_krylov()
        if N > 1:
            logger.debug("Lanczos N=%d, |result_krylov[-1]|=%.3e", N, abs(self._result_krylov[-1]))
        else:
            logger.debug("Lanczos N=%d, first alpha=%.3e, beta=%.3e", N, self._h_krylov[0, 0],
                         self._h_krylov[0, 1])
        if N == 1:
            result_full = self._result_krylov[0] * self.psi0  # _result_krylov[0] is only a phase
        else:
            result_full = self._calc_result_full(N)
        # result_full is normalized at this point
        if normalize is None:
            normalize = np.real(delta) == 0.
        if normalize:
            return result_full, N
        # else:
        return (self._psi0_norm * self._result_norm) * result_full, N

    def _calc_result_krylov(self, k):
        """calculate ``expm(delta h).dot(e0)`` for ``h = _h_krylov[:k+1, :k+1]``"""

        # self._result_krylov should be a normalized vector.
        h = self._h_krylov
        delta = self.delta
        if k == 0:
            E = h[0, 0]
            exp_dE = np.exp(delta * E)
            self._result_norm = np.abs(exp_dE)  # np.linalg.norm for individual element
            self._result_krylov = np.array([exp_dE / self._result_norm])
        else:
            #     e0 = np.zeros(k + 1, dtype=float)
            #     e0[0] = 1.
            #     exp_dH_e0 = expm(_h_krylov[:k + 1, :k + 1] * delta).dot(e0)
            # scipy.linalg.expm is using sparse tools; instead fully diagonalize
            # given that h is hermitian, this is easy:
            # H V = V diag(E)  -> H  = V E V^D
            # exp(H*delta) e_0 = V diag(exp(E*delta)) V^D e_0
            E_kr, v_kr = np.linalg.eigh(h[:k + 1, :k + 1])
            exp_dH_e0 = np.dot(v_kr, np.exp(E_kr * delta) * np.conj(v_kr[0, :]))

            self._result_norm = np.linalg.norm(exp_dH_e0)
            self._result_krylov = exp_dH_e0 / self._result_norm

    def _converged(self, k):
        return np.abs(self._result_krylov[k]) < self.P_tol


def lanczos_arpack(H, psi, options={}):
    """Use :func:`scipy.sparse.linalg.eigsh` to find the ground state of `H`.

    This function has the same call/return structure as :func:`lanczos`, but uses
    the ARPACK package through the functions :func:`~tenpy.tools.math.speigsh` instead of the
    custom lanczos implementation in :class:`LanczosGroundState`.
    This function is mostly intended for debugging, since it requires to convert the vector
    from np_conserved :class:`~tenpy.linalg.np_conserved.Array` into a flat numpy array
    and back during *each* `matvec`-operation!

    Parameters
    ----------
    H, psi, options :
        See :class:`LanczosGroundState`.
        `H` and `psi` should have/use labels.

    Returns
    -------
    E0 : float
        Ground state energy.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        Ground state vector.
    """
    options = asConfig(options, "Lanczos")
    H_flat, psi_flat = FlatHermitianOperator.from_guess_with_pipe(H.matvec, psi, dtype=H.dtype)
    tol = options.get('P_tol', 1.e-14, 'real')
    N_min = options.get('N_min', None, int)
    Es, Vs = H_flat.eigenvectors(num_ev=1, which='SA', v0=psi_flat, tol=tol, ncv=N_min)
    psi0 = Vs[0].split_legs(0).itranspose(psi.get_leg_labels())
    return Es[0], psi0


def gram_schmidt(vecs, rcond=1.e-14):
    """In place Gram-Schmidt Orthogonalization and normalization for npc Arrays.

    Parameters
    ----------
    vecs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The vectors which should be orthogonalized.
        All with the same *order* of the legs. Entries are modified *in place*.
        if a norm < rcond, the entry is set to `None`.
    rcond : float
        Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.

    Returns
    -------
    vecs : list of Array
        The ortho-normalized vectors (without any ``None``).
    """
    res = []
    for vec in vecs:
        for other in res:
            ov = npc.inner(other, vec, 'range', do_conj=True)
            iadd_prefactor_other(vec, -ov, other)
        n = npc.norm(vec)
        if n > rcond:
            iscale_prefactor(vec, 1. / n)
            res.append(vec)
    return res


def iscale_prefactor(w, scale):
    if not isinstance(w, list):
        w.iscale_prefactor(scale)
    else:
        for a in w:
            a.iscale_prefactor(scale)

def iadd_prefactor_other(w, alpha, v):
    if not isinstance(w, list):
        w.iadd_prefactor_other(alpha, v)
    else:
        for a, b in zip(w, v):
            a.iadd_prefactor_other(alpha, b)

def plot_stats(ax, Es):
    """Plot the convergence of the energies.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The axes on which we should plot.
    Es : list of ndarray.
        The energies :attr:`Lanczos.Es`.
    """
    ks = [[k] * len(E) for k, E in enumerate(Es)]
    ks = np.array(sum(ks, []))
    Es = np.array(sum([list(E) for E in Es], []))
    ax.scatter(ks, np.array(Es))
    ax.set_xlabel("Lanczos step")
    ax.set_ylabel("Ritz Values (= energy estimates)")
