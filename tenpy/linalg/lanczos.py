"""Lanczos algorithm for np_conserved arrays."""
# Copyright 2018 TeNPy Developers

from . import np_conserved as npc
from ..tools.params import get_parameter
import numpy as np
from scipy.linalg import expm
import warnings

__all__ = ['LanczosGroundState', 'LanczosEvolution', 'lanczos', 'gram_schmidt', 'plot_stats']


class LanczosGroundState:
    r"""Lanczos algorithm working on npc arrays.

    The Lanczos algorithm can finds extremal eigenvalues (in terms of magnitude) along with
    the corresponding eigenvectors. It assumes that the linear operator `H` is hermitian.
    Given a start vector `psi0`, it generates an orthonormal basis of the Krylov space,
    in which `H` is a small tridiagonal matrix, and solves the eigenvalue problem there.
    Finally, it transform the resulting ground state back into the original space.

    Parameters
    ----------
    H : :class:`~tenpy.linalg.sparse.LinearOperator`-like
        A hermitian linear operator. Must implement the method `matvec` acting on a
        :class:`~tenpy.linalg.np_conserved.Array`; nothing else required.
        The result has to have the same legs as the argument.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        The starting vector defining the Krylov basis.
        For finding the ground state, this should be the best guess available.
    params : dict
        Further optional parameters as described in the following table.
        Add a parameter ``verbose >=1`` to print the used parameters during runtime.
        The algorithm stops if *both* criteria for `e_tol` and `p_tol` are met
        or if the maximum number of steps was reached.

        ======= ====== ===============================================================
        key     type   description
        ======= ====== ===============================================================
        N_min   int    Minimum number of steps to perform.
        ------- ------ ---------------------------------------------------------------
        N_max   int    Maximum number of steps to perform.
        ------- ------ ---------------------------------------------------------------
        E_tol   float  Stop if energy difference per step < `E_tol`
        ------- ------ ---------------------------------------------------------------
        P_tol   float  Tolerance for the error estimate from the
                       Ritz Residual, stop if ``(RitzRes/gap)**2 < P_tol``
        ------- ------ ---------------------------------------------------------------
        min_gap float  Lower cutoff for the gap estimate used in the P_tol criterion.
        ------- ------ ---------------------------------------------------------------
        N_cache int    The maximum number of `psi` to keep in memory during the first
                       iteration. By default, we keep all states (up to N_max).
                       Set this to a number >= 2 if you are short on memory.
                       The penalty is that one needs another Lanczos iteration to
                       determine the ground state in the end, i.e., runtime is large.
        ------- ------ ---------------------------------------------------------------
        reortho bool   For poorly conditioned matrices, one can quickly loose
                       orthogonality of the generated Krylov basis.
                       If `reortho` is True, we re-orthogonalize against all the
                       vectors kept in cache to avoid that problem.
        ------- ------ ---------------------------------------------------------------
        cutoff  float  Cutoff to abort if `beta` (= norm of next vector in Krylov
                       basis before normalizing) is too small.
                       This is necessary if the rank of A is smaller than N_max -
                       then we get a complete basis of the Krylov space,
                       and `beta` will be zero.
        ======= ====== ===============================================================

    orthogonal_to : list of :class:`~tenpy.linalg.np_conserved.Array`
        Vectors (same tensor structure as psi) against which Lanczos will orthogonalize,
        ensuring that the result is perpendicular to them.
        (Assumes that the smallest eigenvalue is smaller than 0, which should *always* be the
        case if you want to find ground states with Lanczos!)

    Attributes
    ----------
    H : :class:`~tenpy.linalg.sparse.LinearOperator`-like
        The hermitian linear operator.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        The starting vector.
    orthogonal_to : list of :class:`~tenpy.linalg.np_conserved.Array`
        Vectors to orthogonalize against.
    N_min, N_max, E_tol, P_tol, N_cache, reortho:
        Parameters as described above.
    Es : ndarray, shape(N_max, N_max)
        ``Es[n, :]`` contains the energies of ``_T[:n+1, :n+1]`` in step `n`.
    _T : ndarray, shape (N_max + 1, N_max +1)
        The tridiagonal matrix representing `H` in the orthonormalized Krylov basis.
    _cutoff : float
        See parameter `cutoff`.
    _cache : list of psi0-like vectors
        The ONB of the Krylov space generated during the iteration.
        FIFO (first in first out) cache of at most N_cache vectors.
    _result_krylov : ndarray
        Result in the ONB of the Krylov space: ground state of `_T`.

    Notes
    -----
    I have computed the Ritz residual `RitzRes` according to
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
    Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
    ``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.
    """

    def __init__(self, H, psi0, params, orthogonal_to=[]):
        self.H = H
        self.psi0 = psi0.copy()
        self._params = params
        self.N_min = get_parameter(params, 'N_min', 2, "Lanczos")
        self.N_max = get_parameter(params, 'N_max', 20, "Lanczos")
        self.E_tol = get_parameter(params, 'E_tol', np.inf, "Lanczos")
        self.P_tol = get_parameter(params, 'P_tol', 1.e-14, "Lanczos")
        self.N_cache = get_parameter(params, 'N_cache', self.N_max, "Lanczos")
        self.min_gap = get_parameter(params, 'min_gap', 1.e-12, "Lanczos")
        self.reortho = get_parameter(params, 'reortho', False, "Lanczos")
        if self.N_cache < 2:
            raise ValueError("Need to cache at least two vectors.")
        if self.N_min < 2:
            raise ValueError("Should perform at least 2 steps.")
        self._cutoff = get_parameter(params, 'cutoff', np.finfo(psi0.dtype).eps * 100, "Lanczos")
        self.verbose = params.get('verbose', 0)
        if len(orthogonal_to) > 0:
            self.orthogonal_to, _ = gram_schmidt(orthogonal_to, self.verbose / 10)
        else:
            self.orthogonal_to = []
        self._cache = []
        self.Es = np.zeros([self.N_max, self.N_max], dtype=np.float)
        # First Lanczos iteration: Form tridiagonal form of A in the Krylov subspace, stored in T
        self._T = np.zeros([self.N_max + 1, self.N_max + 1], dtype=np.float)

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
        N = self._calc_T()
        E0 = self.Es[N - 1, 0]
        if self.verbose >= 1:
            if N > 1:
                msg = "Lanczos N={0:d}, gap={1:.3e}, DeltaE0={2:.3e}, _result_krylov[-1]={3:.3e}"
                print(
                    msg.format(N, self.Es[N - 1, 1] - E0, self.Es[N - 2, 0] - E0,
                               self._result_krylov[-1]))
            else:
                msg = "Lanczos N={0:d}, first alpha={1:.3e}, beta={2:.3e}"
                print(msg.format(N, self._T[0, 0], self._T[0, 1]))
        if N == 1:
            return E0, self.psi0.copy(), N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _calc_T(self):
        """Build the tridiagonal matrix `_T`. Returns the number of steps performed."""
        T = self._T
        w = self.psi0  # initialize
        beta = npc.norm(w)
        for k in range(self.N_max):
            w.iscale_prefactor(1. / beta)
            self._to_cache(w)
            w = self._apply_H(w)
            alpha = np.real(npc.inner(w, self._cache[-1], do_conj=True)).item()
            T[k, k] = alpha
            self._calc_result_krylov(k)
            w.iadd_prefactor_other(-alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    w.iadd_prefactor_other(-npc.inner(c, w, do_conj=True), c)
            elif k > 0:
                w.iadd_prefactor_other(-beta, self._cache[-2])
            beta = npc.norm(w)
            T[k, k + 1] = T[k + 1, k] = beta  # needed for the next step and convergence criteria
            if abs(beta) < self._cutoff or (k + 1 >= self.N_min and self._converged(k)):
                break
        return k + 1

    def _calc_result_full(self, N):
        """Transform self._result_krylov from the Krylov ONB to the original (npc) basis.

        Construct the result ``psi_f = sum_k  _result_krylov[k] psi[k]``, where ``psi[k]``
        is the k-th vector of the ONB of the Krylov space generated during the iteration.
        """
        vf = self._result_krylov
        assert N == len(vf) > 1
        psif = self.psi0 * vf[0]  # the start vector is still known and got normalized
        len_cache = len(self._cache)
        # and the last len_cache vectors have been cached
        for k in range(1, min(len_cache + 1, N)):
            psif.iadd_prefactor_other(vf[N - k], self._cache[-k])
        # other vectors are not cached, so we need to restart the Lanczos iteration.
        self._cache = []  # free memory: we need at least two more vectors

        T = self._T
        w = self.psi0  # initialize
        for k in range(0, N - len_cache - 1):
            self._to_cache(w)
            w = self._apply_H(w)
            alpha = T[k, k]
            w.iadd_prefactor_other(-alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    w.iadd_prefactor_other(-npc.inner(c, w, do_conj=True), c)
            elif k > 0:
                w.iadd_prefactor_other(-beta, self._cache[-2])
            beta = T[k, k + 1]  # = norm(w)
            w.iscale_prefactor(1. / beta)
            psif.iadd_prefactor_other(vf[k + 1], w)
        psif_norm = npc.norm(psif)
        if abs(1. - psif_norm) > 1.e-5:
            warnings.warn("Poorly conditioned Lanczos!")
            # One reason can be that `H` is not Hermitian
            # Otherwise, the matrix (even if small) might be ill conditioned.
            # If you get this warning, you can try to set the parameters
            # `reortho`=True and `N_cache` >= `N_max`
            if self.verbose > 1:
                print("poorly conditioned Lanczos! |psi_0| = {0:f}".format(psif_norm))
        psif.iscale_prefactor(1. / psif_norm)
        return psif

    def _to_cache(self, psi):
        """add psi to cache, keep at most N_cache."""
        cache = self._cache
        cache.append(psi)
        if len(cache) > self.N_cache:
            cache.pop(0)  # remove *first* entry

    def _apply_H(self, w):
        """apply H to w, but orthogonalize agains self.orthogonal_to."""
        # equivalent to using H' = P H P where P is the projector (1-sum_o |o><o|)
        if len(self.orthogonal_to) > 0:
            w = w.copy()
            for o in self.orthogonal_to:  # Project out
                w.iadd_prefactor_other(-npc.inner(o, w, do_conj=True), o)
        w = self.H.matvec(w)
        for o in self.orthogonal_to[::-1]:  # reverse: more obviously Hermitian.
            w.iadd_prefactor_other(-npc.inner(o, w, do_conj=True), o)
        return w

    def _calc_result_krylov(self, k):
        """calculate ground state of _T[:k+1, :k+1]"""
        T = self._T
        if k == 0:
            self.Es[0, 0] = T[0, 0]
            self._result_krylov = np.ones(1, np.float)
        else:
            # Diagonalize T
            E_T, v_T = np.linalg.eigh(T[:k + 1, :k + 1])
            self.Es[k, :k + 1] = E_T
            self._result_krylov = v_T[:, 0]  # ground state of _T

    def _converged(self, k):
        v0 = self._result_krylov
        E = self.Es[k, :]  # current energies
        RitzRes = np.abs(v0[k - 1] * self._T[k, k + 1])
        gap = max(E[1] - E[0], self.min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol


class LanczosEvolution(LanczosGroundState):
    """Calculate :math:`exp(delta H) |psi0>` using Lanczos.

    It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
    applied to the starting vector. Instead of diagonalizing the tri-diagonal `T` and taking the
    ground state, we now calculate ``exp(delta T) e_0 in the Krylov ONB, where
    ``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.

    Parameters
    ----------
    H, psi0, params :
        Hamiltonian, starting vector and parameters as defined in :class:`LanczosGroundState`.
        The parameters `E_tol` and `min_gap` are ignored,
        the parameters `P_tol` defines when convergence is reached, see :meth:`_converged` for
        details.

    Attributes
    ----------
    delta : float/complex
        Prefactor of H in the exponential.
    _result_norm : float
        Norm of the resulting vector.
    """

    def __init__(self, H, psi0, params):
        super().__init__(H, psi0, params)
        self.delta = None
        self._result_norm = 1.

    def run(self, delta):
        """Calculate ``expm(delta H).dot(psi0)`` using Lanczos.

        Parameters
        ----------
        delta : float/complex
            Time step by which we should evolve psi0: prefactor of H in the exponential.
            Note that the complex `i` is *not* included!

        Returns
        -------
        psi_f : :class:`~tenpy.linalg.np_conserved.Array`
            Best approximation for ``expm(delta H).dot(psi0)``
        N : int
            Krylov space dimension used.
        """
        self.delta = delta
        N = self._calc_T()
        if self.verbose >= 1:
            if N > 1:
                msg = "Lanczos N={0:d}, |result_krylov[-1]|={1:.3e}"
                print(msg.format(N, abs(self._result_krylov[-1])))
            else:
                msg = "Lanczos N={0:d}, first alpha={1:.3e}, beta={2:.3e}"
                print(msg.format(N, self._T[0, 0], self._T[0, 1]))
        if N == 1:
            result_full = self._result_krylov[0] * self.psi0
        else:
            result_full = self._calc_result_full(N)
        if delta.real != 0.:
            return result_full * self._result_norm, N
        # else:
        return result_full, N

    def _calc_result_krylov(self, k):
        """calculate expm(delta T) e0 for T= _T[:k+1, :k+1]"""
        T = self._T
        delta = self.delta
        if k == 0:
            E = T[0, 0]
            exp_dE = np.exp(delta * E)
            self._result_norm = np.sqrt(np.abs(exp_dE))
            self._result_krylov = np.ones(1, np.float) * (exp_dE / self._result_norm)
        else:
            e0 = np.zeros(k + 1, dtype=np.float)
            e0[0] = 1.
            exp_dT_e0 = expm(T[:k + 1, :k + 1] * delta).dot(e0)
            self._result_norm = np.linalg.norm(exp_dT_e0)
            self._result_krylov = exp_dT_e0 / self._result_norm

    def _converged(self, k):
        return np.abs(self._result_krylov[k]) < self.P_tol


def lanczos(H, psi, lanczos_params={}, orthogonal_to=[]):
    """Simple wrapper calling ``LanczosGroundState(H, psi, params, orthogonal_to).run()``"""
    return LanczosGroundState(H, psi, lanczos_params, orthogonal_to).run()


def gram_schmidt(vecs, rcond=1.e-14, verbose=0):
    """In place Gram-Schmidt Orthogonalization and normalization for npc Arrays.

    Parameters
    ----------
    vecs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The vectors which should be orthogonalized. Entries are modified *in place*.
        if a norm < rcond, they entry is set to `None`
    rcond : float
        Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.
    verbose : int
        Print additional output if verbose >= 1.

    Returns
    -------
    vecs : list of Array
        The ortho-normalized vectors (without any ``None``).
    ov : 2D Array
        For ``j >= i``, ``ov[j, i] = npc.inner(vecs[j], vecs[i], do_conj=True)``
        (where vecs[j] was orthogonalized to all ``vecs[k], k < i``).
    """
    k = len(vecs)
    ov = np.zeros((k, k), dtype=vecs[0].dtype)
    for j in range(k):
        n = ov[j, j] = npc.norm(vecs[j])
        if n > rcond:
            vecs[j].iscale_prefactor(1. / n)
            for i in range(j + 1, k):
                ov[j, i] = ov_ji = npc.inner(vecs[j], vecs[i], do_conj=True)
                vecs[i].iadd_prefactor_other(-ov_ji, vecs[j])
        else:
            if verbose >= 1:
                print("GramSchmidt: Rank defficient", n)
            vecs[j] = None
    vecs = [q for q in vecs if q is not None]
    if verbose >= 1:
        k = len(vecs)
        G = np.empty((k, k), dtype=vecs[0].dtype)
        for i, v in enumerate(vecs):
            for j, w in enumerate(vecs):
                G[i, j] = npc.inner(v, w, do_conj=True)
        print("GramSchmidt:", k, np.diag(ov), np.linalg.norm(G - np.eye(k)))
    return vecs, ov


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
