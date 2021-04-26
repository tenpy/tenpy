"""Lanczos algorithm for np_conserved arrays."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import warnings
import numpy as np
from scipy.linalg import expm
import scipy.sparse
from .sparse import FlatHermitianOperator, OrthogonalNpcLinearOperator, ShiftNpcLinearOperator
import logging
logger = logging.getLogger(__name__)

from . import np_conserved as npc
from ..tools.params import asConfig
from ..tools.math import speigsh

__all__ = [
    'LanczosGroundState', 'LanczosEvolution', 'lanczos', 'lanczos_arpack', 'gram_schmidt',
    'plot_stats'
]


class LanczosGroundState:
    r"""Lanczos algorithm working on npc arrays.

    The Lanczos algorithm can finds extremal eigenvalues (in terms of magnitude) along with
    the corresponding eigenvectors. It assumes that the linear operator `H` is hermitian.
    Given a start vector `psi0`, it generates an orthonormal basis of the Krylov space,
    in which `H` is a small tridiagonal matrix, and solves the eigenvalue problem there.
    Finally, it transform the resulting ground state back into the original space.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `params` to :attr:`options`.

    .. deprecated :: 0.6.0
        Going to remove the `orthogonal_to` argument.
        Instead, replace H with `OrthogonalNpcLinearOperator(H, orthogonal_to)`
        using the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`.

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
    orthogonal_to : list of :class:`~tenpy.linalg.np_conserved.Array`
        Vectors (same tensor structure as psi) against which Lanczos will orthogonalize,
        ensuring that the result is perpendicular to them.
        (Assumes that the smallest eigenvalue is smaller than 0, which should *always* be the
        case if you want to find ground states with Lanczos!)

    Options
    -------
    .. cfg:config :: Lanczos

        N_min : int
            Minimum number of steps to perform.
        N_max : int
            Maximum number of steps to perform.
        E_tol : float
            Stop if energy difference per step < `E_tol`
        P_tol : float
            Tolerance for the error estimate from the Ritz Residual,
            stop if ``(RitzRes/gap)**2 < P_tol``
        min_gap : float
            Lower cutoff for the gap estimate used in the P_tol criterion.
        N_cache : int
            The maximum number of `psi` to keep in memory during the first iteration.
            By default, we keep all states (up to N_max).
            Set this to a number >= 2 if you are short on memory.
            The penalty is that one needs another Lanczos iteration to
            determine the ground state in the end, i.e., runtime is large.
        reortho : bool
            For poorly conditioned matrices, one can quickly loose orthogonality of the
            generated Krylov basis.
            If `reortho` is True, we re-orthogonalize against all the
            vectors kept in cache to avoid that problem.
        cutoff : float
            Cutoff to abort if `beta` (= norm of next vector in Krylov basis before normalizing)
            is too small.
            This is necessary if the rank of `H` is smaller than `N_max` -
            then we get a complete basis of the Krylov space, and `beta` will be zero.
        E_shift : float
            Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
            :class:`~tenpy.linalg.sparse.ShiftNpcLinearOperator`.
            Since the Lanczos algorithm finds extremal eigenvalues, this can help convergence.
            Moreover, if the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator` is used,
            the orthogonal vectors are *exact* eigenvectors with eigenvalue 0,
            so you have to ensure that the energy is smaller than zero to avoid getting those.
            The ground state energy `E0` returned by :meth:`run` is made independent of the shift.

    Attributes
    ----------
    options : :class:`~tenpy.tools.params.Config`
        Optional parameters.
    H : :class:`~tenpy.linalg.sparse.NpcLinearOperator`-like
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
    _psi0_norm : float
        Initial norm of the `psi0` passed.
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
    def __init__(self, H, psi0, options, orthogonal_to=[]):
        self.H = H
        self.psi0 = psi0.copy()
        self._psi0_norm = None
        self.options = options = asConfig(options, "Lanczos")
        self.N_min = options.get('N_min', 2)
        self.N_max = options.get('N_max', 20)
        self.E_tol = options.get('E_tol', np.inf)
        self.P_tol = options.get('P_tol', 1.e-14)
        self.N_cache = options.get('N_cache', self.N_max)
        self.min_gap = options.get('min_gap', 1.e-12)
        self.reortho = options.get('reortho', False)
        self.E_shift = options.get('E_shift', None)
        if self.N_cache < 2:
            raise ValueError("Need to cache at least two vectors.")
        if self.N_min < 2:
            raise ValueError("Should perform at least 2 steps.")
        self._cutoff = options.get('cutoff', np.finfo(psi0.dtype).eps * 100)
        if self.E_shift is not None:
            if isinstance(self.H, OrthogonalNpcLinearOperator):
                self.H.orig_operator = ShiftNpcLinearOperator(self.H.orig_operator, self.E_shift)
            else:
                self.H = ShiftNpcLinearOperator(self.H, self.E_shift)
        if len(orthogonal_to) > 0:
            msg = ("Lanczos argument `orthogonal_to` is deprecated and will be removed.\n"
                   "Instead, replace `H` with  `OrthogonalNpcLinearOperator(H, orthogonal_to)`.")
            warnings.warn(msg, category=FutureWarning, stacklevel=2)
            self.H = OrthogonalNpcLinearOperator(self.H, orthogonal_to)
        self._cache = []
        self.Es = np.zeros([self.N_max, self.N_max], dtype=np.float64)
        # First Lanczos iteration: Form tridiagonal form of A in the Krylov subspace, stored in T
        self._T = np.zeros([self.N_max + 1, self.N_max + 1], dtype=np.float64)

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
        if N > 1:
            logger.debug("Lanczos N=%d, gap=%.3e, DeltaE0=%.3e, _result_krylov[-1]=%.3e", N,
                         self.Es[N - 1, 1] - E0, self.Es[N - 2, 0] - E0, self._result_krylov[-1])
        else:
            logger.debug("Lanczos N=%d, first alpha=%.3e, beta=%.3e", N, self._T[0, 0],
                         self._T[0, 1])
        if self.E_shift is not None:
            E0 -= self.E_shift
        if N == 1:
            return E0, self.psi0.copy(), N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _calc_T(self):
        """Build the tridiagonal matrix `_T`.

        Returns the number of steps performed.
        """
        T = self._T
        w = self.psi0  # initialize
        beta = npc.norm(w)
        if self._psi0_norm is None:
            # this is only needed for normalization in LanczosEvolution
            self._psi0_norm = beta
        for k in range(self.N_max):
            w.iscale_prefactor(1. / beta)
            self._to_cache(w)
            w = self.H.matvec(w)
            alpha = np.real(npc.inner(w, self._cache[-1], axes='range', do_conj=True)).item()
            T[k, k] = alpha
            self._calc_result_krylov(k)
            w.iadd_prefactor_other(-alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    w.iadd_prefactor_other(-npc.inner(c, w, axes='range', do_conj=True), c)
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
            w = self.H.matvec(w)
            alpha = T[k, k]
            w.iadd_prefactor_other(-alpha, self._cache[-1])
            if self.reortho:
                for c in self._cache[:-1]:
                    w.iadd_prefactor_other(-npc.inner(c, w, 'range', do_conj=True), c)
            elif k > 0:
                w.iadd_prefactor_other(-beta, self._cache[-2])
            beta = T[k, k + 1]  # = norm(w)
            w.iscale_prefactor(1. / beta)
            psif.iadd_prefactor_other(vf[k + 1], w)
        psif_norm = npc.norm(psif)
        if abs(1. - psif_norm) > 1.e-5:
            # One reason can be that `H` is not Hermitian
            # Otherwise, the matrix (even if small) might be ill conditioned.
            # If you get this warning, you can try to set the parameters
            # `reortho`=True and `N_cache` >= `N_max`
            logger.warning("poorly conditioned Lanczos! |psi_0| = %f", psif_norm)
        psif.iscale_prefactor(1. / psif_norm)
        return psif

    def _to_cache(self, psi):
        """add psi to cache, keep at most N_cache."""
        cache = self._cache
        cache.append(psi)
        if len(cache) > self.N_cache:
            cache.pop(0)  # remove *first* entry

    def _calc_result_krylov(self, k):
        """calculate ground state of _T[:k+1, :k+1]"""
        T = self._T
        if k == 0:
            self.Es[0, 0] = T[0, 0]
            self._result_krylov = np.ones(1, np.float64)
        else:
            # Diagonalize T
            E_T, v_T = np.linalg.eigh(T[:k + 1, :k + 1])
            self.Es[k, :k + 1] = E_T
            self._result_krylov = v_T[:, 0]  # ground state of _T

    def _converged(self, k):
        v0 = self._result_krylov
        E = self.Es[k, :]  # current energies
        RitzRes = abs(v0[k - 1]) * self._T[k, k + 1]
        gap = max(E[1] - E[0], self.min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol


class LanczosEvolution(LanczosGroundState):
    """Calculate :math:`exp(delta H) |psi0>` using Lanczos.

    It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
    applied to the starting vector. Instead of diagonalizing the tri-diagonal `T` and taking the
    ground state, we now calculate ``exp(delta T) e_0`` in the Krylov ONB, where
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
        :include: Lanczos

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
        N = self._calc_T()
        if N > 1:
            logger.debug("Lanczos N=%d, |result_krylov[-1]|=%.3e", N, abs(self._result_krylov[-1]))
        else:
            logger.debug("Lanczos N=%d, first alpha=%.3e, beta=%.3e", N, self._T[0, 0],
                         self._T[0, 1])
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
        """calculate ``expm(delta T).dot(e0)`` for ``T = _T[:k+1, :k+1]``"""

        # self._result_krylov should be a normalized vector.
        T = self._T
        delta = self.delta
        if k == 0:
            E = T[0, 0]
            exp_dE = np.exp(delta * E)
            self._result_norm = np.abs(exp_dE)  # np.linalg.norm for individual element
            self._result_krylov = np.array([exp_dE / self._result_norm])
        else:
            #     e0 = np.zeros(k + 1, dtype=float)
            #     e0[0] = 1.
            #     exp_dT_e0 = expm(T[:k + 1, :k + 1] * delta).dot(e0)
            # scipy.linalg.expm is using sparse tools; instead fully diagonalize
            # given that T is hermitian, this is easy:
            # H V = V diag(E)  -> H  = V E V^D
            # exp(H*delta) e_0 = V diag(exp(E*delta)) V^D e_0
            E_T, v_T = np.linalg.eigh(T[:k + 1, :k + 1])
            exp_dT_e0 = np.dot(v_T, np.exp(E_T * delta) * np.conj(v_T[0, :]))

            self._result_norm = np.linalg.norm(exp_dT_e0)
            self._result_krylov = exp_dT_e0 / self._result_norm

    def _converged(self, k):
        return np.abs(self._result_krylov[k]) < self.P_tol


def lanczos(H, psi, options={}, orthogonal_to=[]):
    """Simple wrapper calling ``LanczosGroundState(H, psi, options, orthogonal_to).run()``

    .. deprecated :: 0.6.0
        Going to remove the `orthogonal_to` argument.
        Instead, replace H with `OrthogonalNpcLinearOperator(H, orthogonal_to)`
        using the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`.

    Parameters
    ----------
    H, psi, options, orthogonal_to :
        See :class:`LanczosGroundState`.

    Returns
    -------
    E0, psi0, N :
        See :meth:`LanczosGroundState.run`.
    """
    return LanczosGroundState(H, psi, options, orthogonal_to).run()


def lanczos_arpack(H, psi, options={}, orthogonal_to=[]):
    """Use :func:`scipy.sparse.linalg.eigsh` to find the ground state of `H`.

    This function has the same call/return structure as :func:`lanczos`, but uses
    the ARPACK package through the functions :func:`~tenpy.tools.math.speigsh` instead of the
    custom lanczos implementation in :class:`LanczosGroundState`.
    This function is mostly intended for debugging, since it requires to convert the vector
    from np_conserved :class:`~tenpy.linalg.np_conserved.Array` into a flat numpy array
    and back during *each* `matvec`-operation!

    .. deprecated :: 0.6.0
        Going to remove the `orthogonal_to` argument.
        Instead, replace H with `OrthogonalNpcLinearOperator(H, orthogonal_to)`
        using the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`.

    Parameters
    ----------
    H, psi, options, orthogonal_to :
        See :class:`LanczosGroundState`.
        `H` and `psi` should have/use labels.

    Returns
    -------
    E0 : float
        Ground state energy.
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        Ground state vector.
    """
    if len(orthogonal_to) > 0:
        msg = ("Lanczos argument `orthogonal_to` is deprecated and will be removed.\n"
               "Instead, replace `H` with  `OrthogonalNpcLinearOperator(H, orthogonal_to)`.")
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        H = OrthogonalNpcLinearOperator(self.H, orthogonal_to)
    options = asConfig(options, "Lanczos")
    H_flat, psi_flat = FlatHermitianOperator.from_guess_with_pipe(H.matvec, psi, dtype=H.dtype)
    tol = options.get('P_tol', 1.e-14)
    N_min = options.get('N_min', None)
    Es, Vs = H_flat.eigenvectors(num_ev=1, which='SA', v0=psi_flat, tol=tol, ncv=N_min)
    psi0 = Vs[0].split_legs(0).itranspose(psi.get_leg_labels())
    return Es[0], psi0


def gram_schmidt(vecs, rcond=1.e-14, verbose=None):
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
    ov : 2D Array
        For ``j >= i``, ``ov[j, i] = npc.inner(vecs[j], vecs[i], 'range', do_conj=True)``
        (where vecs[j] was orthogonalized to all ``vecs[k], k < i``).
    """
    if verbose is not None:
        warnings.warn("Dropped verbose argument", category=FutureWarning, stacklevel=2)
    k = len(vecs)
    ov = np.zeros((k, k), dtype=vecs[0].dtype)
    for j in range(k):
        n = ov[j, j] = npc.norm(vecs[j])
        if n > rcond:
            vecs[j].iscale_prefactor(1. / n)
            for i in range(j + 1, k):
                ov[j, i] = ov_ji = npc.inner(vecs[j], vecs[i], 'range', do_conj=True)
                vecs[i].iadd_prefactor_other(-ov_ji, vecs[j])
        else:
            vecs[j] = None
    vecs = [q for q in vecs if q is not None]
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
