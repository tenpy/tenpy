"""Lanczos implementation for np_conserved arrays."""

from . import np_conserved as npc
from ..tools.params import get_parameter
import numpy as np
import warnings


class LinearOperator(object):
    """Generic Linear Operator for :class:`~tenpy.linalg.np_conserved.Array`.

    This is a prototype for a Linear Operator as required by the Lanczos algorithm.

    Parameters
    ----------
    M : :class:`~tenpy.linalg.np_conserved.Array`
        A square matrix defining the `matvec` function as contraction
    """
    def __init__(self, M):
        self.M = M

    def matvec(self, v):
        """Application of the Linear operator to a vector."""
        return npc.tensordot(self.M, v, axes=[[1], [0]])


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
        Print additional output if verbose > 0.

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
            vecs[j] *= 1./n
            for i in range(j+1, k):
                ov[j, i] = ov_ji = npc.inner(vecs[j], vecs[i], do_conj=True)
                vecs[i] -= ov_ji * vecs[j]
        else:
            if verbose > 0:
                print "GramSchmidt: Rank defficient", n
            vecs[j] = None
    vecs = [q for q in vecs if q is not None]
    if verbose > 0:
        k = len(vecs)
        G = np.empty((k, k), dtype=vecs[0].dtype)
        for i, v in enumerate(vecs):
            for j, w in enumerate(vecs):
                G[i, j] = npc.inner(v, w, do_conj=True)
        print "GramSchmidt:", k, np.diag(ov),  np.linalg.norm(G - np.eye(k))
    return vecs, ov


def lanczos(A, psi, lanczos_params={}, orthogonal_to=[]):
    """Frank's Lanczos Algorithm for finding the lowest Eigenvector.

    Parameters
    ----------
    A : LinearOperator
        A hermitian linear operator. Must implement the method `matvec` acting on a
        :class:`~tenpy.linalg.np_conserved.Array`.
    psi : :class:`~tenpy.linalg.np_conserved.Array`
        The starting vector. Should be the best guess available.
    TEBD_parameters : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.

        ======= ====== ====================================================
        key     type   description
        ======= ====== ====================================================
        N_min   int    Minimum number of steps to perform.
        ------- ------ ----------------------------------------------------
        N_max   int    Maximum number of steps to perform.
        ------- ------ ----------------------------------------------------
        e_tol   float  Stop if energy difference per step < `e_tol`
        ------- ------ ----------------------------------------------------
        p_tol   float  Tolerance for the error estimate from the
                       Ritz Residual, stop if ``(RitzRes/gap)**2 < p_tol``
        ------- ------ ----------------------------------------------------
        N_cache int    The maximum number of `psi` to keep in memory.
        ======= ====== ====================================================

        The algorithm stops if *both* criteria for `e_tol` and `p_tol` are met
        or if the maximum number of steps was reached.

    orthogonal_to : A list of :class:`~tenpy.linalg.np_conserved.Array`
        Vectors (same tensor structure as psi) Lanczos will orthogonalize against,
        ensuring that the result is perpendicular to them.

    Returns
    -------
    E0 : float
        Ground state energy (estimate).
    psi0 : :class:`~tenpy.linalg.np_conserved.Array`
        Ground state vector (estimate).
    N : int
        Number of steps performed.
        The results are optimal in the

    Notes
    -----
    I have computed the Ritz residual (RitzRes) according to
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
    Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
    ``err < (RitzRes/gap)**2``.
    I estimate the gap from the full Lanczos spectrum.


    .. todo :
        Even the Wikipedia page contains a warning that one can quickly loose orthogonality.
        Should we include a way of Re-orthogonalization?
        At least orthogonalize against the cached states?
        (it should be much faster than applying A)
    """
    if len(orthogonal_to) > 0:
        orthogonal_to, _ = gram_schmidt(orthogonal_to)
    N_cache = get_parameter(lanczos_params, 'N_cache', 6, "Lanczos")
    if N_cache < 2:
        raise ValueError("Need to cache at least two vectors.")
    cache = []

    N_min = get_parameter(lanczos_params, 'N_min', 2, "Lanczos")
    N_max = get_parameter(lanczos_params, 'N_max', 20, "Lanczos")
    E_tol = get_parameter(lanczos_params, 'e_tol', 5.e-15, "Lanczos")
    P_tol = get_parameter(lanczos_params, 'p_tol', 0., "Lanczos")
    verbose = lanczos_params.get('verbose', 0)
    Delta_E0 = 2.
    P_err = 2.
    Es = []

    # First Lanczos iteration: Form tridiagonal form of A in the Krylov subspace, stored in T
    T = np.zeros([N_max+1, N_max+1], dtype=np.float)
    ULP = 5.e-15  # Cutoff (ULP=unit last place) to abort if beta (= norm of next v) is too small.
    # This is necessary if the rank of A is smaller than N_max - then we get a complete
    # basis of the Krylov space, and beta will be zero.
    above_ULP = True
    w = psi  # initialize
    beta = npc.norm(w)
    for k in range(N_max):
        w /= beta
        _to_cache(w, cache, N_cache)
        w = cache[-1].copy()
        # project out the orthogonal parts:
        # equivalent to using A' = P A P
        for o in orthogonal_to:  # Project out
            w -= o * npc.inner(o, w, do_conj=True)
        w = A.matvec(w)
        for o in orthogonal_to[::-1]:  # reverse: more obviously Hermitian.
            w -= o * npc.inner(o, w, do_conj=True)
        alpha = np.real(npc.inner(w, cache[-1], do_conj=True))
        T[k, k] = alpha
        if k > 0:
            w -= beta * cache[-2]
        w -= alpha * cache[-1]
        beta = npc.norm(w)
        above_ULP = abs(beta) > ULP
        if above_ULP:
            T[k, k+1] = T[k+1, k] = beta

        # Diagonalize T
        if k == 0:
            E_T = [alpha]
        else:
            E_T, v_T = np.linalg.eigh(T[0:k+1, 0:k+1])
            piv = np.argsort(E_T)  # TODO: unnecessary: eigh returns in ascending order
            assert np.all(piv == np.arange(len(piv), dtype=np.int))  # TODO: this is the check
            # E_T = E_T[piv]
            # v_T = v_T[:, piv]
            RitzRes = np.abs(v_T[k, 0] * T[k, k+1])
            Delta_E0 = (Es[-1][0] - E_T[0])
            gap = max(E_T[1] - E_T[0], 1.e-10) # TODO: magic number
            P_err = (RitzRes/gap)**2
            #print 1. - np.abs(np.inner(np.conj(v_T[0:k-1, 0]), v0_T_old))
        Es.append(E_T)
        #v0_T_old = v_T[:, 0]
        if not above_ULP or (k+1 >= N_min and (P_err < P_tol or Delta_E0 < E_tol)):
            break
    N = k + 1  # == len(Es)
    if verbose > 0:
        if verbose > 10:
            _plot_stats(Es)
        if k > 1:
            print "Lanczos", N, gap, "|", Delta_E0, E_tol, "|", P_err, P_tol
        else:
            print "Lanczos", N, alpha, beta

    if N == 1:
        return E_T[0], psi.copy(), N  # no better estimate available

    # Second Lanczos iteration.
    # Now that we know the (Ritz) eigenvector's coefficients v_T[:, 0] in the Krylov subspace,
    # construct the actual vector ``psi0 = sum_k  v_T[k, 0] vec[k]``,
    # where ``vec[k]`` is the k-th vector of the iteration.

    psi0 = psi*v_T[0, 0]  # the start vector is still known
    # and the last len(cache) vectors have been cached
    for k in range(1, len(cache) + 1):
        psi0 += v_T[N-k, 0] * cache[-k]
    len_cache = len(cache)
    del cache  # free memory: we need at least two more vectors
    # other vectors are not cached, so we need to restart the Lanczos iteration.
    q0 = None
    q1 = psi  # start vector; normalized above in place
    for k in range(0, N-len_cache-1):
        w = q1.copy()
        for o in orthogonal_to:  # Project out
            w -= o * npc.inner(o, w, do_conj=True)
        w = A.matvec(w)
        for o in orthogonal_to[::-1]:  # reverse: more obviously Hermitian.
            w -= o * npc.inner(o, w, do_conj=True)
        if k > 0:
            w -= beta * q0
        alpha = T[k, k]
        w -= alpha * q1
        beta = T[k, k+1]
        w /= beta
        q0 = q1
        q1 = w
        psi0 += q1 * v_T[k+1, 0]
    psi0_norm = npc.norm(psi0)
    if abs(1. - psi0_norm) > 1.e-3:
        warnings.warn("poorly conditioned Lanczos: |psi_0| = {0:d}".format(psi0_norm))
    psi0 /= psi0_norm
    #print "Ortho:",  #  TODO: check/test that!!!
    #for o in orthogonal_to:
    #   print np.abs(npc.inner(o, psi0, do_conj=False)),
    #print
    return E_T[0], psi0, N


def _to_cache(psi, cache, N):
    """FIFO (first in first out) cache of at most N entries."""
    cache.append(psi)
    if len(cache) > N:
        cache.pop(0)

def _plot_stats(Es):
    import matplot.pyplot as plt
    ks = np.flatten([[k]*len(E) for k, E in enumerate(Es)])
    plt.scatter(np.flatten(ks), np.flatten(Es))
    plt.xlabel("Lanczos step")
    plt.ylabel("Ritz Values (= energy estimates)")
    plt.show()
