r"""Truncation of Schmidt values.

Often, it is necessary to truncate the number of states on a virtual bond of an MPS,
keeping only the state with the largest Schmidt values.
The function :func:`truncate` picks exactly those from a given Schmidt spectrum
:math:`\lambda_a`, depending on some parameters explained in the doc-string of the function.

Further, we provide :class:`TruncationError` for a simple way to keep track of the
total truncation error.

The SVD on a virtual bond of an MPS actually gives a Schmidt decomposition
:math:`|\psi\rangle = \sum_{a} \lambda_a |L_a\rangle |R_a\rangle`
where :math:`|L_a\rangle` and :math:`|R_a\rangle` form orthonormal bases of the parts
left and right of the virtual bond.
Let us assume that the state is properly normalized,
:math:`\langle\psi | \psi\rangle = \sum_{a} \lambda^2_a = 1`.
Assume that the singular values are ordered descending, and that we keep the first :math:`\chi_c`
of the initially :math:`\chi` Schmidt values.

Then we decompose the untruncated state as
:math:`|\psi\rangle = \sqrt{1-\epsilon}|\psi_{tr}\rangle + \sqrt{\epsilon}|\psi_{tr}^\perp\rangle`
where
:math:`|\psi_{tr}\rangle =
\frac{1}{\sqrt{1-\epsilon}} \sum_{a < \chi_c} \lambda_a|L_a\rangle|R_a\rangle`
is the truncated state kept (normalized to 1),
:math:`|\psi_{tr}^\perp\rangle =
\frac{1}{\sqrt{\epsilon}} \sum_{a >= \chi_c} \lambda_a |L_a\rangle|R_a\rangle`
is the discarded part (orthogonal to the kept part) and the
*truncation error of a single truncation* is defined as
:math:`\epsilon = 1 - |\langle \psi | \psi_{tr}\rangle |^2 = \sum_{a >= \chi_c} \lambda_a^2`.

.. warning ::
    For imaginary time evolution (e.g. with TEBD), you try to project out the ground state.
    Then, looking at the truncation error defined in this module does *not* give you any
    information how good the found state coincides with the actual ground state!
    (Instead, the returned truncation error depends on the overlap with the initial state,
    which is arbitrary > 0)

.. warning ::
    This module takes only track of the errors coming from the truncation of Schmidt values.
    There might be other sources of error as well, for example TEBD has also an discretization
    error depending on the chosen time step.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
from ..linalg import np_conserved as npc
from ..tools.hdf5_io import Hdf5Exportable
import warnings
from ..tools.params import asConfig

__all__ = ['TruncationError', 'truncate', 'svd_theta', 'decompose_theta_qr_based']



class TruncationError(Hdf5Exportable):
    r"""Class representing a truncation error.

    The default initialization represents "no truncation".

    .. warning ::
        For imaginary time evolution, this is *not* the error you are interested in!

    Parameters
    ----------
    eps, ov : float
        See below.


    Attributes
    ----------
    eps : float
        The total sum of all discarded Schmidt values squared.
        Note that if you keep singular values up to 1.e-14 (= a bit more than machine precision
        for 64bit floats), `eps` is on the order of 1.e-28 (due to the square)!
    ov : float
        A lower bound for the overlap :math:`|\langle \psi_{trunc} | \psi_{correct} \rangle|^2`
        (assuming normalization of both states).
        This is probably the quantity you are actually interested in.
        Takes into account the factor 2 explained in the section on Errors in the
        `TEBD Wikipedia article <https://en.wikipedia.org/wiki/Time-evolving_block_decimation>`.
    """
    def __init__(self, eps=0., ov=1.):
        self.eps = eps
        self.ov = ov

    def copy(self):
        """Return a copy of self."""
        return TruncationError(self.eps, self.ov)

    @classmethod
    def from_norm(cls, norm_new, norm_old=1.):
        r"""Construct TruncationError from norm after and before the truncation.

        Parameters
        ----------
        norm_new : float
            Norm of Schmidt values kept, :math:`\sqrt{\sum_{a kept} \lambda_a^2}`
            (before re-normalization).
        norm_old : float
            Norm of all Schmidt values before truncation, :math:`\sqrt{\sum_{a} \lambda_a^2}`.
        """
        eps = 1. - norm_new**2 / norm_old**2  # = (norm_old**2 - norm_new**2)/norm_old**2
        return cls(eps, 1. - 2. * eps)

    @classmethod
    def from_S(cls, S_discarded, norm_old=None):
        r"""Construct TruncationError from discarded singular values.

        Parameters
        ----------
        S_discarded : 1D numpy array
            The singular values discarded.
        norm_old : float
            Norm of all Schmidt values before truncation, :math:`\sqrt{\sum_{a} \lambda_a^2}`.
            Default (``None``) is 1.
        """
        eps = np.sum(np.square(S_discarded))
        if norm_old:
            eps /= norm_old * norm_old
        return cls(eps, 1. - 2. * eps)

    def __add__(self, other):
        res = TruncationError()
        res.eps = self.eps + other.eps  # whatever that actually means...
        res.ov = self.ov * other.ov
        return res

    @property
    def ov_err(self):
        """Error ``1.-ov`` of the overlap with the correct state."""
        return 1. - self.ov

    def __repr__(self):
        if self.eps != 0 or self.ov != 1.:
            return "TruncationError(eps={eps:.4e}, ov={ov:.10f})".format(eps=self.eps, ov=self.ov)
        else:
            return "TruncationError()"


def truncate(S, options):
    """Given a Schmidt spectrum `S`, determine which values to keep.

    Options
    -------
    .. cfg:config:: truncation

        chi_max : int
            Keep at most `chi_max` Schmidt values.
        chi_min : int
            Keep at least `chi_min` Schmidt values.
        degeneracy_tol: float
            Don't cut between neighboring Schmidt values with
            ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
            ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``
            for small `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            Schmidt values are degenerate with a relative error smaller
            than `degeneracy_tol`, which we expect to happen in the case
            of symmetries.
        svd_min : float
            Discard all small Schmidt values ``S[i] < svd_min``.
        trunc_cut : float
            Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``.

    Parameters
    ----------
    S : 1D array
        Schmidt values (as returned by an SVD), not necessarily sorted.
        Should be normalized to ``np.sum(S*S) == 1.``.
    options: dict-like
        Config with constraints for the truncation, see :cfg:config:`truncation`.
        If a constraint can not be fulfilled (without violating a previous one), it is ignored.
        A value ``None`` indicates that the constraint should be ignored.

    Returns
    -------
    mask : 1D bool array
        Index mask, True for indices which should be kept.
    norm_new : float
        The norm of the truncated Schmidt values, ``np.linalg.norm(S[mask])``.
        Useful for re-normalization.
    err : :class:`TruncationError`
        The error of the represented state which is introduced due to the truncation.
    """
    options = asConfig(options, "truncation")
    # by default, only truncate values which are much closer to zero than machine precision.
    # This is only to avoid problems with taking the inverse of `S`.
    chi_max = options.get('chi_max', 100, int)
    chi_min = options.get('chi_min', None, int)
    deg_tol = options.get('degeneracy_tol', None, 'real')
    svd_min = options.get('svd_min', 1.e-14, 'real')
    trunc_cut = options.get('trunc_cut', 1.e-14, 'real')

    if trunc_cut is not None and trunc_cut >= 1.:
        raise ValueError("trunc_cut >=1.")
    if not np.any(S > 1.e-10):
        warnings.warn("no Schmidt value above 1.e-10", stacklevel=2)
    if np.any(S < -1.e-10):
        warnings.warn("negative Schmidt values!", stacklevel=2)

    # use 1.e-100 as replacement for <=0 values for a well-defined logarithm.
    logS = np.log(np.choose(S <= 0., [S, 1.e-100 * np.ones(len(S))]))
    piv = np.argsort(logS)  # sort *ascending*.
    logS = logS[piv]
    # goal: find an index 'cut' such that we keep piv[cut:], i.e. cut between `cut-1` and `cut`.
    good = np.ones(len(piv), dtype=np.bool_)  # good[cut] = (is `cut` a good choice?)
    # we choose the smallest 'good' cut.

    if chi_max is not None:
        # keep at most chi_max values
        good2 = np.zeros(len(piv), dtype=np.bool_)
        good2[-chi_max:] = True
        good = _combine_constraints(good, good2, "chi_max")

    if chi_min is not None and chi_min > 1:
        # keep at most chi_max values
        good2 = np.ones(len(piv), dtype=np.bool_)
        good2[-chi_min + 1:] = False
        good = _combine_constraints(good, good2, "chi_min")

    if deg_tol:
        # don't cut between values (cut-1, cut) with ``log(S[cut]/S[cut-1]) < deg_tol``
        # this is equivalent to
        # ``(S[cut] - S[cut-1])/S[cut-1] < exp(deg_tol) - 1 = deg_tol + O(deg_tol^2)``
        good2 = np.empty(len(piv), np.bool_)
        good2[0] = True
        good2[1:] = np.greater_equal(logS[1:] - logS[:-1], deg_tol)
        good = _combine_constraints(good, good2, "degeneracy_tol")

    if svd_min is not None:
        # keep only values S[i] >= svd_min
        good2 = np.greater_equal(logS, np.log(svd_min))
        good = _combine_constraints(good, good2, "svd_min")

    if trunc_cut is not None:
        good2 = (np.cumsum(S[piv]**2) > trunc_cut * trunc_cut)
        good = _combine_constraints(good, good2, "trunc_cut")

    cut = np.nonzero(good)[0][0]  # smallest possible cut: keep as many S as allowed
    mask = np.zeros(len(S), dtype=np.bool_)
    np.put(mask, piv[cut:], True)
    norm_new = np.linalg.norm(S[mask])
    return mask, norm_new, TruncationError.from_S(S[np.logical_not(mask)]),


def svd_theta(theta, trunc_par, qtotal_LR=[None, None], inner_labels=['vR', 'vL']):
    """Performs SVD of a matrix `theta` (= the wavefunction) and truncates it.

    Perform a singular value decomposition (SVD) with :func:`~tenpy.linalg.np_conserved.svd`
    and truncates with :func:`truncate`.
    The result is an approximation
    ``theta ~= tensordot(U.scale_axis(S*renormalization, 1), VH, axes=1)``

    Parameters
    ----------
    theta : :class:`~tenpy.linalg.np_conserved.Array`, shape ``(M, N)``
        The matrix, on which the singular value decomposition (SVD) is performed.
        Usually, `theta` represents the wavefunction, such that the SVD is a Schmidt decomposition.
    trunc_par : dict
        truncation parameters as described in :func:`truncate`.
    qtotalLR : (charges, charges)
        The total charges for the returned `U` and `VH`.
    inner_labels : (string, string)
        Labels for the `U` and `VH` on the newly-created bond.

    Returns
    -------
    U : :class:`~tenpy.linalg.np_conserved.Array`
        Matrix with left singular vectors as columns.
        Shape ``(M, M)`` or ``(M, K)`` depending on `full_matrices`.
    S : 1D ndarray
        The singular values of the array.
        If no `cutoff` is given, it has length ``min(M, N)``.
        Normalized to ``np.linalg.norm(S)==1``.
    VH : :class:`~tenpy.linalg.np_conserved.Array`
        Matrix with right singular vectors as rows.
        Shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.
    err : :class:`TruncationError`
        The truncation error introduced.
    renormalization : float
        Factor, by which S was renormalized.
    """
    U, S, VH = npc.svd(theta,
                       full_matrices=False,
                       compute_uv=True,
                       qtotal_LR=qtotal_LR,
                       inner_labels=inner_labels)
    renormalization = np.linalg.norm(S)
    S = S / renormalization
    piv, new_norm, err = truncate(S, trunc_par)
    new_len_S = np.sum(piv, dtype=np.int_)
    if new_len_S * 100 < len(S) and (trunc_par['chi_max'] is None
                                     or new_len_S != trunc_par['chi_max']):
        msg = "Catastrophic reduction in chi: {0:d} -> {1:d}".format(len(S), new_len_S)
        # NANs are excluded in npc.svd
        UHU = npc.tensordot(U.conj(), U, axes=[[0], [0]])
        msg += " |U^d U - 1| = {0:f}".format(npc.norm(UHU - npc.eye_like(UHU)))
        VHV = npc.tensordot(VH, VH.conj(), axes=[[1], [1]])
        msg += " |V V - 1| = {0:f}".format(npc.norm(VHV - npc.eye_like(VHV)))
        warnings.warn(msg, stacklevel=2)
    S = S[piv] / new_norm
    renormalization *= new_norm
    U.iproject(piv, axes=1)  # U = U[:, piv]
    VH.iproject(piv, axes=0)  # VH = VH[piv, :]
    return U, S, VH, err, renormalization


def _qr_theta_Y0(B_L: npc.Array, B_R: npc.Array, theta: npc.Array, expand: float, min_block_increase: int):
    """Generate the initial guess Y0 for the right isometry for the QR based theta decomposition decompose_theta_qr_based().

    Parameters
    ----------
    B_L : Array with legs [vL, p, vR]
    B_R : Array with legs [vL, p, vR]
    theta : Array with legs [(vL.p0), (p1.vR)]
    expand : float or None

    Returns
    -------
    Y0 : Array with legs [vL, (p1.vR)]
    """
    if expand is None or expand == 0:
        return B_R.combine_legs(['p', 'vR']).ireplace_labels('(p.vR)', '(p1.vR)')

    assert min_block_increase >= 0

    Y0 = theta.copy(deep=False)
    Y0.legs[0] = Y0.legs[0].to_LegCharge()
    Y0.ireplace_label('(vL.p0)', 'vL')
    if any(B_L.qtotal != 0):
        Y0.gauge_total_charge('vL', new_qtotal=B_R.qtotal)
    vL_old = B_R.get_leg('vL')
    if not vL_old.is_blocked():
        vL_old = vL_old.sort()[1]
    vL_new = Y0.get_leg('vL')  # is blocked, since created from pipe

    # vL_old is guaranteed to be a slice of vL_new by charge rule in B_L
    piv = np.zeros(vL_new.ind_len, dtype=bool)  # indices to keep in vL_new
    increase_per_block = max(min_block_increase, int(vL_old.ind_len * expand // vL_new.block_number))
    sizes_old = vL_old.get_block_sizes()
    sizes_new = vL_new.get_block_sizes()
    # iterate over charge blocks in vL_new and vL_old at the same time
    j_old = 0
    q_old = vL_old.charges[j_old, :]
    qdata_order = np.argsort(Y0._qdata[:, 0])
    qdata_idx = 0
    for j_new, q_new in enumerate(vL_new.charges):
        if all(q_new == q_old):  # have charge block in both vL_new and vL_old
            s_new = sizes_old[j_old] + increase_per_block
            # move to next charge block in next loop iteration
            j_old += 1
            if j_old < len(vL_old.charges):
                q_old = vL_old.charges[j_old, :]
        else:  # charge block only in vL_new
            s_new = increase_per_block
        s_new = min(s_new, sizes_new[j_new])  # don't go beyond block

        if Y0._qdata[qdata_order[qdata_idx], 0] != j_new:
            # block does not exist
            # while we could set corresponding piv entries to True, it would not help, since
            # the corresponding "entries" of Y0 are zero anyway
            continue

        # block has axis [vL, (p1.vR)]. want to keep the s_new slices of the vL axis
        #  that have the largest norm
        norms = np.linalg.norm(Y0._data[qdata_order[qdata_idx]], axis=1)
        kept_slices = np.argsort(-norms)[:s_new]  # negative sign so we sort large to small
        start = vL_new.slices[j_new]
        piv[start + kept_slices] = True

        qdata_idx += 1
        if qdata_idx >= Y0._qdata.shape[0]:
            break

    Y0.iproject(piv, 'vL')
    return Y0

def decompose_theta_qr_based(B_L: npc.Array, B_R: npc.Array, theta: npc.Array, trunc_params: dict, 
             expand: float, use_eig_based_svd: bool, need_A_L: bool, 
             compute_err: bool, min_block_increase: int):
    """Performs a QR based decomposition of a matrix `theta` (= the wavefunction) and truncates it

    Parameters
    ----------
    B_L : Array with legs [vL, p, vR]
    B_R : Array with legs [vL, p, vR]
    theta : Array with legs [(vL.p0), (p1.vR)]
    trunc_par : dict
        truncation parameters as described in :func:`truncate`.
    ...

    Returns
    -------
    A_L : array with legs [(vL.p), vR] or None
    S : 1D numpy array
    B_R : array with legs [vL, (p.vR)]
    trunc_err : TruncationError
    renormalize : float
    """
    # TODO: Do we need to implement a parameter inner_labels as for the svd method?

    if compute_err:
        need_A_L = True

    # Get inital guess for the right isometry
    Y0 = _qr_theta_Y0(B_L, B_R, theta, expand, min_block_increase) # Y0: [vL, (p1.vR)]

    # QR based updates
    theta_i0 = npc.tensordot(theta, Y0.conj(), ['(p1.vR)', '(p1*.vR*)']).ireplace_label('vL*', 'vR')
    A_L, _ = npc.qr(theta_i0, inner_labels=['vR', 'vL']) # A_L: [(vL.p0), vR]

    theta_i1 = npc.tensordot(A_L.conj(), theta, ['(vL*.p0*)', '(vL.p0)']).ireplace_label('vR*', 'vL')
    theta_i1.itranspose(['(p1.vR)', 'vL'])
    B_R, Xi = npc.qr(theta_i1, inner_labels=['vL', 'vR'], inner_qconj=-1)
    B_R.itranspose(['vL', '(p1.vR)'])
    Xi.itranspose(['vL', 'vR'])

    # SVD of bond matrix Xi
    if use_eig_based_svd:
        # TODO: Implement eigendecomposition based svd (could be copied from tebd.py but leave it for now)
        msg = "decompose_theta_qr_based() is not implemented with eigendecomposition based svd. Please use implemente method by setting parameteruse_eig_based_svd=False"
        raise NotImplementedError(msg)
    else:
        U, S, Vd, _, renormalize = svd_theta(Xi, trunc_params)
    B_R = npc.tensordot(Vd, B_R, ['vR', 'vL'])
    if need_A_L:
        A_L = npc.tensordot(A_L, U, ['vR', 'vL'])
    else:
        A_L = None

    if compute_err:
        theta_approx = npc.tensordot(A_L.scale_axis(S, axis='vR'), B_R, ['vR', 'vL'])
        eps = npc.norm(theta - theta_approx) ** 2
        trunc_err = TruncationError(eps, 1. - 2. * eps)
    else:
        trunc_err = TruncationError(np.nan, np.nan)

    B_R = B_R.ireplace_label('(p1.vR)', '(p.vR)')
    if need_A_L:
        A_L = A_L.ireplace_label('(vL.p0)', '(vL.p)')

    return A_L, S, B_R, trunc_err, renormalize

def _combine_constraints(good1, good2, warn):
    """return logical_and(good1, good2) if there remains at least one `True` entry.

    Otherwise print a warning and return just `good1`.
    """
    res = np.logical_and(good1, good2)
    if np.any(res):
        return res
    warnings.warn("truncation: can't satisfy constraint for " + warn, stacklevel=3)
    return good1


# truncation parameter for truncating svd values at machine precision
# excluding 0. and negative S values only
_machine_prec_trunc_par = asConfig({'svd_min': np.finfo(np.float64).eps,
                                    'trunc_cut': None,
                                    'chi_max': None},
                                   'machine_prec_trunc_params')
_machine_prec_trunc_par.unused.clear()
