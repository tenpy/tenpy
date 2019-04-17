r"""Truncation of Schmidt values.

Often, it is necessary to truncate the number of states on a virtual bond of an MPS,
keeping only the state with the largest Schmidt values.
The function :func:`truncation` picks exactly those from a given Schmidt spectrum
:math:`\lambda_a`, depending on some parameters explained in the doc-string of the function.

Further, we provide :class:`TruncationError` for a simple way to keep track of the
total truncation error.

The SVD on a virtual bond of an MPS actually gives a Schmidt decomposition
:math:`|\psi\rangle = \sum_{a} \lambda_a |L_a\rangle |R_a\rangle`
where :math:`|L_a\rangle` and :math:`|R_a\rangle` form orthonormal bases of the parts
left and right of the virtual bond.
Let us assume that the state is properly normalized,
:math:`\langle\psi | \psi\rangle = \sum_{a} \lambda^2 = 1`.
Assume that the singular values are ordered descending, and that we keep the first :math:`\chi_c`
of the initially :math:`\chi` Schmidt values.

Then we decompose the untuncated state as
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
    There might be other sources of error as well, for example TEBD has also an discretisation
    error depending on the chosen time step.

.. todo ::
    The `TEBD wikipedia article <https://en.wikipedia.org/wiki/Time-evolving_block_decimation>`_
    (in the section 'Errors coming from the truncation of the Hilbert space')
    claims that there is a second more subtle error, which stems from the change of the Schmidt
    basis :math:`|R_a\rangle` on bond i-1 if we truncate bond i.
    In the end, that leads just to a factor of 2 in TruncationError.__init__ ???
    (I couldn't follow the argument completely,
    and the factor was definetly not included in the old TenPy.)
"""
# Copyright 2018 TeNPy Developers

import numpy as np
from ..linalg import np_conserved as npc
import warnings
from ..tools.params import get_parameter

__all__ = ['TruncationError', 'truncate', 'svd_theta']


class TruncationError:
    r"""Class representing a truncation error.

    The default initialization represents "no truncation".

    .. warning ::
        For imaginary time evolution, this is *not* the error you are interested in!

    Attributes
    ----------
    ov_err
    eps : float
        The total sum of all discared Schmidt values squared.
    ov : float
        A lower bound for the overlap :math:`|\langle \psi_{trunc} | \psi_{correct} \rangle|^2`
        (assuming normalization of both states).
        This is probably the quantity you are actually interested in.

    Examples
    --------
    >>> TE = TruncationError()
    >>> TE += tebd.time_evolution(...)

    """

    def __init__(self, eps=0., ov=1.):
        self.eps = eps
        self.ov = ov

    def copy(self):
        """Return a copy of self."""
        return TruncationError(self.eps, self.ov)

    @classmethod
    def from_norm(cls, norm_new, norm_old=1.):
        """Construct TruncationError from norm after and before the truncation.

        Parameters
        ----------
        norm_new : float
            Norm of Schmidt values kept, :math:`\sqrt{\sum_{a kept} \lambda_a^2}`
            (before re-normalization).
        norm_old : float
            Norm of all Schmidt values before truncation, :math:`\sqrt{\sum_{a} \lambda_a^2}`.
        """
        res = cls()
        res.eps = 1. - norm_new**2 / norm_old**2  # = (norm_old**2 - norm_new**2)/norm_old**2
        res.ov = 1. - 2. * res.eps  # TODO: include factor of 2? See above link to wikipedia
        return res

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


def truncate(S, trunc_par):
    """Given a Schmidt spectrum Y, determine which values to keep.

    Parameters
    ----------
    S : 1D array
        Schmidt values (as returned by an SVD).
    trunc_par: dict
        Parameters giving constraints for the truncation.
        If a constraint can not be fullfilled (without violating a previous one), it is ignored.
        A value ``None`` indicates that the constraint should be ignored.

        ============ ====== ====================================================
        key          type   constraint
        ============ ====== ====================================================
        chi_max      int    Keep at most `chi_max` Schmidt values.
        ------------ ------ ----------------------------------------------------
        chi_min      int    Keep at least `chi_min` Schmidt values.
        ------------ ------ ----------------------------------------------------
        symmetry_tol float  Don't cut between Schmidt values with
                            ``|log(S[i]/S[j])| < log(symmetry_tol)``
                            (i.e. either keep either both `i` and `j` or none).
                            This is useful to prevent discarding (nearly)
                            degenerate pairs in case of symmetries.
        ------------ ------ ----------------------------------------------------
        svd_min      float  Discard all small Schmidt values ``S[i] < svd_min``.
        ------------ ------ ----------------------------------------------------
        trunc_cut    float  Discard all small Schmidt values as long as
                            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``.
        ============ ====== ====================================================

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
    # by default, only truncate values which are much closer to zero than machine precision.
    # This is only to avoid problems with taking the inverse of `S`.
    chi_max = get_parameter(trunc_par, 'chi_max', 100, 'truncation')
    chi_min = get_parameter(trunc_par, 'chi_min', None, 'truncation')
    sym_tol = get_parameter(trunc_par, 'symmetry_tol', None, 'truncation')
    svd_min = get_parameter(trunc_par, 'svd_min', 1.e-14, 'truncation')
    trunc_cut = get_parameter(trunc_par, 'trunc_cut', 1.e-14, 'truncation')

    if trunc_cut is not None and trunc_cut >= 1.:
        raise ValueError("trunc_cut >=1.")
    if not np.any(S > 1.e-10):
        warnings.warn("no Schmidt value above 1.e-10", stacklevel=2)
    if np.any(S < -1.e-10):
        warnings.warn("negative Schmidt values!", stacklevel=2)

    # use 1.e-100 as replacement for <=0 values for a well-defined logarithm.
    logS = np.log(np.choose(S <= 0., [S, 1.e-100 * np.ones(len(S))]))
    piv = np.argsort(logS)  # sort *ascending*.
    # goal: find an index 'cut' such that we keep piv[cut:].
    logS = logS[piv]
    good = np.ones(len(piv), dtype=np.bool)  # good[cut] = (is `cut` a good choice?)
    # we choose the smalles 'good' cut.

    if chi_max is not None:
        # keep at most chi_max values
        good2 = np.zeros(len(piv), dtype=np.bool)
        good2[-chi_max:] = True
        good = _combine_constraints(good, good2, "chi_max")

    if chi_min is not None and chi_min > 1:
        # keep at most chi_max values
        good2 = np.ones(len(piv), dtype=np.bool)
        good2[-chi_min + 1:] = False
        good = _combine_constraints(good, good2, "chi_min")

    if sym_tol:
        # don't cut between values with log(S[i]/S[j]) < log(sym_tol)
        good2 = np.empty(len(piv), np.bool)
        good2[0] = True
        good2[1:] = np.greater_equal(logS[1:] - logS[:-1], np.log(sym_tol))
        good = _combine_constraints(good, good2, "symmetry_tol")

    if svd_min is not None:
        # keep only values S[i] >= svd_min
        good2 = np.greater_equal(logS, np.log(svd_min))
        good = _combine_constraints(good, good2, "svd_min")

    if trunc_cut is not None:
        good2 = (np.cumsum(S[piv]**2) > trunc_cut * trunc_cut)
        good = _combine_constraints(good, good2, "trunc_cut")

    cut = np.nonzero(good)[0][0]  # smallest possible cut: keep as many S as allowed
    mask = np.zeros(len(S), dtype=np.bool)
    np.put(mask, piv[cut:], True)
    norm_new = np.linalg.norm(S[mask])
    return mask, norm_new, TruncationError.from_norm(norm_new, np.linalg.norm(S)),


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
        The singluar values of the array.
        If no `cutoff` is given, it has lenght ``min(M, N)``.
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


def _combine_constraints(good1, good2, warn):
    """return logical_and(good1, good2) if there remains at least one `True` entry.
    Otherwise print a warning and return just `good1`."""
    res = np.logical_and(good1, good2)
    if np.any(res):
        return res
    warnings.warn("truncation: can't satisfy constraint for " + warn, stacklevel=3)
    return good1
