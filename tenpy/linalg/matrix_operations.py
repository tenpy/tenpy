# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
import warnings
from numbers import Number
from .tensors import DiagonalTensor, AbstractTensor, Tensor, Mask
from ..tools.misc import inverse_permutation
from ..tools.params import asConfig
from ..tools.hdf5_io import Hdf5Exportable

__all__ = ['svd', 'truncate_svd', 'svd_split', 'leg_bipartition', 'exp', 'log']


def svd(a: AbstractTensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
        new_labels: tuple[str, ...] = None, new_vh_leg_dual=False,
        options={}
        ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor]:
    """Singular value decomposition of a tensor.

    The tensor is viewed as a linear map (i.e. matrix) from one set of its legs to the rest.

    TODO: document input format for u_legs / vh_legs, can probably to it centrally for all matrix ops

    TODO: document efficiency issues when combining legs

    Parameters
    ----------
    a : Tensor
    u_legs : list[int  |  str], optional
        Which of the legs belong to "matrix rows" and end up as legs of U.
    vh_legs : list[int  |  str], optional
        Which of the legs belong to "matrix columns" and end up as legs of Vh.
    new_labels : tuple[str, ...], optional
        Specification for the labels of the newly generated leg, i.e. those legs that would need to
        be contracted to recover `a`.
        If two `str`s, the first is used for "right" legs (that of U and the second of S)
                      the second is used for "left" legs (that of Vh and the first of S)
        If four `str`s, all the above are specified, in the order U, first of S, second of S, Vh
        If None (default), the new legs will not be labelled
    new_vh_leg_dual : bool
        Whether the new leg of `vh` will be dual or not.
        TODO what is the intuitive default?
    options : dict | Config
        TODO proper Config style doc
        algorithm : str | None

    Returns
    -------
    U, S, Vh
        The tensors U, S, Vh that form the SVD, such that `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is equal
        to `a` up to numerical precision
    """
    if not isinstance(a, Tensor):
        raise NotImplementedError  # TODO ChargedTensor support

    u_idcs, vh_idcs = leg_bipartition(a, u_legs, vh_legs)
    l_u, l_su, l_sv, l_vh = _svd_new_labels(new_labels)

    need_combine = (len(u_idcs) != 1 or len(vh_idcs) != 1)
    original_labels = a.labels
    if need_combine:
        a = a.combine_legs(u_idcs, vh_idcs, new_axes=[0, 1])
    elif u_idcs[0] == 1:   # both single entry, so v_idcs = [1]
        a = a.permute_legs([1, 0])

    options = asConfig(options, 'SVD')  # TODO if algorithm remains the only key, consider just making it a kwarg
    algorithm = options.get('algorithm', None)
    u_data, s_data, vh_data, new_leg = a.backend.svd(a, new_vh_leg_dual, algorithm=algorithm)

    U = Tensor(u_data, backend=a.backend, legs=[a.legs[0], new_leg.dual])
    S = DiagonalTensor(s_data, first_leg=new_leg, second_leg_dual=True, backend=a.backend, labels=[l_su, l_sv])
    Vh = Tensor(vh_data, backend=a.backend, legs=[new_leg, a.legs[1]])
    if need_combine:
        U = U.split_legs(0)
        Vh = Vh.split_legs(1)
    U.set_labels([original_labels[n] for n in u_idcs] + [l_u])
    Vh.set_labels([l_vh] + [original_labels[n] for n in vh_idcs])
    return U, S, Vh


def truncated_svd(a: AbstractTensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
                  new_labels: tuple[str, ...] = None, new_vh_leg_dual=False,
                  options={}, truncation_options={}
                  ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor]:
    """Truncated singular value decomposition of a tensor.

    The tensor is viewed as a linear map (i.e. matrix) from one set of its legs to the rest.

    Parameters
    ----------
    same as for :meth:`svd`
    truncation_options : dict-like
        Options that determine how many singular values are kept, see :cfg:config:`truncation`.

    Returns
    -------
    U, S, Vh
        The tensors U, S, Vh that form the truncated SVD, such that
        `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is *aproximately* equal to `a`.
        The singular values are renormalized to ``S.norm() == a.norm()``.
    err : :class:`TruncationError`
        the truncation error introduced
    renormalization : float
        Factor, by which `S` was renormalized.
    """
    U, S, V = svd(a, u_legs=u_legs, vh_legs=vh_legs, new_labels=new_labels, new_vh_leg_dual=new_vh_leg_dual,
                  options=options)
    S_norm = S.norm()
    mask, renormalization, err = truncate_singular_values(S / S_norm, options=truncation_options)
    U = U.apply_mask(mask, -1)
    S = S._apply_mask_both_legs(mask)._mul_scalar(1. / renormalization)
    # TODO (JU): unlike svd_theta this does not normalize to 1 but to norm(a).
    #            could introduce an argument, `normalize_to: float | None`, where None means norm(a)
    V = V.apply_mask(mask, 0)
    return U, S, V, err, renormalization
    

def truncate_singular_values(S: DiagonalTensor, options) -> Mask:
    """Given singular values, determine which to keep.

    Options
    -------
    .. cfg:config:: truncation

        chi_max : int
            Keep at most `chi_max` singular values.
        chi_min : int
            Keep at least `chi_min` singular values
        degeneracy_tol : float
            Don't cut between neighboring singular values with
            ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
            ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``
            for small `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            Schmidt values are degenerate with a relative error smaller
            than `degeneracy_tol`, which we expect to happen in the case
            of symmetries.
        svd_min : float
            Discard all small singular values ``S[i] < svd_min``.
        trunc_cut : float
            Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``.

    Parameters
    ----------
    S : DiagonalTensor
        Singular values, normalized to ``S.norm() == 1``.
    options : dict-like
        Config with constraints for the truncation, see :cfg:config:`truncation`.
        If a constraint can not be fullfilled (without violating a previous one), it is ignored.
        A value ``None`` indicates that the constraint should be ignored.

    Returns
    -------
    mask : Mask
        A mask, indicating which of the singular values to keep
    new_norm : float
        The norm of S after truncation
    err : :class:`TruncationError`
        the truncation error introduced
    """
    options = asConfig(options, "truncation")
    # by default, only truncate values which are much closer to zero than machine precision.
    # This is only to avoid problems with taking the inverse of `S`.
    chi_max = options.get('chi_max', 100)
    chi_min = options.get('chi_min', None)
    deg_tol = options.get('degeneracy_tol', None)
    svd_min = options.get('svd_min', 1.e-14)
    trunc_cut = options.get('trunc_cut', 1.e-14)

    # OPTIMIZE should we do all of this logic with block-backend instead of numpy?
    S_block = S.to_numpy_ndarray()

    if trunc_cut is not None and trunc_cut >= 1.:
        raise ValueError("trunc_cut >=1.")
    if S_block.backend.block_sum_all(S_block > 1.e-10) == 0:
        warnings.warn("no Schmidt value above 1.e-10", stacklevel=2)
    if S_block.backend.block_sum_all(S_block < 1.e-10) > 0:
        warnings.warn("negative Schmidt values!", stacklevel=2)

    # use 1.e-100 as replacement for <=0 values for a well-defined logarithm.
    logS = np.log(np.choose(S_block <= 0., [S_block, 1.e-100 * np.ones(len(S_block))]))
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
        good2 = (np.cumsum(S_block[piv]**2) > trunc_cut * trunc_cut)
        good = _combine_constraints(good, good2, "trunc_cut")

    cut = np.nonzero(good)[0][0]  # smallest possible cut: keep as many S as allowed
    mask = np.zeros(len(S_block), dtype=np.bool_)
    np.put(mask, piv[cut:], True)
    new_norm = np.linalg.norm(S_block[mask])
    err = TruncationError.from_S(S[np.logical_not(mask)])
    
    return Mask.from_flat_numpy(mask, large_leg=S.legs[0]), new_norm, err


def qr(a: AbstractTensor, q_legs: list[int | str] = None, r_legs: list[int | str] = None,
       new_labels: tuple[str, str] = [None, None], new_r_leg_dual: bool = False,
       full: bool = False):
    """QR decomposition of a tensor, viewed as a linear map.

    Parameters
    ----------
    a : Tensor
        The tensor to decompose
    q_legs :
        Which of the legs belong to "matrix rows" and end up as legs of Q.
    r_legs :
        Which of the legs belong to "matrix columns" and end up as legs of R.
    new_labels : tuple[str, str], optional
        Labels for the new legs on Q and R.
    new_r_leg_dual: bool
        Whether the new leg on R will be dual or not. This is purely conventional.
        TODO what is the intuitive default? -> match convention with SVD
    full: bool
        Whether the full QR decomposition should be computed
    """
    if not isinstance(a, Tensor):
        raise NotImplementedError  # TODO ChargedTensor support

    a_labels = a.labels
    q_idcs, r_idcs = leg_bipartition(a, q_legs, r_legs)

    need_combine = (len(q_idcs) > 1 or len(r_idcs) > 1)
    if need_combine:
        a = a.combine_legs(q_idcs, r_idcs, new_axes=[0, 1])
    elif q_idcs[0] == 1:  # this implies q_idcs == [1] and r_idcs == [0]
        a = a.permute_legs([1, 0])

    q_data, r_data, new_leg = a.backend.qr(a, new_r_leg_dual, full=full)

    Q = Tensor(q_data, legs=[a.legs[0], new_leg.dual], backend=a.backend)
    R = Tensor(r_data, legs=[new_leg, a.legs[1]], backend=a.backend)
    if need_combine:
        R = R.split_leg(1)
        Q = Q.split_leg(0)
    Q.set_labels([a_labels[n] for n in q_idcs] + [new_labels[0]])
    R.set_labels([new_labels[1]], [a_labels[n] for n in r_idcs])
    return Q, R


def svd_split(a: AbstractTensor, legs1: list[int | str] = None, legs2: list[int | str] = None,
              new_labels: tuple[str, str] = None, options=None, s_exponent: float = .5):
    """Split a tensor via (truncated) svd,
    i.e. compute (U @ S ** s_exponent) and (S ** (1 - s_exponent) @ Vh)"""
    raise NotImplementedError  # TODO


def _combine_constraints(good1, good2, warn):
    """return logical_and(good1, good2) if there remains at least one `True` entry.

    Otherwise print a warning and return just `good1`.
    """
    res = np.logical_and(good1, good2)
    if np.any(res):
        return res
    warnings.warn("truncation: can't satisfy constraint for " + warn, stacklevel=3)
    return good1


def _svd_new_labels(new_labels: tuple[str, ...] | None) -> tuple[str, str, str, str]:
    if new_labels is None:
        l_u = l_su = l_sv = l_vh = None
    else:
        try:
            num_labels = len(new_labels)
        except AttributeError:
            raise ValueError('new_labels must be a sequence')
        if num_labels == 2:
            l_u = l_sv = new_labels[0]
            l_su = l_vh = new_labels[1]
        elif num_labels == 4:
            l_u, l_su, l_sv, l_vh = new_labels
        else:
            raise ValueError('expected 2 or 4 new_labels')
    return l_u, l_su, l_sv, l_vh


def leg_bipartition(a: AbstractTensor, legs1: list[int | str] | None, legs2: list[int | str] | None
                    ) -> tuple[list[int] | list[int]]:
    """Utility function for partitioning the legs of a Tensor into two groups.
    The tensor can then unambiguously be understood as a linear map, and linear algebra concepts
    such as SVD, eigensystems, ... make sense.

    The lists `legs1` and `legs2` can bother either be a list, describing via indices (`int`)
    or via labels (`str`) a subset of the legs of `a`, or they can be `None`.
    - if both are `None`, use the default `legs1=[0]` and `legs2=[1]`, which requires `a` to be a
      two-leg tensor, which we understand as a matrix
    - if exactly one is `None`, it implies "all remaining legs",
      i.e. those legs not contained in the other list
    - if both are lists, a `ValueError` is raised, if they are not a bipartition of all legs of `a`

    Returns a list of indices for both groups
    """

    if legs1 is None and legs2 is None:
        if a.num_legs != 2:
            raise ValueError('Expected a two-leg tensor')
        idcs1 = [0]
        idcs2 = [1]
    elif legs1 is None:
        idcs2 = a.get_leg_idcs(legs2)
        idcs1 = [n for n in range(a.num_legs) if n not in idcs2]
    elif legs2 is None:
        idcs1 = a.get_leg_idcs(legs1)
        idcs2 = [n for n in range(a.num_legs) if n not in idcs1]
    else:
        idcs1 = a.get_leg_idcs(legs1)
        idcs2 = a.get_leg_idcs(legs2)
        in_both_lists = [l for l in idcs1 if l in idcs2]
        if in_both_lists:
            raise ValueError('leg groups must be disjoint')
        missing = [n for n in range(a.num_legs) if n not in idcs1 and n not in idcs2]
        if missing:
            raise ValueError('leg groups together must contain all legs')
    return idcs1, idcs2


def exp(t: AbstractTensor | complex | float, legs1: list[int | str] = None,
        legs2: list[int | str] = None) -> AbstractTensor | complex | float:
    """
    The exponential of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise exponential function.
    """
    if isinstance(t, Tensor):
        return _act_block_diagonal_square_matrix(t, legs1, legs2, 'matrix_exp')
    if isinstance(t, DiagonalTensor):
        raise NotImplementedError  # TODO
    if isinstance(t, Number):
        raise NotImplementedError  # TODO
    raise TypeError(f'Unsupported type for exp: {type(t)}')


def log(t: AbstractTensor | complex | float, legs1: list[int | str] = None,
        legs2: list[int | str] = None) -> AbstractTensor | complex | float:
    """
    The (natural) logarithm of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise logarithm function.
    """
    if isinstance(t, Tensor):
        return _act_block_diagonal_square_matrix(t, legs1, legs2, 'matrix_log')
    if isinstance(t, DiagonalTensor):
        raise NotImplementedError  # TODO
    if isinstance(t, Number):
        raise NotImplementedError  # TODO
    raise TypeError(f'Unsupported type for log: {type(t)}')


def _act_block_diagonal_square_matrix(t: AbstractTensor,
                                      legs1: list[int | str],
                                      legs2: list[int | str],
                                      block_method: str) -> AbstractTensor:
    """

    block_method :
        Name of a BlockBackend method with signature ``block_method(a: Block) -> Block``.
    """
    idcs1, idcs2 = leg_bipartition(t, legs1, legs2)
    assert len(idcs1) == len(idcs2)
    assert all(t.legs[i1].is_dual_of(t.legs[i2]) for i1, i2 in zip(idcs1, idcs2))
    if len(idcs1) > 1:
        pipe = t.make_ProductSpace(idcs1)
        t = t.combine_legs([idcs1, idcs2], new_legs=[pipe.dual, pipe], new_axes=[0, 1])
    res_data = t.backend.act_block_diagonal_square_matrix(t, block_method)
    res = Tensor(res_data, backend=t.backend, legs=t.legs, labels=t.labels)
    if len(idcs1) > 1:
        res = res.split_legs()
        transposed = idcs1 + idcs2
        if any(i != j for i, j in enumerate(transposed)):
            res = res.permute_legs(inverse_permutation(transposed))
    return res


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
        The total sum of all discared Schmidt values squared.
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
