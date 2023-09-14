# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
import warnings
from numbers import Number
from .spaces import ProductSpace
from .tensors import DiagonalTensor, AbstractTensor, Tensor, Mask, ChargedTensor, tdot, _dual_leg_label
from ..tools.misc import inverse_permutation, to_iterable
from ..tools.params import asConfig

__all__ = ['svd', 'svd_apply_mask', 'truncated_svd', 'truncate_singular_values', 'qr', 'svd_split',
           'pinv', 'leg_bipartition', 'exp', 'log']


def svd(a: AbstractTensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
        new_labels: tuple[str, ...] = None, new_vh_leg_dual=False, U_inherits_charge: bool = False,
        options={}
        ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor]:
    """Singular value decomposition of a tensor.

    The tensor is viewed as a linear map (i.e. matrix) from one set of its legs to the rest.
    Returns three tensors `U`, `S`, `Vh` such that::
    
      - `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is equal to `a` up to numerical precision
      - `U` is an isometry, i.e. `tdot(U, conj(U), legs, legs)` is the identity (`legs` are all legs but `-1`)
      - `S` is diagonal with real, non-negative entries
      - `Vh` is an isometry, i.e. `tdot(Vh, conj(Vh), legs, legs)` is the identity (`legs` are all legs but `0`)

    TODO: document input format for u_legs / vh_legs, can probably to it centrally for all matrix ops

    TODO: document efficiency issues when combining legs

    Parameters
    ----------
    a : Tensor | ChargedTensor | DiagonalTensor
        The tensor to decompose.
        If `a` is a :class:`~tenpy.linalg.tensors.Tensor`, both `U` and `Vh` are `Tensor`s.
        If `a` is a :class:`~tenpy.linalg.tensors.ChargedTensor`, so is one of `U` and `Vh`,
        while the other is a ``Tensor``, see `U_inherits_charge`.
        If `a` is a DiagonalTensor, so are `U` and `Vh`.
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
    U_inherits_charge : bool
        If `a` is a :class:`~tenpy.linalg.tensors.ChargedTensor`, this flag controls which of the
        two resulting isometries, `U` or `Vh` will carry the charge.
        It will be a :class:`~tenpy.linalg.tensors.ChargedTensor` while the other is a :class:`~tenpy.linalg.tensors.Tensor`
    options : dict | Config
        TODO proper Config style doc
        algorithm : str | None

    Returns
    -------
    U : Tensor | ChargedTensor | DiagonalTensor
    S : DiagonalTensor
    Vh : Tensor | ChargedTensor | DiagonalTensor
    """
    u_idcs, vh_idcs = leg_bipartition(a, u_legs, vh_legs)
    
    if isinstance(a, ChargedTensor):
        if U_inherits_charge:
            u_idcs.append(-1)
        else:
            vh_idcs.append(-1)
        U, S, Vh = svd(a.invariant_part, u_legs=u_idcs, vh_legs=vh_idcs, new_labels=new_labels,
                       new_vh_leg_dual=new_vh_leg_dual)
        if U_inherits_charge:
            perm = list(range(U.num_legs))
            perm[-2:] = [-1, -2]
            U = ChargedTensor(U.permute_legs(perm), dummy_leg_state=a.dummy_leg_state)
        else:
            Vh = ChargedTensor(Vh, dummy_leg_state=a.dummy_leg_state)
        return U, S, Vh
    
    l_u, l_su, l_sv, l_vh = _svd_new_labels(new_labels)
    
    if isinstance(a, DiagonalTensor):
        S = abs(a)
        U = a / S  # the phases of a
        Vh = DiagonalTensor.eye(first_leg=a.legs[0], backend=a.backend)
        # all legs are either equal or dual, so we can just set them
        if a.second_leg_dual:
            if new_vh_leg_dual == a.legs[0].is_dual:
                new_vh_leg = a.legs[0]
                new_u_leg = a.legs[1]
            else:
                new_vh_leg = a.legs[1]
                new_u_leg = a.legs[0]
        elif new_vh_leg_dual == a.legs[0].is_dual:
            new_vh_leg = a.legs[0]
            new_u_leg = a.legs[0].dual
        else:
            new_vh_leg = a.legs[0].dual
            new_u_leg = a.legs[0]
        U.legs = [a.legs[u_idcs[0]], new_u_leg]
        U.second_leg_dual = a.legs[u_idcs[0]].is_dual != new_u_leg.is_dual
        S.legs = [new_vh_leg, new_u_leg]
        S.second_leg_dual = True
        Vh.legs = [new_vh_leg, a.legs[vh_idcs[0]]]
        Vh.second_leg_dual = new_vh_leg.is_dual != a.legs[vh_idcs[0]].is_dual
        # set labels
        U.set_labels([a.labels[u_idcs[0]], l_u])
        S.set_labels([l_su, l_sv])
        Vh.set_labels([l_vh, a.labels[vh_idcs[0]]])
        return U, S, Vh
        
    if isinstance(a, Tensor):
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

    raise TypeError(f'svd not supported for {type(a)}')


def svd_apply_mask(U: AbstractTensor, S: DiagonalTensor, Vh: AbstractTensor, mask: Mask
                   ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor]:
    """Truncate an existing SVD"""
    U = U.apply_mask(mask, -1)
    S = S._apply_mask_both_legs(mask)
    Vh = Vh.apply_mask(mask, 0)
    return U, S, Vh


def truncated_svd(a: AbstractTensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
                  new_labels: tuple[str, ...] = None, new_vh_leg_dual=False, U_inherits_charge: bool = False,
                  normalize_to: float = None, options={}, truncation_options={}
                  ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor, float, float]:
    """Truncated singular value decomposition of a tensor.

    The tensor is viewed as a linear map (i.e. matrix) from one set of its legs to the rest.

    Parameters
    ----------
    same as for :meth:`svd`
    normalize_to: float or None
        If ``None`` (default), the resulting singular values are not renormalized,
        resulting in an approximation in terms of ``U, S, Vh`` which has smaller norm than `a`.
        If a ``float``, the singular values are scaled such that ``norm(S) == normalize_to``.
    truncation_options : dict-like
        Options that determine how many singular values are kept, see :cfg:config:`truncation`.

    Returns
    -------
    U, S, Vh
        The tensors U, S, Vh that form the truncated SVD, such that
        `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is *aproximately* equal to `a`.
    err : float
        The relative 2-norm truncation error ``norm(a - U_S_Vh) / norm(a)``.
        This is the (relative) 2-norm weight of the discarded singular values.
    renormalization : float
        Factor, by which `S` was renormalized.
    """
    U, S, V = svd(a, u_legs=u_legs, vh_legs=vh_legs, new_labels=new_labels, new_vh_leg_dual=new_vh_leg_dual,
                  U_inherits_charge=U_inherits_charge, options=options)
    S_norm = S.norm()
    mask, new_norm, err = truncate_singular_values(S / S_norm, options=truncation_options)
    U, S, V = svd_apply_mask(U, S, V, mask)
    if normalize_to is None:
        renormalize = 1
    else:
        # norm(S[mask]) == S_norm * new_norm
        renormalize = normalize_to / S_norm / new_norm
        S = S._mul_scalar(renormalize)
    return U, S, V, err, renormalize
    

def truncate_singular_values(S: DiagonalTensor, options) -> Mask:
    r"""Given *normalized* singular values, determine which to keep.

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
    err : float
        the truncation error introduced, i.e. the norm(S_discarded) = sqrt(\sum_{i=k}^{N} S_i^2).
        In the context of truncated SVD, this is the relative error in the 2-norm,
        i.e. ``norm(T - T_approx) / norm(T)``.
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
    S_np = S.diag_numpy

    if trunc_cut is not None and trunc_cut >= 1.:
        raise ValueError("trunc_cut >=1.")
    if not np.any(S_np > 1.e-10):
        warnings.warn("no singular value above 1.e-10", stacklevel=2)
    if np.any(S_np < -1.e-10):
        warnings.warn("negative singular values!", stacklevel=2)

    # use 1.e-100 as replacement for <=0 values for a well-defined logarithm.
    logS = np.log(np.choose(S_np <= 0., [S_np, 1.e-100 * np.ones(len(S_np))]))
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
        good2 = (np.cumsum(S_np[piv]**2) > trunc_cut * trunc_cut)
        good = _combine_constraints(good, good2, "trunc_cut")

    cut = np.nonzero(good)[0][0]  # smallest possible cut: keep as many S as allowed
    mask = np.zeros(len(S_np), dtype=np.bool_)
    np.put(mask, piv[cut:], True)
    new_norm = np.linalg.norm(S_np[mask])
    err = np.linalg.norm(S_np[np.logical_not(mask)])
    
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
        R = R.split_legs(1)
        Q = Q.split_legs(0)
    Q.set_labels([a_labels[n] for n in q_idcs] + [new_labels[0]])
    R.set_labels([new_labels[1], *[a_labels[n] for n in r_idcs]])
    return Q, R


def svd_split(a: AbstractTensor, legs1: list[int | str] = None, legs2: list[int | str] = None,
              new_labels: tuple[str, str] = None, options=None, s_exponent: float = .5):
    """Split a tensor via (truncated) svd,
    i.e. compute (U @ S ** s_exponent) and (S ** (1 - s_exponent) @ Vh)"""
    raise NotImplementedError  # TODO


def pinv(a: AbstractTensor, legs1: list[int | str] = None, legs2: list[int | str] = None, cutoff=1.e-15):
    """The Moore-Penrose pseudo-inverse of a tensor.

    The tensor is viewed as a linear map from (the duals of) `legs1` to `legs2`.
    The resulting pseudo-inverse is a map from (the duals of) `legs2` to `legs1`,
    i.e. it can be composed with `a`.
    """
    # TODO (JU) tests
    legs1, legs2 = leg_bipartition(a, legs1, legs2)
    U, S, Vh = svd(a, u_legs=legs1, vh_legs=legs2)
    P = tdot(U, tdot(1. / S, Vh, 1, 0), -1, 0)
    return P.conj().permute_legs([*legs2, *legs1])


def eigh(a: AbstractTensor, legs1: list[int | str] = None, legs2: list[int | str] = None,
         new_labels: str | list[str] = None):
    r"""Eigenvalue decomposition of a hermitian square tensor.

    A tensor is considered square, if the `legs2` legs are the duals of the `legs1` legs,
    i.e. if the tensor can be viewed as a linear map from a space (the product of those legs) to itself.
    It is considered hermitian, if this linear map is hermitian.

    The eigenvalue decomposition is :math:`a = U \cdot D \cdot U^\dagger`, i.e.
    ``a == tdot(U, tdot(D, U.conj(), 1, 0), -1, 0)``.
    where `U` is (up to combining/splitting legs) a unitary and `D` is diagonal, containing the eigenvalues.

    Parameters
    ----------
    a : Tensor
        The tensor two decompose. Has ``2 * N`` where the ``N`` legs specified by `legs1`
        are duals of those specified by `legs2`.
        Reshaped to a matrix (by combining `legs1` and `legs2`), `a` is assumed to be hermitian.
        This is not checked.
    legs1
        Which of the legs belong to "matrix rows"
    legs2
        Which of the legs belong to "matrix columns"
    new_labels: str or tuple of two str, optional
        Either two labels, where `new_labels[0]` is used for the last leg of `U` and `D`, while
        `new_labels[1]` is used for the first leg of `D`.
        If only a single label, the second is the "dual of" the given one.

    Returns
    -------
    D : DiagonalTensor
        The eigenvalues as a DiagonalTensor with legs ``[new_leg, new_leg.dual]``.
    U : Tensor
        A tensor containing the normalized eigenvectors. It is unitary, in the sense that combining
        the leading legs yields a unitary matrix. Legs are ``[*a.get_legs(legs1), new_leg.dual]``,
        where ``new_leg = ProductSpace(*a.get_legs(leg1))``.
    """
    # TODO (JU) should we support `UPLO` arg? (use lower or upper triangular part)
    # TODO (JU) should we support `sort` arg? (how to sort eigenvalues *within* charge blocks)
    #           if not, should we fix a canonical order? -> enforce it in block_eigh for all backends.
    if not isinstance(a, Tensor):
        raise TypeError(f'eigh not supported for type {type(a)}')
    idcs1, idcs2 = leg_bipartition(a, legs1, legs2)
    assert len(idcs1) == len(idcs2)
    incompatible = [(i1, i2) for i1, i2 in zip(idcs1, idcs2) if not a.legs[i1].can_contract_with(a.legs[i2])]
    if incompatible:
        raise ValueError(f'Incompatible leg pairs: {", ".join(map(str, incompatible))}')
    need_combine = (len(idcs1) > 1)
    backend = a.backend
    if need_combine:
        new_leg = ProductSpace([a.legs[i1] for i1 in idcs1], backend=backend)
        a = a.combine_legs(idcs1, idcs2, product_spaces=[new_leg, new_leg.dual])
    else:
        new_leg = a.legs[0]

    if new_labels is None:
        new_labels = a.labels[::-1]
    new_labels = list(to_iterable(new_labels))
    if len(new_labels) == 1:
        new_labels = [new_labels[0], _dual_leg_label(new_labels[0])]
    assert len(new_labels) == 2
    d_data, u_data = backend.eigh(a)
    D = DiagonalTensor(d_data, first_leg=new_leg, second_leg_dual=True, backend=backend, labels=new_labels[::-1])
    U = Tensor(u_data, legs=[new_leg, new_leg.dual], backend=backend, labels=[a.labels[0], new_labels[0]])
    U = U.split_legs(0)
    return D, U


def _combine_constraints(good1, good2, warn):
    """return logical_and(good1, good2) if there remains at least one `True` entry.

    Otherwise print a warning and return just `good1`.
    """
    assert good1.shape == good2.shape, f'{good1.shape} != {good2.shape}'
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
    assert all(t.legs[i1].can_contract_with(t.legs[i2]) for i1, i2 in zip(idcs1, idcs2))
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
