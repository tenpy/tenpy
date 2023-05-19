# Copyright 2023-2023 TeNPy Developers, GNU GPLv3

from __future__ import annotations
from .tensors import DiagonalTensor, AbstractTensor, Tensor, get_same_backend
from ..tools.misc import inverse_permutation

__all__ = ['svd', 'truncate_svd', 'svd_split', 'leg_bipartition', 'exp', 'log']


def svd(a: AbstractTensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
        new_labels: tuple[str, ...] = None, new_vh_leg_dual=False,
        options=None
        ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor]:
    """SVD of a tensor, viewed as a linear map (i.e. matrix) from one set of its legs to the rest.

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
    options : dict | Config | None
        TODO: things like driver / stable / ... ?

    Returns
    -------
    tuple[Tensor, DiagonalTensor, Tensor]
        The tensors U, S, Vh that form the SVD, such that `tdot(U, tdot(S, Vh, 1, 0), -1, 0)` is equal
        to `a` up to numerical precsision
    """
    if not isinstance(a, Tensor):
        raise NotImplementedError  # TODO ChargedTensor support

    u_idcs, vh_idcs = leg_bipartition(a, u_legs, vh_legs)
    l_u, l_su, l_sv, l_vh = _svd_new_labels(new_labels)

    need_combine = (len(u_idcs) != 1 or len(vh_idcs) != 1)
    if need_combine:
        a = a.combine_legs(u_idcs, vh_idcs, new_axes=[0, 1])
    elif u_idcs[0] == 1:   # both single entry, so v_idcs = [1]
        a = a.permute_legs([1, 0])

    # TODO read algorithm etc from config
    u_data, s_data, vh_data, new_leg = a.backend.svd(a, new_vh_leg_dual)

    U = Tensor(u_data, backend=a.backend, legs=[a.legs[0], new_leg.dual])
    # TODO revisit this once DiagonalTensor is defined
    S = DiagonalTensor(s_data, backend=a.backend, legs=[new_leg, new_leg.dual], labels=[l_su, l_sv])
    Vh = Tensor(vh_data, backend=a.backend, legs=[new_leg, a.legs[1]])
    if need_combine:
        U = U.split_leg(0)
        Vh = Vh.split_leg(1)
    U.set_labels([a.labels[n] for n in u_idcs] + [l_u])
    Vh.set_labels([l_vh] + [a.labels[n] for n in vh_idcs])
    return U, S, Vh


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


# TODO directly provide truncated SVD


def truncate_svd(U: AbstractTensor, S: DiagonalTensor, Vh: AbstractTensor, options=None
                 ) -> tuple[AbstractTensor, DiagonalTensor, AbstractTensor, float]:
    """Truncate an SVD decomposition

    Returns
    -------
    U, S, Vh, trunc_err
    """
    if not isinstance(U, Tensor) or not isinstance(Vh, Tensor):
        raise NotImplementedError

    backend = get_same_backend(U, S, Vh)
    # TODO implement backend.truncate_svd
    u_data, s_data, vh_data, new_leg, trunc_err = backend.truncate_svd(U, S, Vh, options)
    U = Tensor(u_data, backend=backend, legs=U.legs[:-1] + [new_leg], labels=U.labels)
    # TODO revisit this once DiagonalTensor is defined
    S = DiagonalTensor(s_data, backend=backend, legs=[new_leg.dual, new_leg], labels=S.labels)
    Vh = Tensor(vh_data, backend=backend, legs=[new_leg.dual] + Vh.legs[1:], labels=Vh.labels)
    return U, S, Vh, trunc_err


def svd_split(a: AbstractTensor, legs1: list[int | str] = None, legs2: list[int | str] = None,
              new_labels: tuple[str, str] = None, options=None, s_exponent: float = .5):
    """Split a tensor via (truncated) svd,
    i.e. compute (U @ S ** s_exponent) and (S ** (1 - s_exponent) @ Vh)"""
    raise NotImplementedError  # TODO


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
    return _act_block_diagonal_square_matrix(t, legs1, legs2, 'matrix_exp')


def log(t: AbstractTensor | complex | float, legs1: list[int | str] = None,
        legs2: list[int | str] = None) -> AbstractTensor | complex | float:
    """
    The (natural) logarithm of t, viewed as a linear map from legs1 to legs2.
    Requires the two groups of legs to be mutually dual.
    Contrary to numpy, this is *not* the element-wise exponential function.
    """
    return _act_block_diagonal_square_matrix(t, legs1, legs2, 'matrix_log')


def _act_block_diagonal_square_matrix(t: AbstractTensor,
                                      legs1: list[int | str],
                                      legs2: list[int | str],
                                      block_method: str) -> AbstractTensor:
    """

    block_method :
        Name of a BlockBackend method with signature ``block_method(a: Block) -> Block``.
    """
    idcs1, idcs2 = leg_bipartition(legs1, legs2)
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
